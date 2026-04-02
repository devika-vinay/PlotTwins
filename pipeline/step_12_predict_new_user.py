"""
step_12_predict_new_user.py
─────────────────────────────────────────────────────────────────────────
Two-path prediction:

  COLD-START  (new users, no rating history)
    Uses mini model from step_04b — 40 questionnaire-elicitable features.
    Input: answers to 7 questionnaire questions (plain Python types).
    Cold cluster ID is translated to full-model cluster ID via mapping
    table saved by step_04b — ensures persona lookup is always valid.

  WARM        (returning users, ≥20 ratings)
    Uses full model from step_04/05 — 113 features including real SVD.
    Input: all feature blocks including real SVD vector.

Routing:
    predict_new_user()  → cold-start path  (questionnaire input)
    predict_warm_user() → warm path        (full feature blocks)

Both return identical dict structure for clean API consumption.
─────────────────────────────────────────────────────────────────────────
"""

import joblib
import pandas as pd
import numpy as np
from pipeline.step_00_config import CACHE_DIR

# ── Shared artifacts ──────────────────────────────────────────────────
interpretation = pd.read_parquet(CACHE_DIR / "cluster_interpretation.parquet")
feature_matrix = pd.read_parquet(CACHE_DIR / "feature_matrix.parquet")
cluster_personas = pd.read_parquet(CACHE_DIR / "cluster_personas.parquet")

# ── Full model artifacts (warm path — step_04 / step_05) ──────────────
scalers = joblib.load(CACHE_DIR / "scalers.pkl")
selector_full = joblib.load(CACHE_DIR / "variance_selector.pkl")
pca_full = joblib.load(CACHE_DIR / "pca.pkl")
kmeans_full = joblib.load(CACHE_DIR / "kmeans_model.pkl")

BLOCK_NAMES = ["svd", "genre_mean", "genre_share", "popularity", "era", "profiles"]
feature_columns = {
    name: pd.read_parquet(
        CACHE_DIR / f"feature_columns_{name}.parquet"
    )["feature"].tolist()
    for name in BLOCK_NAMES
}

SVD_COLS = feature_columns["svd"]
SVD_MEAN_ROW = feature_matrix[SVD_COLS].mean().values.reshape(1, -1)
PROFILES_TRAIN_MEAN = feature_matrix[feature_columns["profiles"]].mean()

# ── Cold-start model artifacts (step_04b) ────────────────────────────
coldstart_cols = pd.read_parquet(
    CACHE_DIR / "coldstart_feature_columns.parquet"
)["feature"].tolist()

scaler_cold = joblib.load(CACHE_DIR / "coldstart_scaler.pkl")
selector_cold = joblib.load(CACHE_DIR / "coldstart_selector.pkl")
pca_cold = joblib.load(CACHE_DIR / "coldstart_pca.pkl")
kmeans_cold = joblib.load(CACHE_DIR / "coldstart_kmeans.pkl")

# cold cluster ID → full model cluster ID (majority vote mapping from step_04b)
coldstart_cluster_map = pd.read_parquet(
    CACHE_DIR / "coldstart_cluster_map.parquet"
).set_index("cold")["full"].to_dict()

COLDSTART_TRAIN_MEAN = feature_matrix[coldstart_cols].mean()


# ─────────────────────────────────────────────────────────────────────
# COLD-START PATH
# ─────────────────────────────────────────────────────────────────────
def predict_new_user(
        genres: list[str],  # Q1: e.g. ["Animation", "Comedy", "Family"]
        decades: list[int],  # Q2: e.g. [2010, 2020]
        popularity: str,  # Q3: "low" | "mid" | "high"
        like_rate: float,  # Q4: 0.0–1.0  e.g. 0.7
        era_preference: str,  # Q5: "modern" | "classic" | "mixed"
        english_preference: float,  # Q6: 0.0–1.0  e.g. 0.85
        genre_breadth: int,  # Q7: number of genres user enjoys
) -> dict:
    """
    Cold-start prediction from 7 questionnaire answers.
    Returns full result dict with persona, interpretation, and example movies.
    """
    # Start from training means — unset features stay near population center
    row = COLDSTART_TRAIN_MEAN.to_dict()

    # Q1 — genre_share: distribute evenly across selected genres
    genre_share_cols = [c for c in coldstart_cols if c.startswith("genre_share_")]
    for c in genre_share_cols:
        row[c] = 0.0
    if genres:
        share_per_genre = 1.0 / len(genres)
        for g in genres:
            key = f"genre_share_{g}"
            if key in row:
                row[key] = share_per_genre

    # Q2 — era: distribute evenly across selected decades
    era_cols = [c for c in coldstart_cols if c.startswith("era_")]
    for c in era_cols:
        row[c] = 0.0
    if decades:
        share_per_era = 1.0 / len(decades)
        for d in decades:
            key = f"era_{float(d)}"
            if key in row:
                row[key] = share_per_era

    # Q3 — popularity tier (one-hot style)
    for tier in ["pop_low", "pop_mid", "pop_high"]:
        if tier in row:
            row[tier] = 0.0
    pop_key = f"pop_{popularity}"
    if pop_key in row:
        row[pop_key] = 1.0

    # Q4 — like rate
    if "like_rate" in row:
        row["like_rate"] = like_rate

    # Q5 — era preference → modern_share / classic_share
    era_map = {
        "modern": {"modern_share": 0.75, "classic_share": 0.05},
        "classic": {"modern_share": 0.05, "classic_share": 0.75},
        "mixed": {"modern_share": 0.35, "classic_share": 0.35},
    }
    for k, v in era_map.get(era_preference, era_map["mixed"]).items():
        if k in row:
            row[k] = v

    # Q6 — english_share
    if "english_share" in row:
        row["english_share"] = english_preference

    # Q7 — genre breadth
    if "n_unique_genres" in row:
        row["n_unique_genres"] = float(genre_breadth)

    # ── Transform through cold-start pipeline ─────────────────────────
    X_raw = pd.DataFrame([row])[coldstart_cols].fillna(0)

    X_scaled = pd.DataFrame(
        scaler_cold.transform(X_raw),
        columns=coldstart_cols
    )

    selected_cols = X_scaled.columns[selector_cold.get_support()]
    X_filtered = pd.DataFrame(
        selector_cold.transform(X_scaled.values),  # .values avoids feature name warning
        columns=selected_cols
    )

    X_pca = pca_cold.transform(X_filtered.values)
    raw_cluster = int(kmeans_cold.predict(X_pca)[0])

    # Translate cold-start cluster ID → full model cluster ID
    cluster_id = coldstart_cluster_map.get(raw_cluster, raw_cluster)

    return _build_result(cluster_id, model="cold-start")


# ─────────────────────────────────────────────────────────────────────
# WARM PATH
# ─────────────────────────────────────────────────────────────────────
def predict_warm_user(user_blocks: dict) -> dict:
    """
    Full-model prediction for returning users with ≥20 ratings.
    SVD block should contain real latent vectors from rating history.
    """
    scaled_blocks = []

    for name in BLOCK_NAMES:
        if name == "svd":
            # Use real SVD if provided, else fall back to training mean
            if user_blocks.get("svd") is not None:
                block = user_blocks["svd"].reindex(columns=SVD_COLS, fill_value=0)
            else:
                block = pd.DataFrame(SVD_MEAN_ROW, columns=SVD_COLS)
        else:
            block = user_blocks[name].reindex(
                columns=feature_columns[name], fill_value=0
            )

        scaled = pd.DataFrame(
            scalers[name].transform(block),
            columns=feature_columns[name]
        )
        scaled_blocks.append(scaled)

    X_scaled = pd.concat(scaled_blocks, axis=1).fillna(0)

    selected_cols = X_scaled.columns[selector_full.get_support()]
    X_filtered = pd.DataFrame(
        selector_full.transform(X_scaled),
        columns=selected_cols
    )

    X_pca = pd.DataFrame(
        pca_full.transform(X_filtered.values),
        columns=[f"pca_{i}" for i in range(pca_full.n_components_)]
    )

    cluster_id = int(kmeans_full.predict(X_pca)[0])
    return _build_result(cluster_id, model="warm")


# ─────────────────────────────────────────────────────────────────────
# SHARED RESULT BUILDER
# ─────────────────────────────────────────────────────────────────────
def _build_result(cluster_id: int, model: str) -> dict:
    interp = interpretation[interpretation["cluster"] == cluster_id].iloc[0]
    persona = cluster_personas[cluster_personas["cluster"] == cluster_id].iloc[0]

    return {
        "cluster": cluster_id,
        "model_used": model,
        "persona_name": persona["persona_name"],
        "interpretation": persona["interpretation"],
        "example_movies": persona["example_movies"],
        "distinctive_genres": interp["distinctive_genres"],
        "distinctive_decades": interp["distinctive_decades"],
        "dominant_pop_tier": interp["dominant_pop_tier"],
        "classic_share_diff": round(float(interp["classic_share_diff"]), 3),
        "modern_share_diff": round(float(interp["modern_share_diff"]), 3),
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN — Demo both paths
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 60)
    print("  CLUSTER MAP  (cold-start → full model)")
    print("=" * 60)
    for cold, full in coldstart_cluster_map.items():
        print(f"  cold {cold} → full {full}")

    # ── Path 1: cold-start ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PATH 1 — COLD-START  (questionnaire input)")
    print("=" * 60)

    result_cold = predict_new_user(
        genres=["Animation", "Comedy", "Family"],  # Q1
        decades=[2010, 2020],  # Q2
        popularity="low",  # Q3
        like_rate=0.65,  # Q4
        era_preference="modern",  # Q5
        english_preference=0.85,  # Q6
        genre_breadth=4,  # Q7
    )
    for k, v in result_cold.items():
        print(f"  {k:<22} : {v}")

    # ── Path 2: warm ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PATH 2 — WARM  (returning user with rating history)")
    print("=" * 60)

    sample_user = feature_matrix.iloc[0]
    warm_blocks = {
        name: pd.DataFrame(
            [sample_user[feature_columns[name]].values],
            columns=feature_columns[name]
        )
        for name in BLOCK_NAMES
    }

    result_warm = predict_warm_user(warm_blocks)
    for k, v in result_warm.items():
        print(f"  {k:<22} : {v}")

    # ── Sanity check: baseline cold-start (all at training mean) ──────
    print("\n" + "=" * 60)
    print("  SANITY CHECK — cold-start baseline vs user input")
    print("=" * 60)

    result_baseline = predict_new_user(
        genres=[],  # no genre signal
        decades=[],  # no era signal
        popularity="mid",  # neutral
        like_rate=float(COLDSTART_TRAIN_MEAN.get("like_rate", 0.5)),
        era_preference="mixed",
        english_preference=float(COLDSTART_TRAIN_MEAN.get("english_share", 0.5)),
        genre_breadth=int(COLDSTART_TRAIN_MEAN.get("n_unique_genres", 5)),
    )
    print(f"  Baseline cluster   : {result_baseline['cluster']} "
          f"({result_baseline['persona_name']})")
    print(f"  User input cluster : {result_cold['cluster']} "
          f"({result_cold['persona_name']})")
    print(
        f"\n  {'✅ Inputs are driving the prediction' if result_cold['cluster'] != result_baseline['cluster'] else '⚠️  Still same cluster — cold model may need retraining with more components'}"
    )