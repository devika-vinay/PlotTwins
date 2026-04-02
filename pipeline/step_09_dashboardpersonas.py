import pandas as pd
import numpy as np

from pipeline.step_00_config import CACHE_DIR


FEATURE_MATRIX_IN = CACHE_DIR / "feature_matrix.parquet"
CLUSTER_ASSIGNMENTS_IN = CACHE_DIR / "cluster_assignments.parquet"
USER_MATCHES_IN = CACHE_DIR / "user_matches.parquet"
CLUSTER_INTERPRETATION_IN = CACHE_DIR / "cluster_interpretation.parquet"
CLUSTER_MOVIE_KB_IN = CACHE_DIR / "cluster_movie_kb.parquet"

CLUSTER_PERSONAS_OUT = CACHE_DIR / "cluster_personas.parquet"
USER_DASHBOARD_OUT = CACHE_DIR / "user_dashboard.parquet"



def build_persona_library_from_kb() -> dict:
    """Returns {cluster_id: persona_dict} built from step_07/08 KB artifacts."""
    interp = pd.read_parquet(CLUSTER_INTERPRETATION_IN)
    movie_kb = pd.read_parquet(CLUSTER_MOVIE_KB_IN)

    personas = {}
    for _, row in interp.iterrows():
        cid = int(row["cluster"])
        top_genres = [g.strip() for g in str(row["distinctive_genres"]).split(" | ") if g.strip()]
        top_decades = [d.strip() for d in str(row["distinctive_decades"]).split(" | ") if d.strip()]
        dominant_pop = row["dominant_pop_tier"]

        cluster_movies = movie_kb[movie_kb["cluster"] == cid].copy()
        top_movies = (
            cluster_movies
            .sort_values(["representative_score", "n_ratings"], ascending=[False, False])
            ["display_title"]
            .dropna()
            .head(3)
            .tolist()
        )

        name = (
            f"{top_genres[0]} & {top_genres[1]} Viewer"
            if len(top_genres) >= 2 else
            top_genres[0] if top_genres else f"Cluster {cid}"
        )

        interpretation = (
            f"You gravitate toward {', '.join(top_genres[:3])} films, "
            f"mostly from the {top_decades[0]}s, with a "
            f"{str(dominant_pop).replace('pop_', '')} popularity preference."
            if top_decades and top_genres else
            f"You gravitate toward {', '.join(top_genres[:3])} films."
            if top_genres else
            f"Cluster {cid} profile."
        )

        personas[cid] = {
            "persona_key": f"cluster_{cid}",
            "persona_name": name,
            "short_label": name.replace("Viewer", "Fans"),
            "expected_genres": top_genres,
            "expected_eras": [f"era_{d}.0" for d in top_decades],
            "expected_pop": dominant_pop,
            "interpretation": interpretation,
            "example_movies": top_movies,
            "top_genres": top_genres,
            "top_eras": top_decades,
        }

    return personas


def safe_top_from_prefix(row: pd.Series, prefix: str, top_n: int = 3):
    cols = [c for c in row.index if c.startswith(prefix)]
    if not cols:
        return []
    values = row[cols].sort_values(ascending=False).head(top_n)
    cleaned = []
    for col in values.index:
        cleaned.append(col.replace(prefix, ""))
    return cleaned


def describe_popularity(row: pd.Series):
    pop_cols = [c for c in ["pop_low", "pop_mid", "pop_high"] if c in row.index]
    if not pop_cols:
        return "unknown"
    top = row[pop_cols].sort_values(ascending=False).index[0]
    return top


def describe_behavior(row: pd.Series):
    like_rate = row.get("like_rate", np.nan)
    dislike_rate = row.get("dislike_rate", np.nan)

    if pd.notna(like_rate) and pd.notna(dislike_rate):
        if like_rate >= 0.50 and dislike_rate <= 0.20:
            return "strongly positive"
        if like_rate >= 0.35 and dislike_rate <= 0.30:
            return "slightly positive"
        if dislike_rate >= 0.40:
            return "more critical"
        return "neutral"
    if pd.notna(like_rate):
        if like_rate >= 0.55:
            return "strongly positive"
        if like_rate >= 0.35:
            return "slightly positive"
        if like_rate <= 0.30:
            return "more critical"
        return "neutral"
    return "unknown"


def summarize_cluster(row: pd.Series):
    return {
        "top_genres": safe_top_from_prefix(row, "genre_mean_", top_n=5),
        "top_eras": safe_top_from_prefix(row, "era_", top_n=3),
        "popularity_pref": describe_popularity(row),
        "behavior_profile": describe_behavior(row),
    }


def build_user_taste_summary(row: pd.Series):
    return {
        "user_top_genres": ", ".join(safe_top_from_prefix(row, "genre_mean_", top_n=3)),
        "user_top_eras": ", ".join(safe_top_from_prefix(row, "era_", top_n=2)),
        "user_popularity_pref": describe_popularity(row),
        "user_behavior_profile": describe_behavior(row),
    }


def main():
    if CLUSTER_PERSONAS_OUT.exists() and USER_DASHBOARD_OUT.exists():
        print("[09_dashboard_personas] Cache exists. Skipping persona mapping.")
        return
    
    if not CLUSTER_INTERPRETATION_IN.exists() or not CLUSTER_MOVIE_KB_IN.exists():
        raise FileNotFoundError(
            "[09_dashboard_personas] KB artifacts missing. "
            "Run step_07_cluster_kb and step_08_movies_kb first."
        )

    features = pd.read_parquet(FEATURE_MATRIX_IN)
    cluster_assignments = pd.read_parquet(CLUSTER_ASSIGNMENTS_IN)

    df = features.merge(
        cluster_assignments[["user", "cluster"]],
        on="user",
        how="left"
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feature_cols = [c for c in numeric_cols if c not in ["cluster"]]

    cluster_means = df.groupby("cluster")[numeric_feature_cols].mean()

    # Build personas directly from KB — no hardcoding
    personas = build_persona_library_from_kb()

    cluster_persona_rows = []
    for cluster_id, row in cluster_means.iterrows():
        cid = int(cluster_id)
        summary = summarize_cluster(row)
        persona = personas[cid]

        cluster_persona_rows.append({
            "cluster": cid,
            "persona_key": persona["persona_key"],
            "persona_name": persona["persona_name"],
            "short_label": persona["short_label"],
            "interpretation": persona["interpretation"],
            "example_movies": " | ".join(persona["example_movies"]),
            "top_genres": " | ".join(persona["top_genres"]),
            "top_eras": " | ".join(persona["top_eras"]),
            "popularity_pref": summary["popularity_pref"],
            "behavior_profile": summary["behavior_profile"],
        })

    
    cluster_personas = pd.DataFrame(cluster_persona_rows)

    user_summary_rows = []
    for _, row in df.iterrows():
        summary = build_user_taste_summary(row)
        user_summary_rows.append({
            "user": row["user"],
            "cluster": row["cluster"],
            "fsa": row.get("fsa"),
            "region": row.get("region"),
            **summary,
        })

    user_dashboard = pd.DataFrame(user_summary_rows).merge(
        cluster_personas, on="cluster", how="left"
    )

    user_dashboard.to_parquet(USER_DASHBOARD_OUT, index=False)
    cluster_personas.to_parquet(CLUSTER_PERSONAS_OUT, index=False)

    print("[09_dashboard_personas] Saved:", CLUSTER_PERSONAS_OUT)
    print("[09_dashboard_personas] Saved:", USER_DASHBOARD_OUT)


if __name__ == "__main__":
    main()