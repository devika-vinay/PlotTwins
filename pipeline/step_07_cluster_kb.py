"""
step_07_cluster_kb.py
─────────────────────────────────────────────────────────────────────────
KNOWLEDGE BASE 1 — CLUSTER BEHAVIORAL PROFILES

Answers: "Who are the people in each cluster?"

Purpose: WHO are the clusters — behavioral identity, raw percentages, and distinctiveness signals

Two complementary views are produced:

  A) DISTINCTIVENESS (z-score diff from population average)
     Source: feature_matrix.parquet (StandardScaler space)
     Tells you WHAT MAKES each cluster unique — e.g. Cluster 4 is
     +2.97 above average on classic_share, meaning they watch far
     more classic films than the typical user. Used for persona naming
     and LLM prompt generation.

  B) RAW PROFILE (actual percentages from interaction data)
     Source: transformed.parquet (raw ratings)
     Tells you WHAT THEY ACTUALLY WATCH in absolute terms —
     e.g. 22% of Cluster 4's interactions are 1950s films.
     Used for questionnaire design and user-facing descriptions.

Outputs (all saved to CACHE_DIR):
  ┌─ cluster_interpretation.parquet   — one row per cluster, z-score diffs
  ├─ cluster_profile_summary.parquet  — one row per cluster, behavioral stats
  ├─ cluster_genre_breakdown.parquet  — long format: cluster × genre × pct
  ├─ cluster_decade_breakdown.parquet — long format: cluster × decade × pct
  └─ cluster_pop_breakdown.parquet    — long format: cluster × pop_tier × pct
─────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from pipeline.step_00_config import CACHE_DIR

# ── Input artifacts ───────────────────────────────────────────────────
# feature_matrix: user × scaled features (genre_mean_, era_, pop_, svd_)
#                 produced by step_04, filtered to users with ≥100 ratings
# cluster_assignments: user → cluster integer, produced by step_05
# transformed: raw user-movie interactions (rating, rating_centered,
#              like_flag, dislike_flag, genres_list, year_released, etc.)
FEATURE_MATRIX_IN = CACHE_DIR / "feature_matrix.parquet"
CLUSTER_ASSIGNMENTS_IN = CACHE_DIR / "cluster_assignments.parquet"
TRANSFORMED_IN = CACHE_DIR / "transformed.parquet"

# ── Output artifacts ──────────────────────────────────────────────────
CLUSTER_INTERPRETATION_OUT = CACHE_DIR / "cluster_interpretation.parquet"
CLUSTER_PROFILE_SUMMARY_OUT = CACHE_DIR / "cluster_profile_summary.parquet"
CLUSTER_GENRE_BREAKDOWN_OUT = CACHE_DIR / "cluster_genre_breakdown.parquet"
CLUSTER_DECADE_BREAKDOWN_OUT = CACHE_DIR / "cluster_decade_breakdown.parquet"
CLUSTER_POP_BREAKDOWN_OUT = CACHE_DIR / "cluster_pop_breakdown.parquet"


# ─────────────────────────────────────────────────────────────────────
# VIEW A — DISTINCTIVENESS
# Computes per-cluster mean of interpretable features in z-score space,
# then subtracts the overall population mean.
# Result: positive = cluster is above average, negative = below average.
# ─────────────────────────────────────────────────────────────────────
def build_cluster_interpretation(features: pd.DataFrame,
                                 cluster_assign: pd.DataFrame) -> pd.DataFrame:
    df = features.merge(cluster_assign[["user", "cluster"]], on="user", how="inner")

    # Only use human-interpretable feature groups.
    # Skip svd_ (latent PCA space — opaque) and fsa/region (categorical).
    interpretable_prefixes = (
        "genre_mean_", "genre_share_", "era_", "pop_",
        "like_rate", "dislike_rate", "classic_share", "modern_share",
        "english_share", "avg_release_year", "n_unique_genres"
    )
    interp_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if any(c.startswith(p) or c == p for p in interpretable_prefixes)
           and c != "cluster"
    ]

    cluster_means = df.groupby("cluster")[interp_cols].mean()
    overall_mean = df[interp_cols].mean()
    cluster_diff = cluster_means.sub(overall_mean, axis=1)  # distinctiveness

    rows = []
    for cid, row in cluster_diff.iterrows():
        # Top 5 distinctive genres: ranked by how much this cluster rates
        # these genres above the population mean (in scaled space)
        genre_cols = {c: row[c] for c in row.index if c.startswith("genre_mean_")}
        top_genres = [
            g.replace("genre_mean_", "")
            for g in sorted(genre_cols, key=genre_cols.get, reverse=True)[:5]
        ]

        # Top 3 distinctive eras: decades this cluster over-indexes on.
        # Guard against data quality artifacts (e.g. era_1870)
        era_cols = {c: row[c] for c in row.index if c.startswith("era_")}
        top_eras = []
        for e in sorted(era_cols, key=era_cols.get, reverse=True):
            try:
                decade = int(float(e.replace("era_", "")))
                if 1900 <= decade <= 2030:
                    top_eras.append(decade)
            except ValueError:
                pass
            if len(top_eras) == 3:
                break

        # Dominant popularity tier: which of low/mid/high the cluster
        # over-indexes on most strongly
        pop_cols = {
            c: row[c] for c in ["pop_low", "pop_mid", "pop_high"]
            if c in row.index
        }
        dominant_pop = max(pop_cols, key=pop_cols.get) if pop_cols else "unknown"

        rows.append({
            "cluster": int(cid),
            # Pipe-delimited strings — easy to split downstream or feed to LLM
            "distinctive_genres": " | ".join(top_genres),
            "distinctive_decades": " | ".join(str(d) for d in top_eras),
            "dominant_pop_tier": dominant_pop,
            # Direction indicators (z-score scale):
            # like_rate_diff > 0 → cluster likes movies more than average user
            "like_rate_diff": round(float(cluster_means.loc[cid, "like_rate"]), 4)
            if "like_rate" in cluster_means.columns else None,
            # classic_share_diff > 0 → cluster watches more pre-1980 films than average
            "classic_share_diff": round(float(cluster_diff.loc[cid, "classic_share"]), 4)
            if "classic_share" in cluster_diff.columns else None,
            # modern_share_diff > 0 → cluster skews toward post-2010 releases
            "modern_share_diff": round(float(cluster_diff.loc[cid, "modern_share"]), 4)
            if "modern_share" in cluster_diff.columns else None,
            # english_share_diff > 0 → cluster watches more English-language films
            "english_share_diff": round(float(cluster_diff.loc[cid, "english_share"]), 4)
            if "english_share" in cluster_diff.columns else None,
        })

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# VIEW B — RAW PROFILE
# Computes actual percentages from interaction-level data (transformed.parquet).
# These are real proportions — not z-scores — suitable for user-facing text
# and questionnaire design.
# Returns a dict keyed by cluster_id for easy per-cluster access downstream.
# ─────────────────────────────────────────────────────────────────────
def build_cluster_profile(reviews: pd.DataFrame,
                          cluster_assign: pd.DataFrame) -> dict:
    df = reviews.merge(cluster_assign[["user", "cluster"]], on="user", how="inner")

    # Assign popularity bucket at interaction level (consistent with feature matrix)
    df["popularity_bucket"] = pd.qcut(
        df["vote_count"], 3, labels=["low", "mid", "high"], duplicates="drop"
    )
    df["decade"] = ((df["year_released"] // 10) * 10).astype("Int64")

    profiles = {}
    for cid, grp in df.groupby("cluster"):

        # ── Behavioral summary ────────────────────────────────────────
        n_users = grp["user"].nunique()
        n_ratings = len(grp)
        avg_rating = round(float(grp["rating"].mean()), 3)
        # like_flag / dislike_flag are binary (1/0), so mean = proportion
        like_rate = round(grp["like_flag"].mean() * 100, 1)
        dislike_rate = round(grp["dislike_flag"].mean() * 100, 1)

        # ── English share ─────────────────────────────────────────────
        # Explode languages_list (one language per row) and measure
        # what fraction of film interactions are English-language
        l = grp[["languages_list"]].explode("languages_list").dropna()
        english_pct = round(
            l["languages_list"].str.lower()
            .isin(["english", "en", "eng"]).mean() * 100, 1
        )

        # ── Genre breakdown (% of genre interactions) ─────────────────
        # Note: one movie can have multiple genres, so this measures
        # share of genre-tagged interactions, not unique movies.
        # "What genres does this cluster consume?" (volume, not affinity)
        g = grp[["genres_list"]].explode("genres_list").dropna()
        if len(g) > 0 and isinstance(g["genres_list"].iloc[0], np.ndarray):
            g = grp.assign(
                genres_list=grp["genres_list"].apply(
                    lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                )
            )[["genres_list"]].explode("genres_list").dropna()

        genre_pct = (
            g["genres_list"].value_counts(normalize=True)
            .mul(100).round(1)
            .head(8)
            .to_dict()
        )

        # ── Decade breakdown (% of rating interactions per decade) ────
        decade_pct = (
            grp["decade"].value_counts(normalize=True)
            .mul(100).round(1)
            .sort_index(ascending=False)
            .head(6)
            .to_dict()
        )
        # Filter out data quality artifacts (impossible years)
        decade_pct = {
            k: v for k, v in decade_pct.items()
            if pd.notna(k) and 1900 <= int(k) <= 2030
        }

        # ── Popularity tier breakdown ─────────────────────────────────
        pop_pct = (
            grp["popularity_bucket"].value_counts(normalize=True)
            .mul(100).round(1)
            .to_dict()
        )

        profiles[int(cid)] = {
            "n_users": n_users,
            "n_ratings": n_ratings,
            "avg_rating": avg_rating,
            "like_rate_pct": like_rate,
            "dislike_rate_pct": dislike_rate,
            "english_pct": english_pct,
            "genre_pct": genre_pct,  # dict: {genre: pct}
            "decade_pct": decade_pct,  # dict: {decade: pct}
            "pop_pct": pop_pct,  # dict: {tier: pct}
        }

    return profiles


# ─────────────────────────────────────────────────────────────────────
# FLATTEN — Convert nested profile dicts to flat DataFrames for parquet
# ─────────────────────────────────────────────────────────────────────
def flatten_profiles(profiles: dict) -> tuple[pd.DataFrame, pd.DataFrame,
pd.DataFrame, pd.DataFrame]:
    """
    Splits the profiles dict into four flat, parquet-friendly DataFrames:
      summary   — one row per cluster (scalar stats)
      genre_df  — long format: cluster | genre | pct
      decade_df — long format: cluster | decade | pct
      pop_df    — long format: cluster | pop_tier | pct
    """
    summary_rows, genre_rows, decade_rows, pop_rows = [], [], [], []

    for cid, p in sorted(profiles.items()):
        summary_rows.append({
            "cluster": cid,
            "n_users": p["n_users"],
            "n_ratings": p["n_ratings"],
            "avg_rating": p["avg_rating"],
            "like_rate_pct": p["like_rate_pct"],
            "dislike_rate_pct": p["dislike_rate_pct"],
            "english_pct": p["english_pct"],
        })
        for genre, pct in p["genre_pct"].items():
            genre_rows.append({"cluster": cid, "genre": genre, "pct": pct})
        for decade, pct in p["decade_pct"].items():
            decade_rows.append({"cluster": cid, "decade": int(decade), "pct": pct})
        for tier, pct in p["pop_pct"].items():
            pop_rows.append({"cluster": cid, "pop_tier": str(tier), "pct": pct})

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(genre_rows).sort_values(["cluster", "pct"], ascending=[True, False]),
        pd.DataFrame(decade_rows).sort_values(["cluster", "decade"], ascending=[True, False]),
        pd.DataFrame(pop_rows).sort_values(["cluster", "pct"], ascending=[True, False]),
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main() -> tuple[pd.DataFrame, dict]:
    # Skip if all outputs already exist
    if (
        CLUSTER_INTERPRETATION_OUT.exists() and
        CLUSTER_PROFILE_SUMMARY_OUT.exists() and
        CLUSTER_GENRE_BREAKDOWN_OUT.exists() and
        CLUSTER_DECADE_BREAKDOWN_OUT.exists() and
        CLUSTER_POP_BREAKDOWN_OUT.exists()
    ):
        print("[07_cluster_kb] Cache exists. Skipping cluster KB build.")
        return
    # Load input artifacts
    features = pd.read_parquet(FEATURE_MATRIX_IN) # scaled, interpretable + svd features
    cluster_assign = pd.read_parquet(CLUSTER_ASSIGNMENTS_IN) # user → cluster mapping
    reviews = pd.read_parquet(TRANSFORMED_IN) # raw interactions for percentages

    # View A: distinctiveness (z-score diff)
    # Measures how each cluster differs from the overall population on key behaviors
    interpretation = build_cluster_interpretation(features, cluster_assign)

    # View B: raw behavioral percentages
    # Real-world percentages of interactions per cluster (genres, decades, popularity, language)
    profiles = build_cluster_profile(reviews, cluster_assign)

    # Flatten nested profile dicts into long/flat DataFrames for parquet storage
    summary_df, genre_df, decade_df, pop_df = flatten_profiles(profiles)

    # Save all outputs for downstream consumption / LLM persona generation / UI
    interpretation.to_parquet(CLUSTER_INTERPRETATION_OUT, index=False)
    summary_df.to_parquet(CLUSTER_PROFILE_SUMMARY_OUT, index=False)
    genre_df.to_parquet(CLUSTER_GENRE_BREAKDOWN_OUT, index=False)
    decade_df.to_parquet(CLUSTER_DECADE_BREAKDOWN_OUT, index=False)
    pop_df.to_parquet(CLUSTER_POP_BREAKDOWN_OUT, index=False)

    print("[07_cluster_kb] Saved:", CLUSTER_INTERPRETATION_OUT)
    print("[07_cluster_kb] Saved:", CLUSTER_PROFILE_SUMMARY_OUT)
    print("[07_cluster_kb] Saved:", CLUSTER_GENRE_BREAKDOWN_OUT)
    print("[07_cluster_kb] Saved:", CLUSTER_DECADE_BREAKDOWN_OUT)
    print("[07_cluster_kb] Saved:", CLUSTER_POP_BREAKDOWN_OUT)
    # Return data for immediate inspection or testing
    return interpretation, profiles


if __name__ == "__main__":
    main()