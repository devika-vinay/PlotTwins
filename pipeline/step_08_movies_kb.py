"""
step_08_movies_kb.py
─────────────────────────────────────────────────────────────────────────
KNOWLEDGE BASE 2 — CLUSTER MOVIE INVENTORY

Answers: "What movies define each cluster?"

Purpose: WHAT each cluster watches — scored, filtered movie inventory per cluster

Reads cluster_interpretation.parquet (produced by step_07) and
transformed.parquet to build a comprehensive, scored movie list
per cluster.

Key column — representative_score:
  Measures how well each movie aligns with its cluster's learned
  feature profile. Scoring rubric:
    +1.0 per genre match with cluster's top 5 distinctive genres (max 5.0)
    +2.0 if movie's decade is in cluster's top 3 distinctive eras
    +1.5 if avg_rating_centered > 0  (cluster liked it above their baseline)
    +1.5 if like_rate ≥ 0.75         (strong consensus across cluster users)
    +1.0 if popularity bucket matches cluster's dominant tier

  High score = movie is genuinely representative of the cluster's taste,
  not just frequently watched. Used to validate persona names and
  generate questionnaire movie examples.

Key column — cluster_share:
  Fraction of this movie's total ratings that came from this cluster.
  High value → movie is distinctive to this cluster (not cross-cluster noise).

Filtering:
  Movies must be rated by ≥1% of the cluster's users to qualify.
  This removes long-tail noise while keeping niche titles that
  matter to a meaningful subset of the cluster.

Output:
  └─ cluster_movie_kb.parquet  — one row per (cluster, movie)
─────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import re
from pipeline.step_00_config import CACHE_DIR

# ── Input artifacts ───────────────────────────────────────────────────
# cluster_interpretation: z-score diff signals per cluster (from step_07)
#   Used to extract the signal dict (top genres, eras, pop tier)
#   that drives representative_score calculation.
# cluster_assignments: user → cluster integer (from step_05)
# transformed: raw interaction data with rating_centered, like_flag, etc.
CLUSTER_INTERPRETATION_IN = CACHE_DIR / "cluster_interpretation.parquet"
CLUSTER_ASSIGNMENTS_IN = CACHE_DIR / "cluster_assignments.parquet"
TRANSFORMED_IN = CACHE_DIR / "transformed.parquet"

# ── Output artifact ───────────────────────────────────────────────────
CLUSTER_MOVIE_KB_OUT = CACHE_DIR / "cluster_movie_kb.parquet"


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def agg_genres(series) -> list:
    """
    Aggregates genres_list values across rows into a single sorted list.
    Handles both Python lists and numpy arrays (both formats appear
    depending on how parquet deserializes the list column).
    """
    genres = set()
    for val in series.dropna():
        if isinstance(val, (list, np.ndarray)):
            genres.update(val.tolist() if isinstance(val, np.ndarray) else val)
        else:
            genres.add(val)
    return sorted(genres)

def title_norm_to_display(title_norm: str) -> str:
    if pd.isna(title_norm):
        return ""

    s = str(title_norm).strip().lower()

    # replace hyphens/underscores with spaces
    s = re.sub(r"[-_]+", " ", s)

    # remove trailing year like 2017 or 2011 if present at the end
    s = re.sub(r"\b(19|20)\d{2}\b$", "", s).strip()

    # collapse extra spaces
    s = re.sub(r"\s+", " ", s)

    # title case
    return s.title()


def score_movie(row: pd.Series, signal: dict) -> float:
    """
    Scores a movie's representativeness for its assigned cluster.

    The signal dict comes from cluster_interpretation (step_07),
    which reflects what the KMeans model actually learned — not hardcoded rules.
    This ensures movie scoring is backed by the model's learned separation.

    Rubric:
      Genre match:   +1.0 per matching genre (cluster's top 5 distinctive genres)
      Era match:     +2.0 if movie's decade is in cluster's top 3 distinctive eras
      Rating signal: +1.5 if avg_rating_centered > 0 (liked above user baseline)
      Consensus:     +1.5 if like_rate ≥ 0.75 (75%+ of cluster users liked it)
      Pop match:     +1.0 if popularity tier matches cluster's dominant tier
    """
    score = 0.0

    # Genre alignment — uses distinctive_genres from interpretation
    movie_genres = row.get("genres", [])
    if isinstance(movie_genres, np.ndarray):
        movie_genres = movie_genres.tolist()
    score += len(set(movie_genres) & set(signal["top_genres"])) * 1.0

    # Era alignment — uses distinctive_decades from interpretation
    year = row.get("year_released")
    if pd.notna(year):
        if int(year // 10) * 10 in signal["top_decades"]:
            score += 2.0

    # Rating signal — did cluster users rate it above their own average?
    # rating_centered = user's rating minus their personal mean rating
    if pd.notna(row.get("avg_rating_centered")) and row["avg_rating_centered"] > 0:
        score += 1.5

    # Consensus — strong majority of cluster users explicitly liked it
    if pd.notna(row.get("like_rate")) and row["like_rate"] >= 0.75:
        score += 1.5

    # Popularity alignment — niche clusters shouldn't be defined by blockbusters
    pop = row.get("popularity_bucket")
    if pop and signal["dominant_pop"] and pop == signal["dominant_pop"].replace("pop_", ""):
        score += 1.0

    return round(score, 3)


# ─────────────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ─────────────────────────────────────────────────────────────────────
def build_movie_kb(reviews: pd.DataFrame,
                   cluster_assign: pd.DataFrame,
                   interpretation: pd.DataFrame) -> pd.DataFrame:
    # Join every rating with its user's cluster label
    df = reviews.merge(cluster_assign[["user", "cluster"]], on="user", how="inner")

    # Assign popularity bucket at interaction level.
    # Consistent with step_04 feature engineering (qcut into 3 equal buckets).
    df["popularity_bucket"] = pd.qcut(
        df["vote_count"], 3, labels=["low", "mid", "high"], duplicates="drop"
    )

    # Count distinct users per cluster — used for the 1% threshold filter
    cluster_user_counts = (
        cluster_assign.groupby("cluster")["user"]
        .nunique()
        .rename("cluster_n_users")
        .reset_index()
    )

    # Aggregate to (cluster, movie) level — one row per movie per cluster
    kb = (
        df.groupby(["cluster", "title_norm"])
        .agg(
            genres=("genres_list", agg_genres),
            n_ratings=("rating", "size"),
            avg_rating=("rating", "mean"),
            # rating_centered: user's rating minus their personal mean
            # Positive = cluster liked this movie above their usual baseline
            avg_rating_centered=("rating_centered", "mean"),
            # like_flag = 1 if rating_centered > 0, else 0
            like_rate=("like_flag", "mean"),
            dislike_rate=("dislike_flag", "mean"),
            year_released=("year_released", "first"),
            original_language=("original_language", "first"),
            avg_popularity=("popularity", "mean"),
            # Modal popularity bucket for this movie within the cluster
            popularity_bucket=("popularity_bucket",
                               lambda x: x.mode()[0] if not x.empty else None),
        )
        .reset_index()
    )

    kb["display_title"] = kb["title_norm"].apply(title_norm_to_display)

    # ── Filter: minimum 1% of cluster users must have rated the movie ──
    # Prevents long-tail movies (rated by 1-2 users) from polluting the KB.
    # Keeps niche titles that still matter to a meaningful cluster subset.
    kb = kb.merge(cluster_user_counts, on="cluster")
    kb = kb[kb["n_ratings"] >= (kb["cluster_n_users"] * 0.01)].copy()

    # ── Filter: drop movies with no genre metadata ─────────────────────
    kb = kb[kb["genres"].map(len) > 0].copy()

    # ── Cluster share ──────────────────────────────────────────────────
    # Fraction of this movie's cross-cluster ratings that came from this cluster.
    # High value → movie is distinctive to this cluster.
    # Low value → movie is watched broadly across clusters (less diagnostic).
    total_per_movie = (
        kb.groupby("title_norm")["n_ratings"]
        .sum()
        .rename("total_ratings")
        .reset_index()
    )
    kb = kb.merge(total_per_movie, on="title_norm")
    kb["cluster_share"] = (kb["n_ratings"] / kb["total_ratings"]).round(3)

    # ── Build signal dict from interpretation (step_07 View A) ────────
    # Extracts the three signals that drive score_movie():
    # top_genres, top_decades, dominant_pop — all derived from the
    # KMeans feature space, not hardcoded.
    signals = {}
    for _, row in interpretation.iterrows():
        cid = int(row["cluster"])
        signals[cid] = {
            "top_genres": row["distinctive_genres"].split(" | "),
            "top_decades": [int(d) for d in row["distinctive_decades"].split(" | ") if d],
            "dominant_pop": row["dominant_pop_tier"],
        }

    # ── Score every movie against its cluster's signal ─────────────────
    kb["representative_score"] = kb.apply(
        lambda r: score_movie(r, signals[int(r["cluster"])]), axis=1
    )

    # Round float columns for cleaner storage
    for col in ["avg_rating", "avg_rating_centered", "like_rate", "dislike_rate"]:
        kb[col] = kb[col].round(3)

    # Sort: within each cluster, most representative + most-rated first
    return (
        kb
        .sort_values(
            ["cluster", "representative_score", "n_ratings"],
            ascending=[True, False, False]
        )
        .drop(columns=["total_ratings"])  # intermediate column, not needed in output
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main() -> pd.DataFrame:

    if CLUSTER_MOVIE_KB_OUT.exists():
        print("[08_movies_kb] Cache exists. Skipping movie KB build.")
        return

    # Reads interpretation from step_07 — avoids recomputing feature matrix
    interpretation = pd.read_parquet(CLUSTER_INTERPRETATION_IN)
    cluster_assign = pd.read_parquet(CLUSTER_ASSIGNMENTS_IN)
    reviews = pd.read_parquet(TRANSFORMED_IN)

    movie_kb = build_movie_kb(reviews, cluster_assign, interpretation)

    movie_kb.to_parquet(CLUSTER_MOVIE_KB_OUT, index=False)
    print(f"[08_movies_kb] Saved: {CLUSTER_MOVIE_KB_OUT}")
    print(f"[08_movies_kb] Shape: {movie_kb.shape} "
          f"({movie_kb['cluster'].nunique()} clusters, "
          f"{movie_kb['title_norm'].nunique()} unique movies)")

    return movie_kb


if __name__ == "__main__":
    movie_kb = main()