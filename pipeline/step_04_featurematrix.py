"""
step_04_featurematrix.py
─────────────────────────────────────────────────────────────────────────
Builds the user-level feature matrix used for clustering (step_05).

Feature blocks:
  svd          — collaborative filtering signal (TruncatedSVD on user-movie matrix)
  genre_mean   — per-user mean centered rating per genre (affinity signal)
  genre_share  — per-user fraction of ratings in each genre (volume signal)
  popularity   — fraction of ratings in each popularity tier (low/mid/high)
  era          — fraction of ratings per decade
  profiles     — numeric columns from user_profiles (like_rate, classic_share, etc.)

Each block is scaled independently with its own StandardScaler.
Then all blocks are concatenated → VarianceThreshold → PCA.

Saves:
  feature_matrix.parquet       — scaled features in original space (for interpretation)
  feature_matrix_pca.parquet   — PCA-reduced features (input to KMeans in step_05)
  scalers.pkl                  — dict of fitted StandardScalers, one per feature block
  svd.pkl                      — fitted TruncatedSVD (needed to embed new user ratings)
  variance_selector.pkl        — fitted VarianceThreshold
  pca.pkl                      — fitted PCA
  feature_columns_{block}.parquet — ordered column list per block (for alignment)
─────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_selection import VarianceThreshold

from pipeline.step_00_config import CACHE_DIR

# ── Inputs ────────────────────────────────────────────────────────────
TRANSFORMED_IN = CACHE_DIR / "transformed.parquet"
USER_PROFILES_IN = CACHE_DIR / "user_profiles.parquet"

# ── Outputs ───────────────────────────────────────────────────────────
FEATURE_MATRIX_OUT = CACHE_DIR / "feature_matrix.parquet"
FEATURE_MATRIX_PCA_OUT = CACHE_DIR / "feature_matrix_pca.parquet"


def main():
    if FEATURE_MATRIX_OUT.exists() and FEATURE_MATRIX_PCA_OUT.exists():
        print("[04_feature_matrix] Cache exists. Skipping feature matrix build.")
        return


reviews = pd.read_parquet(TRANSFORMED_IN)
profiles_main = pd.read_parquet(USER_PROFILES_IN)

# ── Filter: only users with ≥100 ratings ─────────────────────────
# Sparse users produce noisy feature vectors that hurt clustering.
min_reviews = 100
reviews_per_user = reviews.groupby("user").size()
active_users = reviews_per_user[reviews_per_user >= min_reviews].index

reviews = reviews[reviews["user"].isin(active_users)].copy()
profiles = profiles_main[profiles_main["user"].isin(active_users)].set_index("user")

# ═══════════════════════════════════════════════════════════════════
# FEATURE BLOCK 1 — Genre affinity (genre_mean_)
# Mean of rating_centered per genre per user.
# Positive = user rates this genre above their personal average.
# This is the primary taste signal the model uses for separation.
# ═══════════════════════════════════════════════════════════════════
g = reviews[["user", "genres_list", "rating_centered"]].explode("genres_list")
g = g.dropna(subset=["genres_list"])
g["genres_list"] = g["genres_list"].astype(str)

genre_mean = (
    g.groupby(["user", "genres_list"])["rating_centered"]
    .mean()
    .unstack(fill_value=0)
    .add_prefix("genre_mean_")
)

# ═══════════════════════════════════════════════════════════════════
# FEATURE BLOCK 2 — Genre volume (genre_share_)
# Fraction of a user's total ratings that belong to each genre.
# Complements genre_mean: a user can watch a lot of Drama (high share)
# but not rate it above average (low mean).
# ═══════════════════════════════════════════════════════════════════
genre_count = (
    g.groupby(["user", "genres_list"])
    .size()
    .unstack(fill_value=0)
)
genre_share = (
    genre_count.div(genre_count.sum(axis=1), axis=0)
    .add_prefix("genre_share_")
)

# ═══════════════════════════════════════════════════════════════════
# FEATURE BLOCK 3 — Popularity preference (pop_)
# Fraction of ratings in each popularity tier (low/mid/high).
# Separates niche explorers from mainstream watchers.
# ═══════════════════════════════════════════════════════════════════
reviews["popularity_bucket"] = pd.qcut(
    reviews["vote_count"], 3,
    labels=["low", "mid", "high"],
    duplicates="drop"
)
pop_pref = (
    reviews.groupby(["user", "popularity_bucket"])
    .size()
    .unstack(fill_value=0)
)
pop_pref = pop_pref.div(pop_pref.sum(axis=1), axis=0)
for col in ["low", "mid", "high"]:
    if col not in pop_pref.columns:
        pop_pref[col] = 0.0
pop_pref = pop_pref[["low", "mid", "high"]]
pop_pref.columns = ["pop_low", "pop_mid", "pop_high"]

# ═══════════════════════════════════════════════════════════════════
# FEATURE BLOCK 4 — Era preference (era_)
# Fraction of ratings per decade.
# Separates classic cinema fans from modern release chasers.
# ═══════════════════════════════════════════════════════════════════
reviews["decade"] = (reviews["year_released"] // 10) * 10
era_pref = (
    reviews.groupby(["user", "decade"])
    .size()
    .unstack(fill_value=0)
)
era_pref = era_pref.div(era_pref.sum(axis=1), axis=0)
era_pref.columns = era_pref.columns.astype(str)
era_pref = era_pref.add_prefix("era_")

# Drop era columns outside valid range
valid_era_cols = [
    c for c in era_pref.columns
    if c.startswith("era_") and 1900 <= int(float(c.replace("era_", ""))) <= 2030
]
era_pref = era_pref[valid_era_cols]
era_pref = era_pref.div(era_pref.sum(axis=1), axis=0).fillna(0)

# ═══════════════════════════════════════════════════════════════════
# FEATURE BLOCK 5 — Collaborative embedding (svd_)
# TruncatedSVD on the user × movie centered-rating matrix.
# Captures latent co-rating patterns — users who rate similar
# movies similarly end up close in SVD space regardless of genre labels.
# svd object is saved so new users can be projected into the same space.
# ═══════════════════════════════════════════════════════════════════
user_movie = reviews.pivot_table(
    index="user",
    columns="title_norm",
    values="rating_centered",
    fill_value=0
)
max_svd_components = min(40, user_movie.shape[0] - 1, user_movie.shape[1] - 1)
if max_svd_components < 2:
    raise ValueError("[04_feature_matrix] Not enough users/movies for TruncatedSVD.")

svd = TruncatedSVD(n_components=max_svd_components, random_state=42)
svd_features = svd.fit_transform(user_movie)
svd_df = pd.DataFrame(
    svd_features,
    index=user_movie.index,
    columns=[f"svd_{i}" for i in range(max_svd_components)]
)

# ═══════════════════════════════════════════════════════════════════
# FEATURE BLOCK 6 — User profile stats (profiles)
# Numeric columns from step_03: like_rate, dislike_rate,
# classic_share, modern_share, english_share, genre_entropy, etc.
# These are user-level behavioral summaries, not per-movie signals.
# ═══════════════════════════════════════════════════════════════════
profile_numeric = profiles.select_dtypes(include=[np.number]).copy()
for col in ["fsa", "region"]:
    if col in profile_numeric.columns:
        profile_numeric = profile_numeric.drop(columns=col)

# ── Align all blocks to common users ─────────────────────────────
feature_blocks = {
    "svd": svd_df,
    "genre_mean": genre_mean,
    "genre_share": genre_share,
    "popularity": pop_pref,
    "era": era_pref,
    "profiles": profile_numeric,
}
common_users = sorted(
    set.intersection(*[set(df.index) for df in feature_blocks.values()])
)

# ═══════════════════════════════════════════════════════════════════
# SCALING — one StandardScaler per block
# Each block has different units and variance ranges.
# Scaling per block prevents any single block from dominating.
# All 6 scalers are saved individually in scalers.pkl dict so
# new users can be transformed through the exact same pipeline.
# ═══════════════════════════════════════════════════════════════════
scalers = {}
scaled_blocks = []

for name, df in feature_blocks.items():
    df = df.loc[common_users].copy()
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    scaled_blocks.append(scaled)
    scalers[name] = scaler  # store fitted scaler for this block

    # Save column order for this block — needed to align new user
    # features to the exact same column order during prediction
    pd.Series(df.columns.tolist()).to_frame("feature").to_parquet(
        CACHE_DIR / f"feature_columns_{name}.parquet", index=False
    )

X_scaled = pd.concat(scaled_blocks, axis=1).fillna(0)

# ═══════════════════════════════════════════════════════════════════
# NOISE REDUCTION — VarianceThreshold
# Removes features with near-zero variance across all users
# (e.g. a genre that barely anyone watches has no clustering signal).
# ═══════════════════════════════════════════════════════════════════
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X_scaled)
X_filtered = pd.DataFrame(X_filtered, index=X_scaled.index)

# ═══════════════════════════════════════════════════════════════════
# DIMENSIONALITY REDUCTION — PCA
# Reduces to 50 components (or less if data is small).
# KMeans in step_05 operates on this PCA space.
# The fitted PCA is saved so new users can be projected into
# the same reduced space for prediction.
# ═══════════════════════════════════════════════════════════════════
max_pca_components = min(50, X_filtered.shape[0], X_filtered.shape[1])
if max_pca_components < 2:
    raise ValueError("[04_feature_matrix] Not enough dimensions after variance filtering.")

pca = PCA(n_components=max_pca_components, random_state=42)
X_pca = pca.fit_transform(X_filtered)
X_pca_df = pd.DataFrame(
    X_pca,
    index=X_scaled.index,
    columns=[f"pca_{i}" for i in range(max_pca_components)]
)

# ── Attach location columns for downstream matching ───────────────
location_cols = ["fsa", "region"]
available_location_cols = [c for c in location_cols if c in profiles.columns]

feature_matrix_out = X_scaled.copy().reset_index()
if available_location_cols:
    location_df = profiles.loc[
        feature_matrix_out["user"], available_location_cols
    ].reset_index(drop=True)
    feature_matrix_out = pd.concat([feature_matrix_out, location_df], axis=1)

feature_matrix_pca_out = X_pca_df.copy().reset_index()
if available_location_cols:
    location_df_pca = profiles.loc[
        feature_matrix_pca_out["user"], available_location_cols
    ].reset_index(drop=True)
    feature_matrix_pca_out = pd.concat([feature_matrix_pca_out, location_df_pca], axis=1)

# ── Save all outputs ──────────────────────────────────────────────
feature_matrix_out.to_parquet(FEATURE_MATRIX_OUT, index=False)
feature_matrix_pca_out.to_parquet(FEATURE_MATRIX_PCA_OUT, index=False)

# Fitted transformers — required for predicting new users in step_10
joblib.dump(scalers, CACHE_DIR / "scalers.pkl")
joblib.dump(svd, CACHE_DIR / "svd.pkl")
joblib.dump(selector, CACHE_DIR / "variance_selector.pkl")
joblib.dump(pca, CACHE_DIR / "pca.pkl")

print("[04_feature_matrix] Saved:", FEATURE_MATRIX_OUT)
print("[04_feature_matrix] Saved:", FEATURE_MATRIX_PCA_OUT)
print("[04_feature_matrix] Saved: scalers.pkl, svd.pkl, variance_selector.pkl, pca.pkl")
print("[04_feature_matrix] Saved: feature_columns_{block}.parquet for each block")
print(f"[04_feature_matrix] Feature matrix shape: {feature_matrix_out.shape}")
print(f"[04_feature_matrix] PCA matrix shape:     {feature_matrix_pca_out.shape}")

if __name__ == "__main__":
    main()
