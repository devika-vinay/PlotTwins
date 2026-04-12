import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_selection import VarianceThreshold

from pipeline.step_00_config import CACHE_DIR

TRANSFORMED_IN = CACHE_DIR / "transformed.parquet"
USER_PROFILES_IN = CACHE_DIR / "user_profiles.parquet"

FEATURE_MATRIX_OUT = CACHE_DIR / "feature_matrix.parquet"
FEATURE_MATRIX_PCA_OUT = CACHE_DIR / "feature_matrix_pca.parquet"


def main():
    if FEATURE_MATRIX_OUT.exists() and FEATURE_MATRIX_PCA_OUT.exists():
        print("[04_feature_matrix] Cache exists. Skipping feature matrix build.")
        return

    reviews = pd.read_parquet(TRANSFORMED_IN)
    profiles_main = pd.read_parquet(USER_PROFILES_IN)

    # ----------------------------
    # Filter active users
    # ----------------------------
    min_reviews = 100
    # Keep only users with sufficient interaction history (reduces noise / sparsity)
    reviews_per_user = reviews.groupby("user").size()
    reviews_per_user = reviews_per_user[reviews_per_user >= min_reviews]

    transformed_filtered = reviews[
        reviews["user"].isin(reviews_per_user.index)
    ].copy()

    profiles_filtered = profiles_main[
        profiles_main["user"].isin(reviews_per_user.index)
    ].copy()

    # ----------------------------
    # Base DataFrames
    # ----------------------------
    profiles = profiles_filtered.set_index("user")
    reviews = transformed_filtered.copy()

    # ----------------------------
    # Genre Feature Engineering
    # ----------------------------
    # Explode to create one row per (user, genre) interaction
    g = reviews[["user", "genres_list", "rating_centered"]].explode("genres_list")
    g = g.dropna(subset=["genres_list"])
    g["genres_list"] = g["genres_list"].astype(str)
    # Average preference per genre per user
    genre_mean = (
        g.groupby(["user", "genres_list"])["rating_centered"]
        .mean()
        .unstack(fill_value=0)
    )
    genre_mean = genre_mean.add_prefix("genre_mean_")
    # Frequency-based genre distribution
    genre_count = (
        g.groupby(["user", "genres_list"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalize counts → proportions (user preference distribution)
    genre_share = genre_count.div(genre_count.sum(axis=1), axis=0)
    genre_share = genre_share.add_prefix("genre_share_")

    # ----------------------------
    # Popularity Preference
    # ----------------------------
    # Bucket movies into popularity tiers using quantiles
    reviews["popularity_bucket"] = pd.qcut(
        reviews["vote_count"],
        3,
        labels=["low", "mid", "high"],
        duplicates="drop"
    )
    # Convert counts → proportions per user
    pop_pref = (
        reviews.groupby(["user", "popularity_bucket"])
        .size()
        .unstack(fill_value=0)
    )

    pop_pref = pop_pref.div(pop_pref.sum(axis=1), axis=0)

    # make sure all three expected columns exist
    for col in ["low", "mid", "high"]:
        if col not in pop_pref.columns:
            pop_pref[col] = 0.0

    pop_pref = pop_pref[["low", "mid", "high"]]
    pop_pref.columns = ["pop_low", "pop_mid", "pop_high"]

    # ----------------------------
    # Temporal Preference
    # ----------------------------
    # Convert years into decades for coarser preference patterns
    reviews["decade"] = (reviews["year_released"] // 10) * 10

    era_pref = (
        reviews.groupby(["user", "decade"])
        .size()
        .unstack(fill_value=0)
    )

    era_pref = era_pref.div(era_pref.sum(axis=1), axis=0)
    era_pref.columns = era_pref.columns.astype(str)
    era_pref = era_pref.add_prefix("era_")


    # ----------------------------
    # Collaborative Embedding (SVD)
    # ----------------------------
    # Create user-item matrix (implicit collaborative filtering setup)
    user_movie = reviews.pivot_table(
        index="user",
        columns="title_norm",
        values="rating_centered",
        fill_value=0
    )

    # guard against very small dimensionality
    max_svd_components = min(40, user_movie.shape[0] - 1, user_movie.shape[1] - 1)
    if max_svd_components < 2:
        raise ValueError(
            "[04_feature_matrix] Not enough users/movies after filtering to run TruncatedSVD."
        )
    # Reduce high-dimensional user-item space into latent factors
    svd = TruncatedSVD(n_components=max_svd_components, random_state=42)
    svd_features = svd.fit_transform(user_movie)

    svd_df = pd.DataFrame(
        svd_features,
        index=user_movie.index,
        columns=[f"svd_{i}" for i in range(max_svd_components)]
    )

    # ----------------------------
    # Combine Feature Blocks
    # ----------------------------
    profile_numeric = profiles.select_dtypes(include=[np.number]).copy()

    # explicitly exclude location from numeric matrix if ever encoded later
    for col in ["fsa", "region"]:
        if col in profile_numeric.columns:
            profile_numeric = profile_numeric.drop(columns=col)

    feature_blocks = {
        "svd": svd_df,
        "genre_mean": genre_mean,
        "genre_share": genre_share,
        "popularity": pop_pref,
        "era": era_pref,
        "profiles": profile_numeric
    }

    # align all feature blocks to the same user index
    common_users = sorted(set.intersection(*[set(df.index) for df in feature_blocks.values()]))

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
        scalers[name] = scaler

        # Save column order for this block — needed to align new user
        # features to the exact same column order during prediction
        pd.Series(df.columns.tolist()).to_frame("feature").to_parquet(
            CACHE_DIR / f"feature_columns_{name}.parquet", index=False
        )

    X_scaled = pd.concat(scaled_blocks, axis=1).fillna(0)

    # ----------------------------
    # Noise Reduction + PCA
    # ----------------------------
    # Remove near-constant features (low information)
    selector = VarianceThreshold(threshold=0.01)
    X_filtered = selector.fit_transform(X_scaled)

    X_filtered = pd.DataFrame(
        X_filtered,
        index=X_scaled.index
    )

    max_pca_components = min(50, X_filtered.shape[0], X_filtered.shape[1])
    if max_pca_components < 2:
        raise ValueError(
            "[04_feature_matrix] Not enough dimensions remaining after variance filtering to run PCA."
        )
    # Further compress features into orthogonal components (dimensionality reduction)
    pca = PCA(n_components=max_pca_components, random_state=42)
    X_pca = pca.fit_transform(X_filtered)

    X_pca_df = pd.DataFrame(
        X_pca,
        index=X_scaled.index,
        columns=[f"pca_{i}" for i in range(max_pca_components)]
    )

    # ----------------------------
    # Attach location for downstream matching
    # ----------------------------
    location_cols = ["fsa", "region"]
    available_location_cols = [c for c in location_cols if c in profiles.columns]

    feature_matrix_out = X_scaled.copy().reset_index()

    if available_location_cols:
        location_df = profiles.loc[feature_matrix_out["user"], available_location_cols].reset_index(drop=True)
        feature_matrix_out = pd.concat([feature_matrix_out, location_df], axis=1)

    feature_matrix_pca_out = X_pca_df.copy().reset_index()

    if available_location_cols:
        location_df_pca = profiles.loc[feature_matrix_pca_out["user"], available_location_cols].reset_index(drop=True)
        feature_matrix_pca_out = pd.concat([feature_matrix_pca_out, location_df_pca], axis=1)

    # ----------------------------
    # Save outputs
    # ----------------------------
    feature_matrix_out.to_parquet(FEATURE_MATRIX_OUT, index=False)
    feature_matrix_pca_out.to_parquet(FEATURE_MATRIX_PCA_OUT, index=False)

    # Fitted transformers — required for predicting new users in step_10
    joblib.dump(scalers, CACHE_DIR / "scalers.pkl")
    joblib.dump(svd, CACHE_DIR / "svd.pkl")
    joblib.dump(selector, CACHE_DIR / "variance_selector.pkl")
    joblib.dump(pca, CACHE_DIR / "pca.pkl")

    print("[04_feature_matrix] Saved: scalers.pkl, svd.pkl, variance_selector.pkl, pca.pkl")
    print("[04_feature_matrix] Saved: feature_columns_{block}.parquet for each block")
    print(f"[04_feature_matrix] Feature matrix shape: {feature_matrix_out.shape}")
    print(f"[04_feature_matrix] PCA matrix shape:     {feature_matrix_pca_out.shape}")

    print("[04_feature_matrix] Saved:", FEATURE_MATRIX_OUT)
    print("[04_feature_matrix] Saved:", FEATURE_MATRIX_PCA_OUT)


if __name__ == "__main__":
    main()