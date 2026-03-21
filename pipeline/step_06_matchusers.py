import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from pipeline.step_00_config import CACHE_DIR


FEATURE_MATRIX_PCA_IN = CACHE_DIR / "feature_matrix_pca.parquet"
CLUSTER_ASSIGNMENTS_IN = CACHE_DIR / "cluster_assignments.parquet"

USER_MATCHES_OUT = CACHE_DIR / "user_matches.parquet"


TOP_N = 5


def main():
    if USER_MATCHES_OUT.exists():
        print("[06_match_users] Cache exists. Skipping user matching.")
        return

    feature_matrix_pca = pd.read_parquet(FEATURE_MATRIX_PCA_IN)
    cluster_assignments = pd.read_parquet(CLUSTER_ASSIGNMENTS_IN)

    # Use only PCA columns for similarity
    pca_cols = [col for col in feature_matrix_pca.columns if col.startswith("pca_")]
    if not pca_cols:
        raise ValueError("[06_match_users] No PCA columns found in feature_matrix_pca.parquet")

    # Base info from step 04
    base_cols = [col for col in ["user", "fsa", "region"] if col in feature_matrix_pca.columns]
    base_info = feature_matrix_pca[base_cols].copy()

    # Attach cluster labels from step 05
    cluster_cols = [col for col in ["user", "cluster"] if col in cluster_assignments.columns]
    cluster_info = cluster_assignments[cluster_cols].copy()

    users_df = base_info.merge(cluster_info, on="user", how="left")

    # Similarity matrix
    X = feature_matrix_pca[pca_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    sim_matrix = cosine_similarity(X.values)

    users = feature_matrix_pca["user"].tolist()
    user_to_idx = {user: idx for idx, user in enumerate(users)}

    rows = []

    for user in users:
        i = user_to_idx[user]
        current = users_df[users_df["user"] == user].iloc[0]

        # Candidate pool: same region only
        candidates = users_df[
            (users_df["user"] != user) &
            (users_df["region"] == current["region"])
        ].copy()

        # If same-region pool is too small, fall back to all other users
        if len(candidates) < TOP_N:
            candidates = users_df[users_df["user"] != user].copy()

        candidate_scores = []
        for _, cand in candidates.iterrows():
            j = user_to_idx[cand["user"]]
            score = float(sim_matrix[i, j])

            same_region = int(
                pd.notna(current.get("region")) and
                pd.notna(cand.get("region")) and
                current.get("region") == cand.get("region")
            )

            same_fsa = int(
                pd.notna(current.get("fsa")) and
                pd.notna(cand.get("fsa")) and
                current.get("fsa") == cand.get("fsa")
            )

            same_cluster = int(
                pd.notna(current.get("cluster")) and
                pd.notna(cand.get("cluster")) and
                current.get("cluster") == cand.get("cluster")
            )

            candidate_scores.append({
                "user": user,
                "match_user": cand["user"],
                "similarity": score,
                "same_region": same_region,
                "same_fsa": same_fsa,
                "same_cluster": same_cluster,
                "user_region": current.get("region"),
                "match_region": cand.get("region"),
                "user_fsa": current.get("fsa"),
                "match_fsa": cand.get("fsa"),
                "user_cluster": current.get("cluster"),
                "match_cluster": cand.get("cluster"),
            })

        match_df = pd.DataFrame(candidate_scores)

        # Prioritize:
        # 1. same region
        # 2. same cluster
        # 3. similarity
        match_df = match_df.sort_values(
            by=["same_region", "same_cluster", "similarity"],
            ascending=[False, False, False]
        ).head(TOP_N)

        rows.append(match_df)

    user_matches = pd.concat(rows, ignore_index=True)
    user_matches.to_parquet(USER_MATCHES_OUT, index=False)

    print("[06_match_users] Saved:", USER_MATCHES_OUT)


if __name__ == "__main__":
    main()