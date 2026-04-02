import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.manifold import TSNE

from pipeline.step_00_config import CACHE_DIR


FEATURE_MATRIX_PCA_IN = CACHE_DIR / "feature_matrix_pca.parquet"

CLUSTER_ASSIGNMENTS_OUT = CACHE_DIR / "cluster_assignments.parquet"
CLUSTER_METRICS_OUT = CACHE_DIR / "cluster_metrics.parquet"
CLUSTER_STABILITY_OUT = CACHE_DIR / "cluster_stability.parquet"
CLUSTER_PROFILE_DIFF_OUT = CACHE_DIR / "cluster_profile_diff.parquet"
TASTE_MAP_2D_OUT = CACHE_DIR / "taste_map_2d.parquet"
TASTE_MAP_3D_OUT = CACHE_DIR / "taste_map_3d.parquet"


FINAL_K = 6
K_RANGE = range(2, 20)
N_INIT = 20
RANDOM_STATE = 42
STABILITY_RUNS = 10


def cluster_stability(X: np.ndarray, k: int, runs: int = 10) -> float:
    clusterings = []

    for seed in range(runs):
        km = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT)
        labels = km.fit_predict(X)
        clusterings.append(labels)

    scores = []
    for i in range(len(clusterings)):
        for j in range(i + 1, len(clusterings)):
            score = adjusted_rand_score(clusterings[i], clusterings[j])
            scores.append(score)

    if not scores:
        return np.nan

    return float(np.mean(scores))


def main():
    if (
        CLUSTER_ASSIGNMENTS_OUT.exists()
        and CLUSTER_METRICS_OUT.exists()
        and CLUSTER_STABILITY_OUT.exists()
        and CLUSTER_PROFILE_DIFF_OUT.exists()
    ):
        print("[05_cluster_users] Cache exists. Skipping clustering.")
        return

    feature_matrix_pca = pd.read_parquet(FEATURE_MATRIX_PCA_IN)

    # Keep identifiers / location separate from clustering inputs
    id_cols = [col for col in ["user", "fsa", "region"] if col in feature_matrix_pca.columns]
    pca_cols = [col for col in feature_matrix_pca.columns if col.startswith("pca_")]

    if not pca_cols:
        raise ValueError("[05_cluster_users] No PCA columns found in feature_matrix_pca.parquet")

    base_info = feature_matrix_pca[id_cols].copy()
    X = feature_matrix_pca[pca_cols].copy()

    # Safety cleanup
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(X) < 3:
        raise ValueError("[05_cluster_users] Not enough users to cluster.")

    # ----------------------------
    # Evaluate candidate K values
    # ----------------------------
    max_valid_k = min(max(K_RANGE), len(X) - 1)
    valid_k_values = [k for k in K_RANGE if k <= max_valid_k]

    if len(valid_k_values) == 0:
        raise ValueError("[05_cluster_users] No valid k values available for clustering.")

    results = []

    for k in valid_k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = kmeans.fit_predict(X)

        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        inertia = kmeans.inertia_

        results.append([k, sil, db, ch, inertia])

    metrics_df = pd.DataFrame(
        results,
        columns=["k", "silhouette", "davies_bouldin", "calinski_harabasz", "inertia"],
    )

    # ----------------------------
    # Cluster stability
    # ----------------------------
    stability_scores = {}
    for k in valid_k_values:
        stability_scores[k] = cluster_stability(X.values, k, runs=STABILITY_RUNS)

    stability_df = pd.DataFrame(
        {
            "k": list(stability_scores.keys()),
            "stability": list(stability_scores.values()),
        }
    )

    # ----------------------------
    # Final clustering
    # ----------------------------
    final_k = FINAL_K if FINAL_K in valid_k_values else valid_k_values[-1]

    kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=N_INIT)
    clusters = kmeans.fit_predict(X)
    joblib.dump(kmeans, CACHE_DIR / "kmeans_model.pkl")

    cluster_assignments = base_info.copy()
    cluster_assignments["cluster"] = clusters

    # ----------------------------
    # Cluster interpretation
    # ----------------------------
    # Use PCA-space means vs overall mean, since this step receives PCA outputs
    X_cluster = X.copy()
    X_cluster["cluster"] = clusters

    cluster_means = X_cluster.groupby("cluster").mean()
    overall_mean = X.mean()
    cluster_diff = cluster_means.sub(overall_mean, axis=1).reset_index()

    # ----------------------------
    # 2D t-SNE map
    # ----------------------------
    # Perplexity must be < n_samples; clamp to a safe range
    n_samples = len(X)
    perplexity_2d = min(40, max(5, n_samples - 1))
    if perplexity_2d >= n_samples:
        perplexity_2d = max(1, n_samples // 3)

    tsne_2d = TSNE(
        n_components=2,
        perplexity=perplexity_2d,
        learning_rate="auto",
        early_exaggeration=15,
        init="pca",
        random_state=RANDOM_STATE,
    )
    embedding_2d = tsne_2d.fit_transform(X.values)

    taste_map_2d = base_info.copy()
    taste_map_2d["x"] = embedding_2d[:, 0]
    taste_map_2d["y"] = embedding_2d[:, 1]
    taste_map_2d["cluster"] = clusters

    # ----------------------------
    # 3D t-SNE map
    # ----------------------------
    perplexity_3d = min(40, max(5, n_samples - 1))
    if perplexity_3d >= n_samples:
        perplexity_3d = max(1, n_samples // 3)

    tsne_3d = TSNE(
        n_components=3,
        perplexity=perplexity_3d,
        init="pca",
        learning_rate="auto",
        random_state=RANDOM_STATE,
    )
    embedding_3d = tsne_3d.fit_transform(X.values)

    taste_map_3d = base_info.copy()
    taste_map_3d["x"] = embedding_3d[:, 0]
    taste_map_3d["y"] = embedding_3d[:, 1]
    taste_map_3d["z"] = embedding_3d[:, 2]
    taste_map_3d["cluster"] = clusters

    # ----------------------------
    # Save outputs
    # ----------------------------
    cluster_assignments.to_parquet(CLUSTER_ASSIGNMENTS_OUT, index=False)
    metrics_df.to_parquet(CLUSTER_METRICS_OUT, index=False)
    stability_df.to_parquet(CLUSTER_STABILITY_OUT, index=False)
    cluster_diff.to_parquet(CLUSTER_PROFILE_DIFF_OUT, index=False)
    taste_map_2d.to_parquet(TASTE_MAP_2D_OUT, index=False)
    taste_map_3d.to_parquet(TASTE_MAP_3D_OUT, index=False)

    print("[05_cluster_users] Final k used:", final_k)
    print("[05_cluster_users] Saved:", CLUSTER_ASSIGNMENTS_OUT)
    print("[05_cluster_users] Saved:", CLUSTER_METRICS_OUT)
    print("[05_cluster_users] Saved:", CLUSTER_STABILITY_OUT)
    print("[05_cluster_users] Saved:", CLUSTER_PROFILE_DIFF_OUT)
    print("[05_cluster_users] Saved:", TASTE_MAP_2D_OUT)
    print("[05_cluster_users] Saved:", TASTE_MAP_3D_OUT)


if __name__ == "__main__":
    main()