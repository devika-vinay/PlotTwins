"""
step_11_coldstart.py
─────────────────────────────────────────────────────────────────────────
MINI MODEL — COLD-START CLUSTER ASSIGNMENT

Fits a lightweight KMeans on questionnaire-elicitable features only.
No SVD, no computed stats that require rating history.

Questionnaire → Feature mapping:
  Q1  "Pick your favourite genres"              → genre_share_*    (19 cols)
  Q2  "Which decades do you enjoy most?"        → era_*            (13 cols)
  Q3  "Mainstream or niche films?"              → pop_low/mid/high  (3 cols)
  Q4  "How often do you enjoy what you watch?"  → like_rate         (1 col)
  Q5  "Modern films, classics, or both?"        → modern_share,
                                                   classic_share    (2 cols)
  Q6  "Prefer English-language films?"          → english_share     (1 col)
  Q7  "How many genres do you watch?"           → n_unique_genres   (1 col)

  Total: 40 features — all directly elicitable, no rating history needed.

Outputs (saved to CACHE_DIR):
  ├─ coldstart_feature_columns.parquet  — ordered list of feature names
  ├─ coldstart_scaler.pkl               — StandardScaler
  ├─ coldstart_selector.pkl             — VarianceThreshold
  ├─ coldstart_pca.pkl                  — PCA (n_components=15)
  ├─ coldstart_kmeans.pkl               — KMeans (n_clusters=6)
  └─ coldstart_cluster_map.parquet      — cold cluster ID → full model cluster ID
─────────────────────────────────────────────────────────────────────────
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from pipeline.step_00_config import CACHE_DIR

# ── Input artifacts ───────────────────────────────────────────────────
FEATURE_MATRIX_IN = CACHE_DIR / "feature_matrix.parquet"
CLUSTER_ASSIGNMENTS_IN = CACHE_DIR / "cluster_assignments.parquet"

# ── Output artifacts ──────────────────────────────────────────────────
COLDSTART_COLS_OUT = CACHE_DIR / "coldstart_feature_columns.parquet"
COLDSTART_SCALER_OUT = CACHE_DIR / "coldstart_scaler.pkl"
COLDSTART_SELECTOR_OUT = CACHE_DIR / "coldstart_selector.pkl"
COLDSTART_PCA_OUT = CACHE_DIR / "coldstart_pca.pkl"
COLDSTART_KMEANS_OUT = CACHE_DIR / "coldstart_kmeans.pkl"
COLDSTART_MAP_OUT = CACHE_DIR / "coldstart_cluster_map.parquet"

# ── Questionnaire-elicitable feature groups ───────────────────────────
COLDSTART_PREFIXES = [
    "genre_share_",  # Q1 — genre selection
    "era_",  # Q2 — decade preference
    "pop_",  # Q3 — mainstream vs niche
]

COLDSTART_PROFILE_COLS = [
    "like_rate",  # Q4 — enjoyment rate
    "modern_share",  # Q5 — modern vs classic
    "classic_share",  # Q5 — modern vs classic
    "english_share",  # Q6 — language preference
    "n_unique_genres",  # Q7 — genre breadth
]


def select_coldstart_cols(feature_matrix: pd.DataFrame) -> list[str]:
    """Returns ordered list of questionnaire-elicitable columns present in feature_matrix."""
    prefix_cols = [
        c for c in feature_matrix.columns
        if any(c.startswith(p) for p in COLDSTART_PREFIXES)
    ]
    profile_cols = [
        c for c in COLDSTART_PROFILE_COLS
        if c in feature_matrix.columns
    ]
    return prefix_cols + profile_cols


def build_cluster_map(
        kmeans: KMeans,
        feature_matrix: pd.DataFrame,
        cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Maps each cold-start cluster ID to the full-model cluster ID
    that contains the majority of its users (majority vote per cold cluster).
    Ensures cold-start predictions resolve to valid full-model personas.
    """
    cold_labels = pd.Series(
        kmeans.labels_,
        index=feature_matrix["user"].values,
        name="cold_cluster"
    )
    full_labels = cluster_assignments.set_index("user")["cluster"]

    overlap = cold_labels.to_frame().join(
        full_labels.rename("full_cluster"), how="inner"
    )

    cluster_map = (
        overlap.groupby("cold_cluster")["full_cluster"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
        .rename(columns={"cold_cluster": "cold", "full_cluster": "full"})
    )

    return cluster_map


def main():

    if COLDSTART_COLS_OUT.exists() and COLDSTART_MAP_OUT.exists():
        print("[11_coldstart] Cache exists. Skipping cold-start model build.")
        return

    feature_matrix = pd.read_parquet(FEATURE_MATRIX_IN)
    cluster_assignments = pd.read_parquet(CLUSTER_ASSIGNMENTS_IN)

    coldstart_cols = select_coldstart_cols(feature_matrix)

    print("=" * 60)
    print("  COLD-START FEATURE COLUMNS")
    print("=" * 60)
    for c in coldstart_cols:
        print(f"  {c}")
    print(f"\n  Total: {len(coldstart_cols)} features")

    X_raw = (
        feature_matrix[coldstart_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # ── Scale ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── VarianceThreshold ─────────────────────────────────────────────
    selector = VarianceThreshold(threshold=0.0)
    X_filtered = selector.fit_transform(X_scaled)
    n_surviving = X_filtered.shape[1]
    print(f"\n[11_coldstart] After VarianceThreshold: {n_surviving} features")

    # ── PCA ───────────────────────────────────────────────────────────
    n_components = min(15, n_surviving)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_filtered)
    explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"[11_coldstart] PCA → {n_components} components, "
          f"{explained:.1%} variance explained")

    # ── KMeans — same k as full model ─────────────────────────────────
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    print(f"[11_coldstart] KMeans fitted with {n_clusters} clusters")

    # ── Cluster label mapping ─────────────────────────────────────────────
    cluster_map = build_cluster_map(kmeans, feature_matrix, cluster_assignments)

    # Fix: compute agreement at the user level, not the cluster-map level
    cold_labels = pd.Series(
        kmeans.labels_,
        index=feature_matrix["user"].values,
        name="cold_cluster"
    )
    full_labels = cluster_assignments.set_index("user")["cluster"]

    overlap = cold_labels.to_frame().join(
        full_labels.rename("full_cluster"), how="inner"
    )

    # Translate each cold label to its mapped full-model label, then compare
    overlap["mapped"] = overlap["cold_cluster"].map(
        cluster_map.set_index("cold")["full"]
    )
    agreement = (overlap["mapped"] == overlap["full_cluster"]).mean()

    print(f"\n[11_coldstart] Cluster label mapping (cold → full model):")
    print(cluster_map.to_string(index=False))
    print(f"\n[11_coldstart] User-level agreement with full model: {agreement:.1%}")

    # ── Save all artifacts ────────────────────────────────────────────
    pd.DataFrame({"feature": coldstart_cols}).to_parquet(COLDSTART_COLS_OUT, index=False)
    joblib.dump(scaler, COLDSTART_SCALER_OUT)
    joblib.dump(selector, COLDSTART_SELECTOR_OUT)
    joblib.dump(pca, COLDSTART_PCA_OUT)
    joblib.dump(kmeans, COLDSTART_KMEANS_OUT)
    cluster_map.to_parquet(COLDSTART_MAP_OUT, index=False)

    print(f"\n[11_coldstart] Saved all artifacts to {CACHE_DIR}")
    print(f"[11_coldstart] Ready for step_10 cold-start prediction.")


if __name__ == "__main__":
    main()