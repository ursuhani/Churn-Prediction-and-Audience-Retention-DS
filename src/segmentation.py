"""
Audience Segmentation module using K-Means Clustering and PCA dimensionality reduction.
Profiles behavioral subscriber cohorts and provides evidence-backed cluster labels.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import engineer_churn_features


SEGMENTATION_FEATURES = [
    "watch_hours",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles",
    "engagement_score",
    "watch_to_fee_ratio",
]


def evaluate_cluster_k(
    X_scaled: np.ndarray,
    k_range: range = range(2, 7),
    random_state: int = 42,
) -> Dict[str, List[float]]:
    """
    Compute Inertia and Silhouette scores across candidate values of k.
    """
    inertias = []
    silhouettes = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X_scaled, labels, sample_size=2000, random_state=random_state))
        silhouettes.append(round(sil, 4))

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
    }


def name_clusters_from_profiles(profile_df: pd.DataFrame) -> Dict[int, str]:
    """
    Assign descriptive, data-grounded segment labels based strictly on observed metrics.
    """
    cluster_names = {}
    for idx, row in profile_df.iterrows():
        churn_rate = row.get("churn_rate", 0.5)
        watch_hrs = row.get("watch_hours_mean", 20.0)
        recency = row.get("last_login_days_mean", 15.0)
        profiles = row.get("number_of_profiles_mean", 2.5)

        if churn_rate > 0.65 or (recency > 18 and watch_hrs < 15):
            name = "Dormant / High Churn Risk"
        elif watch_hrs > 25 and recency < 10:
            name = "Core High-Engagement Streamers"
        elif profiles >= 3.5:
            name = "Family Multi-Profile Accounts"
        elif watch_hrs < 15 and recency < 15:
            name = "Casual Low-Frequency Viewers"
        else:
            name = f"Moderate Streamers (Cohort {idx})"
        cluster_names[int(idx)] = name

    # Ensure unique names
    seen = {}
    unique_names = {}
    for cid, name in cluster_names.items():
        if name in seen:
            seen[name] += 1
            unique_names[cid] = f"{name} (Group {seen[name]})"
        else:
            seen[name] = 1
            unique_names[cid] = name

    return unique_names


def perform_segmentation(
    df: pd.DataFrame,
    k: int = 4,
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fit K-Means clustering, perform PCA 2D projection, profile clusters,
    and serialize models and diagnostics.
    """
    df_feat = engineer_churn_features(df)
    X = df_feat[SEGMENTATION_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster diagnostics for k=2..6
    k_diag = evaluate_cluster_k(X_scaled, random_state=random_state)

    # Fit selected K-Means
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_feat["cluster_id"] = cluster_labels

    # 2D PCA projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    df_feat["pca_x"] = pca_coords[:, 0].round(3)
    df_feat["pca_y"] = pca_coords[:, 1].round(3)

    # Compute cluster profiles
    profiles = {}
    for c in range(k):
        sub = df_feat[df_feat["cluster_id"] == c]
        profiles[c] = {
            "cluster_id": c,
            "count": len(sub),
            "percentage": round(100.0 * len(sub) / len(df_feat), 2),
            "churn_rate": round(float(sub["churned"].mean()), 4),
            "watch_hours_mean": round(float(sub["watch_hours"].mean()), 2),
            "last_login_days_mean": round(float(sub["last_login_days"].mean()), 2),
            "monthly_fee_mean": round(float(sub["monthly_fee"].mean()), 2),
            "number_of_profiles_mean": round(float(sub["number_of_profiles"].mean()), 2),
            "engagement_score_mean": round(float(sub["engagement_score"].mean()), 2),
            "watch_to_fee_mean": round(float(sub["watch_to_fee_ratio"].mean()), 2),
        }

    profile_df = pd.DataFrame(profiles).T
    cluster_names = name_clusters_from_profiles(profile_df)

    for c in range(k):
        profiles[c]["cluster_name"] = cluster_names[c]

    # Save artifacts
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "kmeans": kmeans,
            "scaler": scaler,
            "pca": pca,
            "features": SEGMENTATION_FEATURES,
            "cluster_names": cluster_names,
            "profiles": profiles,
        },
        out_dir / "segmentation_kmeans.joblib",
    )

    with open(out_dir / "segmentation_metrics.json", "w") as f:
        json.dump(
            {
                "selected_k": k,
                "k_diagnostics": k_diag,
                "pca_explained_variance_ratio": [
                    round(float(v), 4) for v in pca.explained_variance_ratio_
                ],
                "profiles": profiles,
            },
            f,
            indent=2,
        )

    return {
        "kmeans": kmeans,
        "scaler": scaler,
        "pca": pca,
        "k_diagnostics": k_diag,
        "profiles": profiles,
        "clustered_df": df_feat,
    }
