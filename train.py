"""
Master Training and Pipeline Execution Script.
Orchestrates data loading, EDA generation, statistical hypothesis testing,
churn model selection, retention model training, segmentation, and figure export.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.churn_model import train_and_evaluate_churn_models
from src.data_loader import load_churn_data, load_retention_data, summarize_data_quality
from src.eda import generate_all_eda_figures, run_statistical_tests
from src.explainability import compute_global_feature_importance
from src.retention_model import train_and_evaluate_retention_models
from src.segmentation import perform_segmentation


def save_model_figures(
    churn_bundle: dict,
    retention_bundle: dict,
    segmentation_bundle: dict,
    figures_dir: str = "reports/figures",
):
    """
    Generate and save evaluation plots for trained models:
    - Confusion matrices
    - ROC & PR curves
    - Feature importances
    - Audience segmentation clusters (PCA 2D)
    """
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    churn_champ = churn_bundle["champion_pipeline"]
    churn_metrics = churn_bundle["metrics"][churn_bundle["champion_name"]]
    X_test_c = churn_bundle["X_test"]
    y_test_c = churn_bundle["y_test"]

    # 1. Confusion Matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_c = np.array(churn_metrics["confusion_matrix"])
    sns.heatmap(
        cm_c,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=["Retained (0)", "Churned (1)"],
        yticklabels=["Retained (0)", "Churned (1)"],
    )
    axes[0].set_title(f"Churn Model CM ({churn_bundle['champion_name']})")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Retention Full CM
    ret_full_champ_name = retention_bundle["full"]["champion_name"]
    ret_cm = np.array(retention_bundle["full"]["metrics"][ret_full_champ_name]["confusion_matrix"])
    sns.heatmap(
        ret_cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        ax=axes[1],
        xticklabels=["Completed (0)", "Drop-Off (1)"],
        yticklabels=["Completed (0)", "Drop-Off (1)"],
    )
    axes[1].set_title(f"Episode Retention CM ({ret_full_champ_name})")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    # 2. ROC & PR Curves plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    curves_c = churn_metrics["curves"]
    axes[0].plot(
        curves_c["fpr"],
        curves_c["tpr"],
        color="#2980b9",
        lw=2,
        label=f"Churn Model (AUC = {churn_metrics['roc_auc']:.3f})",
    )
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[0].set_title("Receiver Operating Characteristic (ROC)")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        curves_c["recall"],
        curves_c["precision"],
        color="#e67e22",
        lw=2,
        label=f"Churn Model (F1 = {churn_metrics['f1_score']:.3f})",
    )
    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=200)
    plt.close(fig)

    # 3. Global Feature Importance plot
    fig_imp = compute_global_feature_importance(churn_champ, X_test_c, y_test_c, top_n=12)
    native_imp = fig_imp["native_importances"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    keys = list(reversed(list(native_imp.keys())))
    vals = [native_imp[k] for k in keys]
    ax.barh(keys, vals, color="#3498db")
    ax.set_title(f"Global Feature Importance: {churn_bundle['champion_name']}")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=200)
    plt.close(fig)

    # 4. Audience Clusters (PCA 2D projection)
    clustered_df = segmentation_bundle["clustered_df"]
    cluster_names = segmentation_bundle["kmeans"]  # dict
    profiles = segmentation_bundle["profiles"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for cid, prof in profiles.items():
        sub = clustered_df[clustered_df["cluster_id"] == cid]
        ax.scatter(
            sub["pca_x"],
            sub["pca_y"],
            label=f"Cluster {cid}: {prof['cluster_name']} ({prof['percentage']}%)",
            alpha=0.65,
            edgecolors="none",
            s=35,
        )
    ax.set_title("Subscriber Audience Segments (PCA 2D Projection)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "audience_clusters_pca.png", dpi=200)
    plt.close(fig)


def main():
    print("=" * 65)
    print("OTT AUDIENCE RETENTION & CHURN PREDICTOR - TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load data
    print("\n[Step 1/6] Loading datasets and validating schemas...")
    df_churn = load_churn_data()
    df_retention = load_retention_data()

    quality_churn = summarize_data_quality(df_churn, "netflix_customer_churn")
    quality_retention = summarize_data_quality(df_retention, "ott_episode_retention")

    print(f" -> Churn dataset loaded: {df_churn.shape[0]:,} rows, {df_churn.shape[1]} cols")
    print(f" -> Retention dataset loaded: {df_retention.shape[0]:,} rows, {df_retention.shape[1]} cols")

    # 2. EDA and Statistical Hypothesis Testing
    print("\n[Step 2/6] Generating EDA figures and running SciPy statistical tests...")
    eda_figs = generate_all_eda_figures(df_churn, df_retention, "reports/figures")
    stats_results = run_statistical_tests(df_churn)

    with open("reports/statistical_tests.json", "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f" -> Generated {len(eda_figs)} EDA figures in reports/figures/")
    print(f" -> Saved statistical hypothesis test results in reports/statistical_tests.json")

    # 3. Train Churn Models
    print("\n[Step 3/6] Training & benchmarking Subscriber Churn models...")
    churn_bundle = train_and_evaluate_churn_models(df_churn, models_dir="models")

    # 4. Train Retention Models
    print("\n[Step 4/6] Training Episode Retention models (Full & Early-Warning)...")
    retention_bundle = train_and_evaluate_retention_models(df_retention, models_dir="models")

    # 5. Audience Segmentation
    print("\n[Step 5/6] Performing K-Means audience clustering and PCA...")
    segmentation_bundle = perform_segmentation(df_churn, k=4, models_dir="models")

    # 6. Save Model Evaluation Figures
    print("\n[Step 6/6] Generating and exporting model evaluation figures...")
    save_model_figures(churn_bundle, retention_bundle, segmentation_bundle, "reports/figures")

    print("\n" + "=" * 65)
    print("TRAINING PIPELINE COMPLETE! ALL MODELS AND FIGURES EXPORTED.")
    print("=" * 65)
    print(f"Champion Churn Model: {churn_bundle['champion_name']}")
    champ_metrics = churn_bundle["metrics"][churn_bundle["champion_name"]]
    print(f" - ROC-AUC: {champ_metrics.get('roc_auc')}")
    print(f" - F1 Score: {champ_metrics['f1_score']}")
    print(f" - Accuracy: {champ_metrics['accuracy']}")
    print(f" - Precision: {champ_metrics['precision']}")
    print(f" - Recall: {champ_metrics['recall']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
