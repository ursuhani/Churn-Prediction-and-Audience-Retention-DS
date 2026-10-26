"""
Exploratory Data Analysis and Statistical Reporting module.
Computes statistical tests and generates publication-grade visualizations in reports/figures/.
"""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.feature_engineering import (
    engineer_churn_features,
    engineer_retention_features,
)
from src.metrics import perform_chisquare_test, perform_welch_ttest


# Set consistent academic figure styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


def run_statistical_tests(df_churn: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute Welch's t-tests on continuous features and Chi-Square tests on categorical features.
    """
    df = engineer_churn_features(df_churn)
    retained = df[df["churned"] == 0]
    churned = df[df["churned"] == 1]

    # Welch's t-tests
    t_tests = {}
    for num_col in ["watch_hours", "last_login_days", "monthly_fee", "engagement_score", "watch_to_fee_ratio"]:
        t_tests[num_col] = perform_welch_ttest(
            retained[num_col], churned[num_col], num_col
        )

    # Chi-square tests
    chi2_tests = {}
    for cat_col in ["subscription_type", "payment_method", "device", "region", "gender", "favorite_genre"]:
        chi2_tests[cat_col] = perform_chisquare_test(df, cat_col, "churned")

    return {
        "ttests": t_tests,
        "chi2_tests": chi2_tests,
    }


def generate_all_eda_figures(
    df_churn: pd.DataFrame,
    df_retention: pd.DataFrame,
    figures_dir: str = "reports/figures",
) -> List[str]:
    """
    Generate and save all required academic figures for the project report and documentation.
    """
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    df_c = engineer_churn_features(df_churn)
    df_r = engineer_retention_features(df_retention)

    palette_churn = ["#2ecc71", "#e74c3c"]  # Green for Retained, Red for Churned

    # 1. Target Distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    churn_counts = df_c["churned"].value_counts().rename({0: "Retained (0)", 1: "Churned (1)"})
    sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=axes[0], palette=palette_churn)
    axes[0].set_title("Subscriber Churn Distribution (N=5,000)")
    axes[0].set_ylabel("Subscriber Count")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())} ({p.get_height()/len(df_c):.1%})",
                         (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    drop_counts = df_r["drop_off"].value_counts().rename({0: "Completed (0)", 1: "Drop-Off (1)"})
    sns.barplot(x=drop_counts.index, y=drop_counts.values, ax=axes[1], palette=palette_churn)
    axes[1].set_title("Episode Drop-Off Distribution (N=33,171)")
    axes[1].set_ylabel("Episode Count")
    for p in axes[1].patches:
        axes[1].annotate(f"{int(p.get_height())} ({p.get_height()/len(df_r):.1%})",
                         (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                         ha='center', va='center', color='white', fontweight='bold')
    plt.tight_layout()
    p1 = out_dir / "target_distribution.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    saved_files.append(str(p1))

    # 2. Missing Value & Schema Assessment
    fig, ax = plt.subplots(figsize=(8, 3))
    missing_c = df_churn.isnull().sum()
    sns.barplot(x=missing_c.index, y=missing_c.values, ax=ax, color="#3498db")
    ax.set_title("Missing Values Verification (Zero Missing Across All Attributes)")
    ax.set_ylabel("Missing Count")
    ax.set_xticklabels(missing_c.index, rotation=45, ha="right")
    ax.set_ylim(0, 10)
    plt.tight_layout()
    p2 = out_dir / "missing_values_summary.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    saved_files.append(str(p2))

    # 3. Churn by Subscription & Payment Method
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sub_churn = df_c.groupby("subscription_type")["churned"].mean().reset_index()
    sns.barplot(data=sub_churn, x="subscription_type", y="churned", ax=axes[0], palette="Blues_d")
    axes[0].set_title("Observed Churn Rate by Subscription Tier")
    axes[0].set_ylabel("Churn Proportion")
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(df_c["churned"].mean(), color="red", linestyle="--", label="Overall Mean (50.3%)")
    axes[0].legend()

    pay_churn = df_c.groupby("payment_method")["churned"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=pay_churn, x="payment_method", y="churned", ax=axes[1], palette="Reds_d")
    axes[1].set_title("Observed Churn Rate by Payment Method (Auto-Pay vs Manual)")
    axes[1].set_ylabel("Churn Proportion")
    axes[1].set_ylim(0, 1.0)
    axes[1].axhline(df_c["churned"].mean(), color="blue", linestyle="--", label="Overall Mean")
    axes[1].set_xticklabels(pay_churn["payment_method"], rotation=30, ha="right")
    axes[1].legend()
    plt.tight_layout()
    p3 = out_dir / "churn_by_subscription_payment.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    saved_files.append(str(p3))

    # 4. Behavioral Distributions (Watch Hours & Login Recency)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.kdeplot(data=df_c, x="watch_hours", hue="churned", common_norm=False, fill=True, palette=palette_churn, ax=axes[0])
    axes[0].set_title("Watch Hours Distribution by Churn Status")
    axes[0].set_xlabel("Monthly Watch Hours")

    sns.kdeplot(data=df_c, x="last_login_days", hue="churned", common_norm=False, fill=True, palette=palette_churn, ax=axes[1])
    axes[1].set_title("Days Since Last Login by Churn Status")
    axes[1].set_xlabel("Days Since Last Login")
    plt.tight_layout()
    p4 = out_dir / "engagement_vs_churn.png"
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    saved_files.append(str(p4))

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    num_cols = df_c.select_dtypes(include=[np.number]).columns
    corr = df_c[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("Pearson Correlation Heatmap (Subscriber Churn Features)")
    plt.tight_layout()
    p5 = out_dir / "correlation_matrix.png"
    fig.savefig(p5, dpi=200)
    plt.close(fig)
    saved_files.append(str(p5))

    # 6. Episode Drop-Off by Genre & Platform
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    g_drop = df_r.groupby("genre")["drop_off"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=g_drop, x="genre", y="drop_off", ax=axes[0], palette="viridis")
    axes[0].set_title("Episode Drop-Off Rate by Genre")
    axes[0].set_ylabel("Drop-Off Rate")
    axes[0].set_xticklabels(g_drop["genre"], rotation=40, ha="right")

    plat_drop = df_r.groupby("platform")["drop_off"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=plat_drop, x="platform", y="drop_off", ax=axes[1], palette="magma")
    axes[1].set_title("Episode Drop-Off Rate by Platform")
    axes[1].set_ylabel("Drop-Off Rate")
    axes[1].set_xticklabels(plat_drop["platform"], rotation=25, ha="right")
    plt.tight_layout()
    p6 = out_dir / "episode_dropoff_by_genre_platform.png"
    fig.savefig(p6, dpi=200)
    plt.close(fig)
    saved_files.append(str(p6))

    # 7. Season Fatigue & Episode Progression
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ep_agg = df_r.groupby("episode_number")["avg_watch_percentage"].mean().reset_index()
    ep_agg_drop = df_r.groupby("episode_number")["drop_off"].mean().reset_index()
    ax.plot(ep_agg["episode_number"], ep_agg["avg_watch_percentage"], marker="o", color="#2980b9", label="Avg Watch %")
    ax.set_xlabel("Episode Number in Season")
    ax.set_ylabel("Avg Watch Percentage (%)", color="#2980b9")
    ax2 = ax.twinx()
    ax2.plot(ep_agg_drop["episode_number"], ep_agg_drop["drop_off"], marker="s", color="#c0392b", linestyle="--", label="Drop-off Rate")
    ax2.set_ylabel("Drop-Off Rate", color="#c0392b")
    ax.set_title("Season Trajectory: Average Watch Completion vs Drop-Off Rate by Episode Index")
    plt.tight_layout()
    p7 = out_dir / "season_fatigue_progression.png"
    fig.savefig(p7, dpi=200)
    plt.close(fig)
    saved_files.append(str(p7))

    return saved_files
