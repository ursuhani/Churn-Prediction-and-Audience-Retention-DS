"""
Evaluation metrics and statistical hypothesis testing module.
Provides rigorous academic metrics and hypothesis testing using SciPy.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, Union[float, int, List]]:
    """
    Compute comprehensive classification metrics for binary targets.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (
        int(cm[0, 0]),
        int(cm[0, 1]),
        int(cm[1, 0]),
        int(cm[1, 1]),
    )
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "confusion_matrix": cm.tolist(),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            brier = float(brier_score_loss(y_true, y_prob))
            metrics["roc_auc"] = round(auc, 4)
            metrics["brier_score"] = round(brier, 4)
        except Exception:
            metrics["roc_auc"] = None
            metrics["brier_score"] = None

    return metrics


def get_roc_pr_curves(
    y_true: Union[np.ndarray, pd.Series],
    y_prob: Union[np.ndarray, pd.Series],
) -> Dict[str, List[float]]:
    """
    Compute ROC and Precision-Recall curve coordinates.
    """
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_thresholds": roc_thresh.tolist(),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "pr_thresholds": pr_thresh.tolist(),
    }


def perform_welch_ttest(
    series_group_0: pd.Series,
    series_group_1: pd.Series,
    feature_name: str,
    group_0_name: str = "Retained",
    group_1_name: str = "Churned",
) -> Dict[str, Union[str, float, int]]:
    """
    Perform Welch's t-test (unequal variances assumed) between two independent cohorts.
    """
    s0 = series_group_0.dropna()
    s1 = series_group_1.dropna()

    t_stat, p_val = stats.ttest_ind(s0, s1, equal_var=False)

    mean_0 = float(s0.mean())
    std_0 = float(s0.std())
    mean_1 = float(s1.mean())
    std_1 = float(s1.std())

    is_significant = p_val < 0.05

    return {
        "feature": feature_name,
        "test": "Welch's t-test",
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_val),
        "is_significant": bool(is_significant),
        f"mean_{group_0_name}": round(mean_0, 3),
        f"std_{group_0_name}": round(std_0, 3),
        f"n_{group_0_name}": len(s0),
        f"mean_{group_1_name}": round(mean_1, 3),
        f"std_{group_1_name}": round(std_1, 3),
        f"n_{group_1_name}": len(s1),
        "interpretation": (
            f"Statistically significant difference (p < 0.05) in {feature_name} between {group_0_name} and {group_1_name}."
            if is_significant
            else f"No statistically significant difference (p >= 0.05) detected in {feature_name}."
        ),
    }


def perform_chisquare_test(
    df: pd.DataFrame,
    cat_col: str,
    target_col: str,
) -> Dict[str, Union[str, float, int]]:
    """
    Perform Pearson's Chi-Square Test of Independence between a categorical variable and binary target.
    """
    contingency = pd.crosstab(df[cat_col], df[target_col])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)

    is_significant = p_val < 0.05

    # Cramér's V for effect size
    n = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0

    return {
        "feature": cat_col,
        "test": "Chi-Square Test of Independence",
        "chi2_statistic": round(float(chi2), 4),
        "p_value": float(p_val),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "is_significant": bool(is_significant),
        "interpretation": (
            f"Significant association (p < 0.05, Cramér's V = {round(cramers_v, 3)}) between {cat_col} and {target_col}."
            if is_significant
            else f"No significant dependency detected between {cat_col} and {target_col}."
        ),
    }
