"""
Model Explainability and Feature Importance module.
Implements Permutation Importance, Model-Specific Global Importances,
SHAP analysis with an automated graceful fallback, and local prediction explanations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.preprocessing import get_feature_names_from_preprocessor


def compute_global_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    top_n: int = 15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Calculate global feature importance using:
    1. Model-native feature importances (e.g., MDI for trees, absolute weights for linear models).
    2. Model-agnostic Permutation Importance on held-out test data.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    X_trans = preprocessor.transform(X_test)

    # 1. Native importances
    native_importances = {}
    if hasattr(classifier, "feature_importances_"):
        raw_imp = classifier.feature_importances_
        native_importances = {
            name: round(float(val), 4)
            for name, val in sorted(
                zip(feature_names, raw_imp), key=lambda x: x[1], reverse=True
            )[:top_n]
        }
    elif hasattr(classifier, "coef_"):
        raw_coef = np.abs(classifier.coef_[0])
        native_importances = {
            name: round(float(val), 4)
            for name, val in sorted(
                zip(feature_names, raw_coef), key=lambda x: x[1], reverse=True
            )[:top_n]
        }

    # 2. Permutation Importance
    perm = permutation_importance(
        classifier,
        X_trans,
        y_test,
        n_repeats=5,
        random_state=random_state,
        scoring="roc_auc",
    )
    perm_importances = {
        name: {
            "mean": round(float(perm.importances_mean[i]), 4),
            "std": round(float(perm.importances_std[i]), 4),
        }
        for i, name in enumerate(feature_names)
    }
    # Sort permutation importances
    sorted_perm = dict(
        sorted(perm_importances.items(), key=lambda x: x[1]["mean"], reverse=True)[:top_n]
    )

    return {
        "native_importances": native_importances,
        "permutation_importances": sorted_perm,
        "disclaimer": (
            "Feature importance reflects statistical association with model predictions. "
            "It does NOT establish causal mechanisms or guarantee that altering a feature will reduce churn."
        ),
    }


def explain_local_prediction(
    pipeline: Pipeline,
    user_record_df: pd.DataFrame,
    population_baseline_df: pd.DataFrame,
    top_n: int = 6,
) -> Dict[str, Any]:
    """
    Generate local prediction explanation for an individual user profile.
    Attempts SHAP explanation first; if unavailable, seamlessly uses
    feature-attribution deviation against population baselines.
    """
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    feature_names = get_feature_names_from_preprocessor(preprocessor)

    # Calculate predicted probability
    prob = float(pipeline.predict_proba(user_record_df)[:, 1][0])

    # Try SHAP
    shap_used = False
    contributions = []

    try:
        import shap

        X_user_trans = preprocessor.transform(user_record_df)
        X_pop_trans = preprocessor.transform(population_baseline_df.sample(min(100, len(population_baseline_df)), random_state=42))

        if hasattr(classifier, "predict_proba"):
            explainer = shap.Explainer(classifier, X_pop_trans)
            shap_values = explainer(X_user_trans)
            vals = shap_values.values[0]
            if len(vals.shape) > 1 and vals.shape[1] == 2:
                vals = vals[:, 1]
            for name, val in zip(feature_names, vals):
                contributions.append(
                    {
                        "feature": name,
                        "impact": round(float(val), 4),
                        "direction": "Increases Risk" if val > 0 else "Decreases Risk",
                    }
                )
            shap_used = True
    except Exception:
        shap_used = False

    # Robust Fallback if SHAP is not used
    if not shap_used:
        X_user_trans = preprocessor.transform(user_record_df)[0]
        # X_user_trans features in standardized space reflect deviations from mean (Z-scores)
        if hasattr(classifier, "feature_importances_"):
            weights = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            weights = classifier.coef_[0]
        else:
            weights = np.ones(len(feature_names)) / len(feature_names)

        for name, z_val, w in zip(feature_names, X_user_trans, weights):
            impact = float(z_val * w)
            if abs(impact) > 1e-4:
                contributions.append(
                    {
                        "feature": name,
                        "impact": round(impact, 4),
                        "direction": "Increases Risk" if impact > 0 else "Decreases Risk",
                    }
                )

    # Sort contributions by magnitude
    contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)[:top_n]

    return {
        "churn_probability": round(prob, 4),
        "method": "SHAP" if shap_used else "Standardized Feature Attribution (Fallback)",
        "contributions": contributions,
        "disclaimer": (
            "Local explanations indicate which attributes most heavily swayed this specific prediction "
            "relative to the training population average. These are observational model drivers, not guaranteed causes."
        ),
    }
