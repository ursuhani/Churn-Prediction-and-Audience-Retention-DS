"""
Episode Retention & Drop-off Modeling module.
Trains both a Full Behavioral Pipeline and an Early-Warning Pre-Completion Pipeline.
Guarantees zero data leakage by excluding derived synthetic probabilities.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.feature_engineering import engineer_retention_features
from src.metrics import compute_classification_metrics, get_roc_pr_curves
from src.preprocessing import (
    RETENTION_EARLY_BINARY,
    RETENTION_EARLY_CATEGORICAL,
    RETENTION_EARLY_NUMERIC,
    RETENTION_FULL_BINARY,
    RETENTION_FULL_CATEGORICAL,
    RETENTION_FULL_NUMERIC,
    build_preprocessor,
)


def prepare_retention_data(
    df: pd.DataFrame,
    mode: str = "full",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Feature engineer episode data and split into train/test.
    Mode 'full': includes in-episode engagement metrics (watch %, pauses, rewinds).
    Mode 'early': pre-completion features only (content characteristics, pacing, cognitive load).
    """
    df_feat = engineer_retention_features(df)

    if mode == "early":
        num_cols = RETENTION_EARLY_NUMERIC
        cat_cols = RETENTION_EARLY_CATEGORICAL
        bin_cols = RETENTION_EARLY_BINARY
    else:
        num_cols = RETENTION_FULL_NUMERIC
        cat_cols = RETENTION_FULL_CATEGORICAL
        bin_cols = RETENTION_FULL_BINARY

    feature_cols = num_cols + cat_cols + bin_cols

    X = df_feat[feature_cols]
    y = df_feat["drop_off"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, feature_cols


def train_and_evaluate_retention_models(
    df: pd.DataFrame,
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train and benchmark both Full and Early-Warning Drop-off models.
    """
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_all = {}

    for mode in ["full", "early"]:
        print(f"\n--- BENCHMARKING RETENTION MODEL ({mode.upper()} MODE) ---")
        X_train, X_test, y_train, y_test, feature_cols = prepare_retention_data(
            df, mode=mode, random_state=random_state
        )

        if mode == "early":
            preprocessor = build_preprocessor(
                numeric_features=RETENTION_EARLY_NUMERIC,
                categorical_features=RETENTION_EARLY_CATEGORICAL,
                binary_features=RETENTION_EARLY_BINARY,
            )
        else:
            preprocessor = build_preprocessor(
                numeric_features=RETENTION_FULL_NUMERIC,
                categorical_features=RETENTION_FULL_CATEGORICAL,
                binary_features=RETENTION_FULL_BINARY,
            )

        # Class imbalance handling: 14.5% positive drop-offs
        candidates = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state,
            ),
        }

        mode_metrics = {}
        best_name = None
        best_f1 = -1.0
        best_pipe = None

        for name, clf in candidates.items():
            pipe = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", clf),
                ]
            )
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
            metrics["curves"] = get_roc_pr_curves(y_test, y_prob)
            mode_metrics[name] = metrics

            print(
                f"[{mode.upper()}] {name:<20} | Test ROC-AUC: {metrics.get('roc_auc')} | "
                f"Recall: {metrics['recall']} | F1: {metrics['f1_score']} | Acc: {metrics['accuracy']}"
            )

            # Evaluate best by F1 on minority class
            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_name = name
                best_pipe = pipe

        filename = "retention_pipeline.joblib" if mode == "full" else "retention_early_pipeline.joblib"
        joblib.dump(best_pipe, out_dir / filename)

        results_all[mode] = {
            "champion_name": best_name,
            "feature_columns": feature_cols,
            "metrics": mode_metrics,
        }

    metrics_path = out_dir / "retention_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results_all, f, indent=2)

    return results_all
