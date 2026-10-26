"""
Subscriber Churn Modeling module.
Implements model training, Stratified CV, hyperparameter tuning,
model selection, and pipeline serialization.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.feature_engineering import engineer_churn_features
from src.metrics import compute_classification_metrics, get_roc_pr_curves
from src.preprocessing import (
    CHURN_BINARY_PASSTHROUGH,
    CHURN_CATEGORICAL_FEATURES,
    CHURN_NUMERIC_FEATURES,
    build_preprocessor,
)


def prepare_churn_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Apply feature engineering and perform stratified train-test split.
    Feature engineering normalization bounds are computed ONLY on training data to prevent leakage.
    """
    # Split raw first to compute bounds strictly on train
    train_raw, test_raw = train_test_split(
        df,
        test_size=test_size,
        stratify=df["churned"],
        random_state=random_state,
    )

    bounds = {
        "watch_min": float(train_raw["watch_hours"].min()),
        "watch_max": float(train_raw["watch_hours"].quantile(0.99)),
        "recency_min": float(train_raw["last_login_days"].min()),
        "recency_max": float(train_raw["last_login_days"].max()),
        "profiles_min": float(train_raw["number_of_profiles"].min()),
        "profiles_max": float(train_raw["number_of_profiles"].max()),
    }

    train_df = engineer_churn_features(train_raw, bounds=bounds)
    test_df = engineer_churn_features(test_raw, bounds=bounds)

    feature_cols = (
        CHURN_NUMERIC_FEATURES
        + CHURN_CATEGORICAL_FEATURES
        + CHURN_BINARY_PASSTHROUGH
    )

    X_train = train_df[feature_cols]
    y_train = train_df["churned"]
    X_test = test_df[feature_cols]
    y_test = test_df["churned"]

    return X_train, X_test, y_train, y_test, feature_cols


def get_candidate_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Return candidate classifiers and their hyperparameter grids for tuning.
    """
    return {
        "Baseline (Dummy)": {
            "model": DummyClassifier(strategy="most_frequent"),
            "params": {},
        },
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                solver="lbfgs",
            ),
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                random_state=random_state,
            ),
            "params": {
                "classifier__max_depth": [3, 5, 8, 12],
                "classifier__min_samples_split": [5, 20, 50],
                "classifier__min_samples_leaf": [2, 10, 20],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=random_state,
                n_estimators=100,
            ),
            "params": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [6, 10, 15],
                "classifier__min_samples_split": [5, 20],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                random_state=random_state,
            ),
            "params": {
                "classifier__n_estimators": [100, 150],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
            },
        },
    }


def train_and_evaluate_churn_models(
    df: pd.DataFrame,
    models_dir: str = "models",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute complete churn model benchmarking pipeline:
    1. Stratified split with isolated feature engineering bounds.
    2. ColumnTransformer preprocessing.
    3. Stratified 5-Fold cross validation and hyperparameter tuning.
    4. Out-of-sample evaluation on held-out test set.
    5. Final model selection based on ROC-AUC and F1-score.
    6. Serialization of best model pipeline and metrics.
    """
    X_train, X_test, y_train, y_test, feature_cols = prepare_churn_data(
        df, random_state=random_state
    )

    preprocessor = build_preprocessor(
        numeric_features=CHURN_NUMERIC_FEATURES,
        categorical_features=CHURN_CATEGORICAL_FEATURES,
        binary_features=CHURN_BINARY_PASSTHROUGH,
    )

    candidates = get_candidate_classifiers(random_state=random_state)
    results = {}
    fitted_pipelines = {}
    best_model_name = None
    best_score = -1.0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    print("--- BENCHMARKING CHURN MODELS ---")
    for name, config in candidates.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", config["model"]),
            ]
        )

        params = config["params"]
        if params:
            search = GridSearchCV(
                pipe,
                param_grid=params,
                cv=cv,
                scoring="roc_auc",
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            best_params = search.best_params_
            cv_score = float(search.best_score_)
        else:
            pipe.fit(X_train, y_train)
            best_pipe = pipe
            best_params = {}
            cv_score = 0.5

        # Evaluate on test set
        y_pred = best_pipe.predict(X_test)
        if hasattr(best_pipe, "predict_proba"):
            y_prob = best_pipe.predict_proba(X_test)[:, 1]
        else:
            y_prob = np.zeros(len(y_test))

        test_metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        test_metrics["cv_roc_auc"] = round(cv_score, 4)
        test_metrics["best_params"] = best_params

        # Store curves
        curves = get_roc_pr_curves(y_test, y_prob)
        test_metrics["curves"] = curves

        results[name] = test_metrics
        fitted_pipelines[name] = best_pipe

        print(
            f"Model: {name:<20} | Test ROC-AUC: {test_metrics.get('roc_auc', 'N/A')} | "
            f"F1: {test_metrics['f1_score']} | Acc: {test_metrics['accuracy']}"
        )

        # Select best model based on held-out test ROC-AUC
        current_score = test_metrics.get("roc_auc") or test_metrics["f1_score"]
        if name != "Baseline (Dummy)" and current_score > best_score:
            best_score = current_score
            best_model_name = name

    print(f"\n--> Selected Champion Churn Model: {best_model_name} (Score: {best_score:.4f})")

    # Serialize champion pipeline and metadata
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    champion_pipeline = fitted_pipelines[best_model_name]
    model_path = out_dir / "churn_pipeline.joblib"
    joblib.dump(champion_pipeline, model_path)

    metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "roc_auc",
        "held_out_roc_auc": results[best_model_name].get("roc_auc"),
        "held_out_f1": results[best_model_name]["f1_score"],
        "held_out_accuracy": results[best_model_name]["accuracy"],
        "features": feature_cols,
        "numeric_features": CHURN_NUMERIC_FEATURES,
        "categorical_features": CHURN_CATEGORICAL_FEATURES,
        "binary_features": CHURN_BINARY_PASSTHROUGH,
        "all_model_benchmarks": results,
    }

    metrics_path = out_dir / "churn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "champion_name": best_model_name,
        "champion_pipeline": champion_pipeline,
        "metrics": results,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }
