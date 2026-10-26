"""
Preprocessing pipelines and transformers for OTT Analytics.
Implements scikit-learn ColumnTransformer and Pipeline to strictly prevent data leakage.
"""

from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Feature definitions for Subscriber Churn
CHURN_NUMERIC_FEATURES = [
    "age",
    "watch_hours",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "engagement_score",
    "watch_to_fee_ratio",
    "watch_hours_per_profile",
]

CHURN_CATEGORICAL_FEATURES = [
    "gender",
    "subscription_type",
    "region",
    "device",
    "payment_method",
    "favorite_genre",
]

CHURN_BINARY_PASSTHROUGH = [
    "is_high_inactivity",
    "is_single_profile",
    "non_recurring_payment",
]


# Feature definitions for Episode Retention (Full Model)
RETENTION_FULL_NUMERIC = [
    "episode_duration_min",
    "pacing_score",
    "hook_strength",
    "visual_intensity",
    "avg_watch_percentage",
    "pause_count",
    "rewind_count",
    "cognitive_load",
    "season_progress_ratio",
    "friction_index",
    "cognitive_to_pacing_ratio",
    "dialogue_density_num",
    "attention_required_num",
]

RETENTION_FULL_CATEGORICAL = [
    "platform",
    "genre",
    "dialogue_density",
    "attention_required",
]

RETENTION_FULL_BINARY = [
    "skip_intro",
    "night_watch_safe",
    "is_season_premiere",
]


# Feature definitions for Episode Retention (Early-Warning Model - No Post-Completion Metrics)
# Explicitly excludes 'avg_watch_percentage', 'pause_count', 'rewind_count', 'skip_intro'
# to simulate assessing drop-off risk from content design alone.
RETENTION_EARLY_NUMERIC = [
    "episode_duration_min",
    "pacing_score",
    "hook_strength",
    "visual_intensity",
    "cognitive_load",
    "season_progress_ratio",
    "cognitive_to_pacing_ratio",
    "dialogue_density_num",
    "attention_required_num",
]

RETENTION_EARLY_CATEGORICAL = [
    "platform",
    "genre",
    "dialogue_density",
    "attention_required",
]

RETENTION_EARLY_BINARY = [
    "night_watch_safe",
    "is_season_premiere",
]


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    binary_features: List[str] = None,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer that scales numerics,
    one-hot encodes categoricals, and passes through binary indicators.
    Fitted exclusively on training splits to prevent data leakage.
    """
    transformers = []

    # Numeric pipeline: median imputation + standard scaling
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    transformers.append(("num", num_pipeline, numeric_features))

    # Categorical pipeline: most frequent imputation + OneHotEncoder
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    transformers.append(("cat", cat_pipeline, categorical_features))

    # Binary passthrough: simple binary imputation
    if binary_features:
        bin_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        transformers.append(("bin", bin_pipeline, binary_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_feature_names_from_preprocessor(
    preprocessor: ColumnTransformer,
) -> List[str]:
    """
    Extract transformed feature names from a fitted ColumnTransformer.
    """
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if hasattr(transformer, "named_steps"):
            last_step = transformer.named_steps[list(transformer.named_steps.keys())[-1]]
            if hasattr(last_step, "get_feature_names_out"):
                names = list(last_step.get_feature_names_out(cols))
            else:
                names = list(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            names = list(transformer.get_feature_names_out(cols))
        else:
            names = list(cols)
        feature_names.extend(names)
    return feature_names
