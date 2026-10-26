"""
Unit tests for preprocessing transformers and ColumnTransformers.
"""

import numpy as np
import pandas as pd
from src.preprocessing import (
    CHURN_BINARY_PASSTHROUGH,
    CHURN_CATEGORICAL_FEATURES,
    CHURN_NUMERIC_FEATURES,
    build_preprocessor,
    get_feature_names_from_preprocessor,
)


def test_build_and_fit_preprocessor():
    dummy_data = pd.DataFrame(
        {
            "age": [30, 45, 60],
            "watch_hours": [12.5, 25.0, 5.0],
            "last_login_days": [10, 2, 28],
            "monthly_fee": [9.99, 14.99, 19.99],
            "number_of_profiles": [1, 2, 4],
            "avg_watch_time_per_day": [0.4, 0.8, 0.15],
            "engagement_score": [50.0, 85.0, 20.0],
            "watch_to_fee_ratio": [1.25, 1.67, 0.25],
            "watch_hours_per_profile": [12.5, 12.5, 1.25],
            "gender": ["Male", "Female", "Other"],
            "subscription_type": ["Basic", "Standard", "Premium"],
            "region": ["Asia", "Europe", "North America"],
            "device": ["TV", "Mobile", "Tablet"],
            "payment_method": ["Credit Card", "PayPal", "Gift Card"],
            "favorite_genre": ["Action", "Drama", "Comedy"],
            "is_high_inactivity": [0, 0, 1],
            "is_single_profile": [1, 0, 0],
            "non_recurring_payment": [0, 0, 1],
        }
    )

    preprocessor = build_preprocessor(
        numeric_features=CHURN_NUMERIC_FEATURES,
        categorical_features=CHURN_CATEGORICAL_FEATURES,
        binary_features=CHURN_BINARY_PASSTHROUGH,
    )

    transformed = preprocessor.fit_transform(dummy_data)
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > len(CHURN_NUMERIC_FEATURES)

    feature_names = get_feature_names_from_preprocessor(preprocessor)
    assert len(feature_names) == transformed.shape[1]
