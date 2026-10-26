"""
Unit tests for feature engineering logic and composite score bounds.
"""

import numpy as np
import pandas as pd
from src.feature_engineering import (
    compute_subscriber_engagement_score,
    engineer_churn_features,
    engineer_retention_features,
)


def test_engagement_score_bounds_and_monotonicity():
    df = pd.DataFrame(
        {
            "watch_hours": [0.0, 20.0, 40.0],
            "last_login_days": [30, 15, 0],
            "number_of_profiles": [1, 3, 5],
        }
    )
    scores = compute_subscriber_engagement_score(df)

    assert len(scores) == 3
    # Bounded between 0 and 100
    assert (scores >= 0).all() and (scores <= 100).all()
    # Monotonically increasing engagement
    assert scores.iloc[0] < scores.iloc[1] < scores.iloc[2]
    # Check extremes
    assert scores.iloc[0] == 0.0
    assert scores.iloc[2] == 100.0


def test_engineer_churn_features():
    df = pd.DataFrame(
        {
            "watch_hours": [10.0, 30.0],
            "last_login_days": [25, 2],
            "monthly_fee": [10.0, 20.0],
            "number_of_profiles": [1, 4],
            "payment_method": ["Gift Card", "Credit Card"],
        }
    )
    feat_df = engineer_churn_features(df)

    assert "engagement_score" in feat_df.columns
    assert "is_high_inactivity" in feat_df.columns
    assert "watch_to_fee_ratio" in feat_df.columns
    assert "is_single_profile" in feat_df.columns
    assert "non_recurring_payment" in feat_df.columns

    # Inactivity flag
    assert feat_df.loc[0, "is_high_inactivity"] == 1
    assert feat_df.loc[1, "is_high_inactivity"] == 0

    # Non-recurring payment flag
    assert feat_df.loc[0, "non_recurring_payment"] == 1
    assert feat_df.loc[1, "non_recurring_payment"] == 0


def test_engineer_retention_features():
    df = pd.DataFrame(
        {
            "show_id": [101, 101, 101],
            "episode_number": [1, 5, 10],
            "pacing_score": [3, 4, 5],
            "hook_strength": [3, 4, 5],
            "cognitive_load": [8, 5, 4],
            "pause_count": [4, 2, 1],
            "dialogue_density": ["high", "medium", "low"],
            "attention_required": ["high", "medium", "low"],
        }
    )
    feat_df = engineer_retention_features(df)

    assert "season_progress_ratio" in feat_df.columns
    assert "is_season_premiere" in feat_df.columns
    assert "friction_index" in feat_df.columns
    assert "dialogue_density_num" in feat_df.columns

    assert feat_df.loc[0, "is_season_premiere"] == 1
    assert feat_df.loc[1, "is_season_premiere"] == 0
    assert feat_df.loc[0, "friction_index"] > feat_df.loc[2, "friction_index"]
