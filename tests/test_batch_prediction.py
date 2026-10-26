"""
Unit tests for batch CSV prediction module.
"""

from pathlib import Path
import io
import joblib
import pandas as pd
import pytest

from src.batch_prediction import (
    generate_sample_csv_template,
    score_bulk_subscribers,
    validate_and_prepare_batch_df,
)


def test_generate_sample_csv_template():
    template_str = generate_sample_csv_template()
    assert isinstance(template_str, str)
    df = pd.read_csv(io.StringIO(template_str))
    assert len(df) == 5
    assert "subscription_type" in df.columns
    assert "watch_hours" in df.columns


def test_validate_and_prepare_batch_df_with_missing_columns():
    sparse_df = pd.DataFrame(
        {
            "age": [25, 40],
            "watch_hours": [10.0, 25.0],
            "last_login_days": [20, 3],
            "monthly_fee": [9.99, 19.99],
        }
    )
    df_clean, diag = validate_and_prepare_batch_df(sparse_df)
    assert diag["is_valid"] is True
    assert "gender" in df_clean.columns
    assert "subscription_type" in df_clean.columns
    assert "customer_id" in df_clean.columns
    assert len(df_clean) == 2


def test_score_bulk_subscribers():
    model_path = Path("models/churn_pipeline.joblib")
    assert model_path.exists(), "Trained churn pipeline missing!"
    churn_pipe = joblib.load(model_path)

    template_str = generate_sample_csv_template()
    test_df = pd.read_csv(io.StringIO(template_str))

    df_scored, summary = score_bulk_subscribers(test_df, churn_pipe)

    assert len(df_scored) == 5
    assert "predicted_churn_prob" in df_scored.columns
    assert "risk_tier" in df_scored.columns
    assert "monthly_revenue_at_risk" in df_scored.columns

    # Bounds
    assert (df_scored["predicted_churn_prob"] >= 0.0).all()
    assert (df_scored["predicted_churn_prob"] <= 1.0).all()
    assert set(df_scored["risk_tier"].unique()).issubset(
        {"Low Risk", "Medium Risk", "High Risk"}
    )

    # Summary integrity
    assert summary["total_subscribers"] == 5
    assert (
        summary["high_risk_count"]
        + summary["med_risk_count"]
        + summary["low_risk_count"]
        == 5
    )
    assert summary["total_monthly_revenue_at_risk"] >= 0.0
