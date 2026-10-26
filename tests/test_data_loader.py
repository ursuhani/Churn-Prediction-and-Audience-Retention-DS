"""
Unit tests for data loading and schema validation.
"""

import pytest
import pandas as pd
from src.data_loader import (
    EXPECTED_CHURN_COLUMNS,
    EXPECTED_RETENTION_COLUMNS,
    load_churn_data,
    load_retention_data,
    validate_schema,
    summarize_data_quality,
)


def test_load_churn_data_schema():
    df = load_churn_data()
    assert df is not None
    assert len(df) == 5000
    for col in EXPECTED_CHURN_COLUMNS:
        assert col in df.columns
    assert set(df["churned"].unique()).issubset({0, 1})


def test_load_retention_data_schema():
    df = load_retention_data()
    assert df is not None
    assert len(df) == 33171
    for col in EXPECTED_RETENTION_COLUMNS:
        assert col in df.columns
    assert set(df["drop_off"].unique()).issubset({0, 1})


def test_validate_schema_missing_column_raises():
    dummy_df = pd.DataFrame({"age": [25], "gender": ["Male"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(dummy_df, ["age", "non_existent_col"], "test_dataset")


def test_validate_schema_invalid_target_raises():
    dummy_df = pd.DataFrame({"churned": [0, 1, 99]})
    with pytest.raises(ValueError, match="must be binary"):
        validate_schema(dummy_df, ["churned"], "test_dataset", target_col="churned")


def test_summarize_data_quality():
    df = load_churn_data()
    summary = summarize_data_quality(df, "netflix_churn")
    assert summary["shape"] == (5000, 14)
    assert summary["duplicates"] == 0
    assert isinstance(summary["numeric_features"], list)
    assert len(summary["numeric_features"]) > 0
