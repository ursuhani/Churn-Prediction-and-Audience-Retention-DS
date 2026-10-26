"""
Data loading and schema validation module for OTT Churn & Retention Analytics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


EXPECTED_CHURN_COLUMNS = [
    "customer_id",
    "age",
    "gender",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "region",
    "device",
    "monthly_fee",
    "churned",
    "payment_method",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "favorite_genre",
]

EXPECTED_RETENTION_COLUMNS = [
    "show_id",
    "title",
    "platform",
    "genre",
    "release_year",
    "season_number",
    "episode_number",
    "episode_duration_min",
    "pacing_score",
    "hook_strength",
    "dialogue_density",
    "visual_intensity",
    "avg_watch_percentage",
    "pause_count",
    "rewind_count",
    "skip_intro",
    "cognitive_load",
    "attention_required",
    "night_watch_safe",
    "drop_off",
    "drop_off_probability",
    "retention_risk",
    "dataset_version",
]


def resolve_file_path(filename: str, preferred_dir: str = "data") -> Path:
    """
    Search for a dataset file in preferred_dir, current directory, or parent directory.
    Raises FileNotFoundError with clear instructions if missing.
    """
    candidates = [
        Path(preferred_dir) / filename,
        Path(filename),
        Path("..") / preferred_dir / filename,
        Path(__file__).resolve().parent.parent / preferred_dir / filename,
        Path(__file__).resolve().parent.parent / filename,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()

    raise FileNotFoundError(
        f"Required dataset file '{filename}' was not found.\n"
        f"Checked locations:\n" + "\n".join(f" - {c.resolve()}" for c in candidates) + "\n"
        f"Please place '{filename}' in the '{preferred_dir}/' directory or project root."
    )


def validate_schema(
    df: pd.DataFrame,
    expected_cols: List[str],
    dataset_name: str,
    target_col: Optional[str] = None,
) -> Dict[str, Union[bool, List[str], int, str]]:
    """
    Validate that dataframe contains expected columns, target, and valid records.
    """
    missing_cols = [c for c in expected_cols if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in expected_cols]

    if missing_cols:
        raise ValueError(
            f"Dataset '{dataset_name}' schema mismatch! Missing required columns: {missing_cols}"
        )

    if target_col and target_col in df.columns:
        valid_targets = df[target_col].dropna().unique()
        if not set(valid_targets).issubset({0, 1}):
            raise ValueError(
                f"Target column '{target_col}' in '{dataset_name}' must be binary (0 or 1), got {valid_targets}"
            )

    duplicates = int(df.duplicated().sum())

    return {
        "dataset": dataset_name,
        "is_valid": True,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_expected_columns": missing_cols,
        "extra_columns": extra_cols,
        "duplicate_rows": duplicates,
        "target_col": target_col,
    }


def load_churn_data(
    file_path: Optional[str] = None,
    preferred_dir: str = "data",
) -> pd.DataFrame:
    """
    Load subscriber churn dataset and validate schema.
    """
    path = Path(file_path) if file_path else resolve_file_path("netflix_customer_churn.csv", preferred_dir)
    df = pd.read_csv(path)
    validate_schema(df, EXPECTED_CHURN_COLUMNS, "netflix_customer_churn", target_col="churned")
    return df


def load_retention_data(
    file_path: Optional[str] = None,
    preferred_dir: str = "data",
) -> pd.DataFrame:
    """
    Load episode retention dataset and validate schema.
    """
    path = Path(file_path) if file_path else resolve_file_path("ott_episode_retention.csv", preferred_dir)
    df = pd.read_csv(path)
    validate_schema(df, EXPECTED_RETENTION_COLUMNS, "ott_episode_retention", target_col="drop_off")
    return df


def summarize_data_quality(df: pd.DataFrame, dataset_name: str) -> Dict[str, object]:
    """
    Generate comprehensive data quality diagnostics for report generation.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    null_summary = df.isnull().sum()
    null_counts = {col: int(cnt) for col, cnt in null_summary[null_summary > 0].items()}

    outlier_counts = {}
    for col in num_cols:
        q25, q75 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q75 - q25
        if iqr > 0:
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            outlier_counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
        else:
            outlier_counts[col] = 0

    return {
        "name": dataset_name,
        "shape": df.shape,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "duplicates": int(df.duplicated().sum()),
        "missing_columns": null_counts,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "iqr_outliers": outlier_counts,
    }
