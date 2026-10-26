"""
Feature engineering module for Subscriber Churn and Episode Retention.
All formulas, transformations, and assumptions are explicitly documented.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# Explicit weights for composite engagement score
ENGAGEMENT_WEIGHTS = {
    "watch_hours": 0.45,
    "login_recency": 0.35,
    "household_profiles": 0.20,
}


def compute_subscriber_engagement_score(
    df: pd.DataFrame,
    watch_col: str = "watch_hours",
    recency_col: str = "last_login_days",
    profiles_col: str = "number_of_profiles",
    watch_min: float = 0.0,
    watch_max: float = 40.0,
    recency_min: float = 0.0,
    recency_max: float = 30.0,
    profiles_min: float = 1.0,
    profiles_max: float = 5.0,
) -> pd.Series:
    """
    Compute a normalized composite Subscriber Engagement Score on [0, 100].

    Formula:
        norm_watch = clip((watch_hours - watch_min) / (watch_max - watch_min), 0, 1)
        norm_recency = 1.0 - clip((last_login_days - recency_min) / (recency_max - recency_min), 0, 1)
        norm_profiles = clip((number_of_profiles - profiles_min) / (profiles_max - profiles_min), 0, 1)

        Score = 100 * (0.45 * norm_watch + 0.35 * norm_recency + 0.20 * norm_profiles)

    Assumptions:
    - Watch hours indicates active platform consumption (45% weight).
    - Login recency indicates active connection (35% weight, inverted so recent = higher).
    - Profiles indicate multi-user household stickiness (20% weight).
    - Uses fixed baseline bounds or empirical quantiles to prevent test set data leakage.
    """
    norm_watch = np.clip(
        (df[watch_col] - watch_min) / max(watch_max - watch_min, 1e-5), 0.0, 1.0
    )
    norm_recency = 1.0 - np.clip(
        (df[recency_col] - recency_min) / max(recency_max - recency_min, 1e-5), 0.0, 1.0
    )
    norm_profiles = np.clip(
        (df[profiles_col] - profiles_min) / max(profiles_max - profiles_min, 1e-5), 0.0, 1.0
    )

    score = 100.0 * (
        ENGAGEMENT_WEIGHTS["watch_hours"] * norm_watch
        + ENGAGEMENT_WEIGHTS["login_recency"] * norm_recency
        + ENGAGEMENT_WEIGHTS["household_profiles"] * norm_profiles
    )
    return score.round(2)


def engineer_churn_features(
    df: pd.DataFrame,
    bounds: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Engineer domain features for Subscriber Churn analysis:
    1. engagement_score: Multi-component engagement metric [0, 100].
    2. is_high_inactivity: Binary flag for subscribers inactive for >= 20 days.
    3. watch_to_fee_ratio: Perceived value = watch hours per monthly dollar spent.
    4. is_single_profile: Household indicator (1 profile vs multi-profile).
    5. non_recurring_payment: Flag for Gift Card and Crypto payment methods.
    6. watch_hours_per_profile: Watch volume normalized by household size.
    """
    df_feat = df.copy()

    # Determine normalization bounds
    if bounds is None:
        w_min = float(df_feat["watch_hours"].min())
        w_max = float(df_feat["watch_hours"].quantile(0.99))
        r_min = float(df_feat["last_login_days"].min())
        r_max = float(df_feat["last_login_days"].max())
        p_min = float(df_feat["number_of_profiles"].min())
        p_max = float(df_feat["number_of_profiles"].max())
    else:
        w_min = bounds.get("watch_min", 0.0)
        w_max = bounds.get("watch_max", 40.0)
        r_min = bounds.get("recency_min", 0.0)
        r_max = bounds.get("recency_max", 30.0)
        p_min = bounds.get("profiles_min", 1.0)
        p_max = bounds.get("profiles_max", 5.0)

    df_feat["engagement_score"] = compute_subscriber_engagement_score(
        df_feat,
        watch_min=w_min,
        watch_max=w_max,
        recency_min=r_min,
        recency_max=r_max,
        profiles_min=p_min,
        profiles_max=p_max,
    )

    df_feat["is_high_inactivity"] = (df_feat["last_login_days"] >= 20).astype(int)
    df_feat["watch_to_fee_ratio"] = (
        df_feat["watch_hours"] / np.maximum(df_feat["monthly_fee"], 1.0)
    ).round(3)
    df_feat["is_single_profile"] = (df_feat["number_of_profiles"] == 1).astype(int)

    # Gift Card & Crypto are manual/non-recurring in real streaming platforms
    df_feat["non_recurring_payment"] = df_feat["payment_method"].isin(
        ["Gift Card", "Crypto"]
    ).astype(int)

    df_feat["watch_hours_per_profile"] = (
        df_feat["watch_hours"] / np.maximum(df_feat["number_of_profiles"], 1)
    ).round(2)

    return df_feat


def engineer_retention_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer domain features for Episode-Level Retention & Drop-off:
    1. season_progress_ratio: Episode position within the show's season.
    2. friction_index: High cognitive load + pauses vs pacing + hook.
    3. is_season_premiere: Episode 1 indicator (entry hook).
    4. dialogue_density_num: Numeric encoding of dialogue density (low=1, med=2, high=3).
    5. attention_required_num: Numeric encoding of attention level (low=1, med=2, high=3).
    6. cognitive_to_pacing_ratio: Mental friction relative to tempo.

    CRITICAL NOTE:
    'drop_off_probability' and 'retention_risk' are explicitly REMOVED from predictors
    to prevent target leakage.
    """
    df_feat = df.copy()

    # Episode position within season
    max_ep_per_show = df_feat.groupby("show_id")["episode_number"].transform("max")
    df_feat["season_progress_ratio"] = (
        df_feat["episode_number"] / np.maximum(max_ep_per_show, 1)
    ).round(3)

    # Premiere indicator
    df_feat["is_season_premiere"] = (df_feat["episode_number"] == 1).astype(int)

    # Density encodings for interaction metrics
    dialogue_map = {"low": 1, "medium": 2, "high": 3}
    attention_map = {"low": 1, "medium": 2, "high": 3}
    df_feat["dialogue_density_num"] = (
        df_feat["dialogue_density"].str.lower().map(dialogue_map).fillna(2)
    )
    df_feat["attention_required_num"] = (
        df_feat["attention_required"].str.lower().map(attention_map).fillna(2)
    )

    # Friction Index = (Cognitive Load + Pauses) / (Pacing + Hook + 1.0)
    df_feat["friction_index"] = (
        (df_feat["cognitive_load"] + df_feat["pause_count"])
        / (df_feat["pacing_score"] + df_feat["hook_strength"] + 1.0)
    ).round(3)

    # Cognitive to pacing ratio
    df_feat["cognitive_to_pacing_ratio"] = (
        df_feat["cognitive_load"] / (df_feat["pacing_score"] + 1.0)
    ).round(3)

    return df_feat
