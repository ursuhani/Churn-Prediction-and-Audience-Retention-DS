"""
Batch CSV Prediction Module for Subscriber Churn.
Enables bulk upload, schema validation, batch model inference,
risk tier assignment, revenue-at-risk calculation, and scored CSV export.
"""

from typing import Dict, Tuple, Optional
import io
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.feature_engineering import engineer_churn_features
from src.recommendations import categorize_churn_risk


REQUIRED_COLUMNS = [
    "age",
    "gender",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "region",
    "device",
    "monthly_fee",
    "payment_method",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "favorite_genre",
]

DEFAULT_VALUES = {
    "age": 35,
    "gender": "Male",
    "subscription_type": "Standard",
    "watch_hours": 15.0,
    "last_login_days": 10,
    "region": "North America",
    "device": "TV",
    "monthly_fee": 13.99,
    "payment_method": "Credit Card",
    "number_of_profiles": 2,
    "avg_watch_time_per_day": 0.5,
    "favorite_genre": "Drama",
}


def validate_and_prepare_batch_df(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Validate uploaded dataframe columns. Fill missing non-critical features with defaults.
    Returns cleaned dataframe ready for feature engineering and a diagnostics dict.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df_clean.columns]
    present_cols = [c for c in REQUIRED_COLUMNS if c in df_clean.columns]

    # Impute missing columns with defaults if some features are missing
    for col in missing_cols:
        df_clean[col] = DEFAULT_VALUES[col]

    # Ensure avg_watch_time_per_day is present or calculated
    if "avg_watch_time_per_day" in missing_cols and "watch_hours" in df_clean.columns:
        df_clean["avg_watch_time_per_day"] = (df_clean["watch_hours"] / 30.0).round(2)

    # Ensure customer_id exists
    if "customer_id" not in df_clean.columns:
        df_clean["customer_id"] = [f"batch_{i+1:05d}" for i in range(len(df_clean))]

    diagnostics = {
        "initial_rows": initial_rows,
        "present_required_columns": present_cols,
        "imputed_missing_columns": missing_cols,
        "is_valid": len(present_cols) >= 4,  # Minimum viable signal
    }
    return df_clean, diagnostics


def score_bulk_subscribers(
    df: pd.DataFrame,
    churn_pipeline,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Execute end-to-end bulk scoring on subscriber records:
    1. Feature engineering
    2. Model inference via pipeline
    3. Probability calibration & risk categorization
    4. Expected loss calculations
    """
    df_clean, diagnostics = validate_and_prepare_batch_df(df)

    if not diagnostics["is_valid"]:
        raise ValueError(
            f"Uploaded CSV does not contain sufficient subscriber features. "
            f"Found: {diagnostics['present_required_columns']}. "
            f"At least 4 of the following are required: {REQUIRED_COLUMNS}"
        )

    # Feature engineer
    df_features = engineer_churn_features(df_clean)

    # Predict probabilities and classes
    probs = churn_pipeline.predict_proba(df_features)[:, 1]
    preds = churn_pipeline.predict(df_features)

    # Attach results
    df_scored = df_clean.copy()
    df_scored["predicted_churn_prob"] = probs.round(4)
    df_scored["predicted_churn"] = preds
    df_scored["risk_tier"] = [categorize_churn_risk(p) for p in probs]
    df_scored["monthly_revenue_at_risk"] = (
        df_scored["monthly_fee"] * df_scored["predicted_churn_prob"]
    ).round(2)
    df_scored["annual_revenue_at_risk"] = (
        df_scored["monthly_revenue_at_risk"] * 12.0
    ).round(2)

    # Aggregate metrics
    high_risk_count = int((df_scored["risk_tier"] == "High Risk").sum())
    med_risk_count = int((df_scored["risk_tier"] == "Medium Risk").sum())
    low_risk_count = int((df_scored["risk_tier"] == "Low Risk").sum())
    total_rev_at_risk = float(df_scored["monthly_revenue_at_risk"].sum())

    summary = {
        "total_subscribers": len(df_scored),
        "high_risk_count": high_risk_count,
        "med_risk_count": med_risk_count,
        "low_risk_count": low_risk_count,
        "high_risk_percentage": round(100.0 * high_risk_count / max(len(df_scored), 1), 1),
        "mean_churn_probability": round(float(probs.mean()), 4),
        "total_monthly_revenue_at_risk": round(total_rev_at_risk, 2),
        "total_annual_revenue_at_risk": round(total_rev_at_risk * 12.0, 2),
    }

    return df_scored, summary


def generate_sample_csv_template() -> str:
    """
    Generate sample CSV template with 5 realistic subscriber rows for quick testing.
    """
    sample_df = pd.DataFrame(
        [
            {
                "customer_id": "SUB-9001",
                "age": 32,
                "gender": "Male",
                "subscription_type": "Basic",
                "watch_hours": 4.5,
                "last_login_days": 24,
                "region": "North America",
                "device": "Mobile",
                "monthly_fee": 8.99,
                "payment_method": "Gift Card",
                "number_of_profiles": 1,
                "avg_watch_time_per_day": 0.15,
                "favorite_genre": "Action",
            },
            {
                "customer_id": "SUB-9002",
                "age": 45,
                "gender": "Female",
                "subscription_type": "Premium",
                "watch_hours": 32.0,
                "last_login_days": 2,
                "region": "Europe",
                "device": "TV",
                "monthly_fee": 19.99,
                "payment_method": "Credit Card",
                "number_of_profiles": 4,
                "avg_watch_time_per_day": 1.07,
                "favorite_genre": "Drama",
            },
            {
                "customer_id": "SUB-9003",
                "age": 28,
                "gender": "Other",
                "subscription_type": "Standard",
                "watch_hours": 14.0,
                "last_login_days": 12,
                "region": "Asia",
                "device": "Laptop",
                "monthly_fee": 13.99,
                "payment_method": "PayPal",
                "number_of_profiles": 2,
                "avg_watch_time_per_day": 0.47,
                "favorite_genre": "Comedy",
            },
            {
                "customer_id": "SUB-9004",
                "age": 58,
                "gender": "Male",
                "subscription_type": "Basic",
                "watch_hours": 2.0,
                "last_login_days": 28,
                "region": "South America",
                "device": "Tablet",
                "monthly_fee": 8.99,
                "payment_method": "Crypto",
                "number_of_profiles": 1,
                "avg_watch_time_per_day": 0.07,
                "favorite_genre": "Horror",
            },
            {
                "customer_id": "SUB-9005",
                "age": 39,
                "gender": "Female",
                "subscription_type": "Standard",
                "watch_hours": 21.5,
                "last_login_days": 5,
                "region": "North America",
                "device": "TV",
                "monthly_fee": 13.99,
                "payment_method": "Debit Card",
                "number_of_profiles": 3,
                "avg_watch_time_per_day": 0.72,
                "favorite_genre": "Sci-Fi",
            },
        ]
    )
    return sample_df.to_csv(index=False)


def render_batch_prediction_page(churn_pipeline):
    """
    Render the Batch CSV Prediction page in the Streamlit UI.
    """
    st.markdown(
        """<div class="page-heading">Batch CSV Prediction</div>
        <div class="page-subheading">Upload external subscriber cohorts to score churn probability, classify risk tiers, and quantify revenue at risk.</div>""",
        unsafe_allow_html=True,
    )

    c_upload, c_template = st.columns([2.5, 1.2])

    with c_upload:
        uploaded_file = st.file_uploader(
            "Select Subscriber CSV File",
            type=["csv"],
            help="Upload a CSV file containing subscriber records.",
        )

    with c_template:
        st.markdown(
            """<div style="background:#FAF8F5;border:1px solid #E3D8CF;border-radius:8px;padding:12px 14px;margin-top:2px;">
              <div style="font-size:0.80rem;font-weight:600;color:#30343B;margin-bottom:3px;">Need a test template?</div>
              <div style="font-size:0.73rem;color:#6B625C;margin-bottom:8px;">Download a sample CSV with 5 formatted records.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        template_csv = generate_sample_csv_template()
        st.download_button(
            label="Download Template CSV",
            data=template_csv,
            file_name="subscriber_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.markdown(f"<div style='font-size:0.80rem;color:#6B625C;margin-top:6px;margin-bottom:12px;'>Uploaded <strong>{len(input_df):,}</strong> records &middot; {len(input_df.columns)} columns</div>", unsafe_allow_html=True)

            with st.spinner("Scoring subscriber cohort with champion Gradient Boosting pipeline..."):
                df_scored, summary = score_bulk_subscribers(input_df, churn_pipeline)

            # KPI Row
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(
                    f"""<div class="kpi-card">
                      <div class="kpi-top"><span class="kpi-label">Records Scored</span></div>
                      <div class="kpi-value">{summary['total_subscribers']:,}</div>
                      <div class="kpi-caption">Batch size processed</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with k2:
                st.markdown(
                    f"""<div class="kpi-card">
                      <div class="kpi-top"><span class="kpi-label">High-Risk Accounts</span></div>
                      <div class="kpi-value" style="color:#C25E50;">{summary['high_risk_count']:,}</div>
                      <div class="kpi-caption">{summary['high_risk_percentage']}% of uploaded cohort</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with k3:
                st.markdown(
                    f"""<div class="kpi-card">
                      <div class="kpi-top"><span class="kpi-label">Mean Churn Probability</span></div>
                      <div class="kpi-value">{summary['mean_churn_probability']:.1%}</div>
                      <div class="kpi-caption">Average cohort attrition risk</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with k4:
                st.markdown(
                    f"""<div class="kpi-card">
                      <div class="kpi-top"><span class="kpi-label">Monthly Exposure</span></div>
                      <div class="kpi-value">${summary['total_monthly_revenue_at_risk']:,.0f}</div>
                      <div class="kpi-caption">${summary['total_annual_revenue_at_risk']:,.0f} annual run-rate</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            # Results Section
            r_col1, r_col2 = st.columns([1.1, 2.0])

            with r_col1:
                st.markdown(
                    """<div class="s-card"><div class="s-card-title">Batch Risk Distribution</div><div class="s-card-desc">Proportion of subscribers in each risk category.</div></div>""",
                    unsafe_allow_html=True,
                )
                donut_df = pd.DataFrame(
                    [
                        {"Tier": "High Risk", "Count": summary["high_risk_count"]},
                        {"Tier": "Medium Risk", "Count": summary["med_risk_count"]},
                        {"Tier": "Low Risk", "Count": summary["low_risk_count"]},
                    ]
                )
                fig_donut = px.pie(
                    donut_df,
                    names="Tier",
                    values="Count",
                    color="Tier",
                    color_discrete_map={"High Risk": "#C25E50", "Medium Risk": "#D49E5B", "Low Risk": "#729B79"},
                    hole=0.55,
                )
                fig_donut.update_traces(textposition="outside", textinfo="label+percent", textfont=dict(size=10, color="#30343B"))
                fig_donut.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif", color="#6B625C", size=11),
                    height=270,
                    margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            with r_col2:
                st.markdown(
                    """<div class="s-card"><div class="s-card-title">Scored Records Preview</div><div class="s-card-desc">Top at-risk subscribers sorted by predicted churn probability.</div></div>""",
                    unsafe_allow_html=True,
                )
                preview_cols = [
                    c for c in [
                        "customer_id",
                        "subscription_type",
                        "monthly_fee",
                        "watch_hours",
                        "last_login_days",
                        "payment_method",
                        "predicted_churn_prob",
                        "risk_tier",
                        "monthly_revenue_at_risk",
                    ] if c in df_scored.columns
                ]
                df_sorted = df_scored.sort_values("predicted_churn_prob", ascending=False)
                st.dataframe(df_sorted[preview_cols].head(100), use_container_width=True, hide_index=True)

            # Export button
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            csv_data = df_scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download Complete Scored CSV ({len(df_scored):,} Records)",
                data=csv_data,
                file_name="scored_subscribers_batch.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as err:
            st.error(f"Error processing batch CSV: {err}")
    else:
        st.info("Upload a CSV file or download the template above to run batch scoring.")
