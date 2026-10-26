"""
OTT Intelligence - Premium SaaS Streaming Analytics Platform.
"""

from pathlib import Path
import json, joblib, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_churn_data, load_retention_data
from src.explainability import explain_local_prediction
from src.feature_engineering import engineer_churn_features, engineer_retention_features
from src.recommendations import (
    categorize_churn_risk, generate_episode_retention_recommendations,
    generate_subscriber_recommendations,
)

st.set_page_config(
    page_title="OTT Intelligence", page_icon="<>",
    layout="wide", initial_sidebar_state="expanded",
)

C_SKY="#A2CDF2"; C_STEEL="#6F8CA6"; C_ICE="#DEEEFC"; C_SLATE="#4D6173"
C_BG_PAGE="#0B1118"; C_BG_SIDEBAR="#111A24"; C_BG_CARD="#172330"
C_BG_INPUT="#202F3D"; C_BORDER="#243140"
C_TEXT_PRI="#DEEEFC"; C_TEXT_SEC="#7FA3BE"; C_ACCENT="#A2CDF2"
print("tokens ok")
