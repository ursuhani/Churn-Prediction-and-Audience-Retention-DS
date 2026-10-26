"""
Unit tests for model loading, prediction probabilities, and inference integrity.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_churn_data, load_retention_data
from src.feature_engineering import engineer_churn_features, engineer_retention_features


def test_churn_pipeline_loading_and_prediction():
    model_path = Path("models/churn_pipeline.joblib")
    assert model_path.exists(), "Trained churn pipeline file missing!"

    pipeline = joblib.load(model_path)
    df = load_churn_data().head(10)
    df_feat = engineer_churn_features(df)

    probs = pipeline.predict_proba(df_feat)
    preds = pipeline.predict(df_feat)

    assert probs.shape == (10, 2)
    assert preds.shape == (10,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    # Row probabilities sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_retention_pipeline_loading_and_prediction():
    full_path = Path("models/retention_pipeline.joblib")
    early_path = Path("models/retention_early_pipeline.joblib")

    assert full_path.exists(), "Trained retention full pipeline missing!"
    assert early_path.exists(), "Trained retention early pipeline missing!"

    pipe_full = joblib.load(full_path)
    pipe_early = joblib.load(early_path)

    df_ret = load_retention_data().head(10)
    df_feat = engineer_retention_features(df_ret)

    probs_full = pipe_full.predict_proba(df_feat)[:, 1]
    probs_early = pipe_early.predict_proba(df_feat)[:, 1]

    assert len(probs_full) == 10
    assert len(probs_early) == 10
    assert (probs_full >= 0.0).all() and (probs_full <= 1.0).all()
    assert (probs_early >= 0.0).all() and (probs_early <= 1.0).all()


def test_segmentation_kmeans_loading_and_projection():
    seg_path = Path("models/segmentation_kmeans.joblib")
    assert seg_path.exists(), "Segmentation joblib artifact missing!"

    artifact = joblib.load(seg_path)
    kmeans = artifact["kmeans"]
    scaler = artifact["scaler"]
    pca = artifact["pca"]
    features = artifact["features"]

    df = load_churn_data().head(15)
    df_feat = engineer_churn_features(df)
    X_scaled = scaler.transform(df_feat[features])

    clusters = kmeans.predict(X_scaled)
    pca_coords = pca.transform(X_scaled)

    assert len(clusters) == 15
    assert pca_coords.shape == (15, 2)
    assert set(clusters).issubset(set(range(kmeans.n_clusters)))
