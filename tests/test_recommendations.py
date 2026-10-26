"""
Unit tests for rule-based recommendation logic and revenue-at-risk calculations.
"""

from src.recommendations import (
    categorize_churn_risk,
    generate_episode_retention_recommendations,
    generate_subscriber_recommendations,
)


def test_categorize_churn_risk_tiers():
    assert categorize_churn_risk(0.15) == "Low Risk"
    assert categorize_churn_risk(0.35) == "Medium Risk"
    assert categorize_churn_risk(0.50) == "Medium Risk"
    assert categorize_churn_risk(0.65) == "High Risk"
    assert categorize_churn_risk(0.95) == "High Risk"


def test_generate_subscriber_recommendations_high_risk_inactive():
    features = {
        "monthly_fee": 14.99,
        "last_login_days": 25,
        "watch_hours": 3.0,
        "payment_method": "Gift Card",
        "number_of_profiles": 1,
    }
    rec = generate_subscriber_recommendations(0.85, features)

    assert rec["risk_tier"] == "High Risk"
    assert rec["churn_probability"] == 0.85
    assert rec["monthly_revenue_at_risk"] == round(14.99 * 0.85, 2)
    assert rec["annual_revenue_at_risk"] == round(14.99 * 12 * 0.85, 2)
    assert len(rec["recommended_interventions"]) > 0

    # Ensure actions address inactivity and manual payment
    actions = [i["action"] for i in rec["recommended_interventions"]]
    assert any("Inactivity" in a for a in actions)
    assert any("Auto-Pay" in a for a in actions)


def test_generate_episode_recommendations():
    episode_info = {
        "title": "Pilot",
        "episode_number": 1,
        "pacing_score": 2,
        "hook_strength": 2,
        "cognitive_load": 8,
        "pause_count": 6,
    }
    rec = generate_episode_retention_recommendations(0.68, episode_info)

    assert rec["drop_off_probability"] == 0.68
    assert len(rec["editorial_suggestions"]) >= 2
    # Ensure hook and premiere advice is surfaced
    suggestions_text = " ".join(rec["editorial_suggestions"])
    assert "hook" in suggestions_text.lower()
