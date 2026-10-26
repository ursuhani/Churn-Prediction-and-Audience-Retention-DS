"""
Retention Recommendation Engine.
Transparent, evidence-backed rule layer translating model predictions and
behavioral markers into actionable, testable retention hypotheses.
"""

from typing import Any, Dict, List, Optional
import pandas as pd


RISK_THRESHOLDS = {
    "low": 0.35,
    "high": 0.65,
}


def categorize_churn_risk(probability: float) -> str:
    """
    Classify predicted probability into standard risk tiers.
    """
    if probability >= RISK_THRESHOLDS["high"]:
        return "High Risk"
    elif probability >= RISK_THRESHOLDS["low"]:
        return "Medium Risk"
    return "Low Risk"


def generate_subscriber_recommendations(
    probability: float,
    user_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate customized retention recommendations and revenue-at-risk diagnostics
    for an individual subscriber based on model score and behavioral signals.
    """
    risk_tier = categorize_churn_risk(probability)
    monthly_fee = float(user_features.get("monthly_fee", 9.99))
    last_login = float(user_features.get("last_login_days", 10))
    watch_hours = float(user_features.get("watch_hours", 15.0))
    payment_method = str(user_features.get("payment_method", "Credit Card"))
    profiles = int(user_features.get("number_of_profiles", 1))

    # Revenue-at-risk calculation
    # Expected value of lost revenue = fee * probability
    monthly_revenue_at_risk = round(monthly_fee * probability, 2)
    annual_revenue_at_risk = round(monthly_fee * 12 * probability, 2)

    interventions: List[Dict[str, str]] = []
    signals: List[str] = []

    # Evaluate specific behavioral triggers
    if last_login >= 20:
        signals.append(f"Severe inactivity ({int(last_login)} days since last platform login)")
        interventions.append(
            {
                "action": "Trigger Inactivity Re-engagement Campaign",
                "channel": "Mobile Push / Direct Email",
                "rationale": "Subscriber is disengaged from daily habit. Prompt with personalized 'Continue Watching' or high-match new releases.",
                "hypothesis": "Sending a targeted title update within 24 hours of 20-day inactivity may prompt a login session for testing in an A/B trial.",
            }
        )
    elif last_login >= 10:
        signals.append(f"Moderate inactivity ({int(last_login)} days since last login)")

    if payment_method in ["Gift Card", "Crypto"]:
        signals.append(f"Non-recurring payment method ({payment_method})")
        interventions.append(
            {
                "action": "Incentivize Recurring Auto-Pay Enrollment",
                "channel": "In-App Renewal Banner",
                "rationale": "Manual payment methods require active intent at each billing cycle, causing passive involuntary churn when balance expires.",
                "hypothesis": "Offering a small one-time credit (e.g. 10% off next 3 months) to link auto-debit/credit card could reduce friction-induced drops.",
            }
        )

    if watch_hours < 8.0:
        signals.append(f"Low viewing volume ({watch_hours:.1f} hrs/month)")
        interventions.append(
            {
                "action": "Surface Shorter Form Content / Quick-Binge Collections",
                "channel": "Home Screen Dynamic Carousel",
                "rationale": "Low watch hours suggest low value realization or lack of time for full movie commitments.",
                "hypothesis": "Highlighting 20-30 minute episodes or curated short series may increase weekly session frequency.",
            }
        )

    if profiles == 1 and risk_tier in ["High Risk", "Medium Risk"]:
        signals.append("Single profile account (Lower household stickiness)")
        interventions.append(
            {
                "action": "Promote Multi-Profile Setup & Family Sharing",
                "channel": "App Settings & Welcome Guide",
                "rationale": "Multi-profile households exhibit higher switching barriers and lower overall churn rates.",
                "hypothesis": "Encouraging secondary profiles for household members may raise retention persistence.",
            }
        )

    # Fallback default intervention if no specific flags triggered
    if not interventions:
        if risk_tier == "High Risk":
            interventions.append(
                {
                    "action": "Personalized Content Recommendation Refresh",
                    "channel": "Email Digest",
                    "rationale": "High predicted risk despite regular metrics suggests potential catalog fatigue.",
                    "hypothesis": "Surfacing diverse genres outside current consumption patterns could revive interest.",
                }
            )
        elif risk_tier == "Medium Risk":
            interventions.append(
                {
                    "action": "Monitor Engagement Velocity",
                    "channel": "Passive Analytics",
                    "rationale": "Subscriber shows borderline engagement signals; intrusive discounts are unwarranted.",
                    "hypothesis": "Track login frequency over the next 14 days before triggering proactive outreach.",
                }
            )
        else:
            interventions.append(
                {
                    "action": "Maintain Core Experience & Loyalty Delighters",
                    "channel": "Standard Platform UX",
                    "rationale": "Highly engaged subscriber with low churn propensity.",
                    "hypothesis": "Avoid disruptive discounting; maintain high-quality streaming recommendations.",
                }
            )

    return {
        "risk_tier": risk_tier,
        "churn_probability": round(probability, 4),
        "monthly_fee": monthly_fee,
        "monthly_revenue_at_risk": monthly_revenue_at_risk,
        "annual_revenue_at_risk": annual_revenue_at_risk,
        "behavioral_signals": signals,
        "recommended_interventions": interventions,
        "disclaimer": (
            "All recommendations are hypotheses designed for controlled A/B testing, "
            "not proven commercial guarantees. Revenue-at-risk is an expected value estimate "
            "based on historical synthetic model probabilities."
        ),
    }


def generate_episode_retention_recommendations(
    drop_off_prob: float,
    episode_row: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate content editorial recommendations for episodes exhibiting high drop-off risk.
    """
    title = episode_row.get("title", "Episode")
    ep_num = int(episode_row.get("episode_number", 1))
    pacing = float(episode_row.get("pacing_score", 5))
    hook = float(episode_row.get("hook_strength", 5))
    cog_load = float(episode_row.get("cognitive_load", 5))
    pauses = float(episode_row.get("pause_count", 2))

    suggestions = []

    if hook < 4:
        suggestions.append(
            "Strengthen cold-open / first 5-minute hook: Low hook strength is heavily correlated with early-episode abandonment."
        )
    if pacing < 4 and cog_load > 6:
        suggestions.append(
            "Re-edit pacing in acts 2 and 3: Combination of slow pacing and high cognitive load introduces viewer fatigue."
        )
    if pauses > 4:
        suggestions.append(
            "Review scene complexity and clarity: Frequent pauses indicate viewers need to re-read subtitles or decipher dense plot points."
        )
    if ep_num == 1 and drop_off_prob > 0.4:
        suggestions.append(
            "Season Premiere Alert: Critical entry point. Consider test screening and tightening intro sequence."
        )

    if not suggestions:
        suggestions.append("Content pacing and hook metrics align well with benchmark completion standards.")

    return {
        "title": title,
        "episode_number": ep_num,
        "drop_off_probability": round(drop_off_prob, 4),
        "editorial_suggestions": suggestions,
        "disclaimer": (
            "Content recommendations represent editorial guidelines derived from statistical associations, "
            "intended for creator review rather than automated video alterations."
        ),
    }
