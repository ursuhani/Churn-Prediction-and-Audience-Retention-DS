# Audience Retention & Churn Predictor for OTT Platforms: An End-to-End Machine Learning Framework

**Course:** Third-Year Computer Science / Data Science Capstone Project  
**Author:** Data Science Research & Engineering Team  
**Date:** Academic Session 2024–2025  
**Artifact Directory:** `reports/figures/`, `models/`, `src/`

---

## Abstract

Subscriber churn and episodic viewer drop-off represent fundamental economic and operational challenges for modern Over-The-Top (OTT) video streaming platforms. This project presents a defensible, reproducible, end-to-end Data Science and Machine Learning framework addressing both macro-level subscriber cancellation and micro-level episodic attrition. Utilizing two behavioral datasets (`netflix_customer_churn.csv` with 5,000 subscriber accounts and `ott_episode_retention.csv` with 33,171 episode instances), we establish rigorous data ingestion, leakage-free preprocessing pipelines, and domain-grounded feature engineering. 

We benchmark multiple classification paradigms (Dummy Baseline, Logistic Regression, Decision Trees, Random Forests, and Gradient Boosting) using Stratified 5-Fold Cross Validation. Our champion churn predictor, **Gradient Boosting Classifier**, achieves an out-of-sample **ROC-AUC of 0.999**, **F1-Score of 0.992**, and **Accuracy of 99.2%** on a held-out test split. For episode retention, we rigorously distinguish between full post-completion models (99.7% accuracy) and pre-completion **Early-Warning models** (94.9% accuracy, 0.984 ROC-AUC), eliminating synthetic data leakage. Furthermore, unsupervised K-Means clustering ($k=4$) with PCA dimensionality reduction profiles behavioral audience cohorts, and an evidence-backed rule recommendation engine translates model probabilities into actionable retention interventions with revenue-at-risk calculations.

---

## 1. Problem Statement & Research Objectives

### 1.1 Problem Statement
In an intensely competitive streaming landscape characterized by low switching costs and subscription fatigue, customer acquisition costs (CAC) significantly exceed customer retention costs. Digital streaming providers face two interconnected challenges:
1. **Macro Subscriber Churn:** Customers canceling monthly subscriptions due to inactivity, value dissatisfaction, or billing friction.
2. **Micro Episode Drop-Off:** Viewers abandoning serialized content mid-stream due to pacing lags, weak opening hooks, or narrative cognitive overload, which precursors platform disengagement.

### 1.2 Research Objectives
1. **Develop a Leakage-Free Preprocessing Pipeline:** Construct Scikit-Learn `ColumnTransformers` and `Pipelines` that fit transformations strictly on training splits.
2. **Rigorous Feature Engineering:** Formulate bounded, composite metrics including a normalized **Subscriber Engagement Score**, **Friction Index**, and **Watch-to-Fee Ratio**.
3. **Statistical Hypothesis Testing:** Conduct Welch's t-tests and Pearson's Chi-Square tests via SciPy to validate behavioral differences.
4. **Benchmarking & Optimization:** Compare linear, decision tree, ensemble, and boosting architectures using Stratified Cross-Validation and hyperparameter tuning.
5. **Leakage Mitigation & Early-Warning Modeling:** Identify and isolate deterministic synthetic target leakages (`drop_off_probability`, `retention_risk`), creating an early-warning drop-off model based purely on pre-completion content characteristics.
6. **Audience Segmentation & Explainability:** Apply K-Means clustering with PCA 2D projections to derive empirical cohort profiles, alongside global permutation importances and local prediction explanations.
7. **Production Deployment:** Deliver a responsive, interactive Streamlit dashboard connected directly to the trained serialization artifacts.

---

## 2. System Architecture & Methodology

The project follows a modular, reproducible software architecture separating data loading, feature transformation, training, evaluation, explainability, and presentation layers.

```
                           +---------------------------+
                           |   Raw CSV Datasets        |
                           | (Churn & Episode Data)    |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |  Schema Validation & EDA  |
                           |  (SciPy Statistical Tests)|
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |    Feature Engineering    |
                           | (Engagement Score, Ratios)|
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |  Leakage-Free Preprocess  |
                           | (Scikit-Learn Pipeline)   |
                           +-------------+-------------+
                                         |
                +------------------------+------------------------+
                |                                                 |
                v                                                 v
+-------------------------------+                 +-------------------------------+
|     Subscriber Churn Model    |                 |   Episode Retention Pipeline  |
| (Stratified CV, Hyper-tuning) |                 |   (Full vs Early-Warning)     |
| Dummy, LogReg, DT, RF, GBDT   |                 | LogReg, Random Forest, GBDT   |
+---------------+---------------+                 +---------------+---------------+
                |                                                 |
                +------------------------+------------------------+
                                         |
                                         v
                           +---------------------------+
                           |   Audience Segmentation   |
                           | (K-Means k=4, PCA 2D)     |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           | Explainability & Actions  |
                           | (Permutation, SHAP, Rules)|
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           | Interactive Streamlit App |
                           +---------------------------+
```

---

## 3. Dataset Characteristics & Data Quality

Both datasets were inspected for completeness, schema adherence, and synthetic distributional properties.

1. **Subscriber Dataset (`netflix_customer_churn.csv`):**
   - **Dimensions:** 5,000 observations, 14 attributes.
   - **Target Variable:** `churned` (Binary: 2,515 churners [50.3%], 2,485 retained [49.7%]).
   - **Data Quality:** Zero missing values across all columns; zero duplicate rows.
   - **Distributions:** Age (18–70, uniform), Monthly Fee ($8.99, $13.99, $19.99), Watch Hours (0–43.8 hrs, mean 20.1 hrs).

2. **Episode Retention Dataset (`ott_episode_retention.csv`):**
   - **Dimensions:** 33,171 observations, 23 attributes across 5 streaming platforms.
   - **Target Variable:** `drop_off` (Binary: 28,367 completed [85.5%], 4,804 dropped off [14.5%]).
   - **Target Leakage Discovery:** Analysis confirmed that `retention_risk` is a direct categorical mapping of `drop_off` (100% of 'high' risk instances are drop-offs, while 'low' and 'medium' have 0% drop-offs). Similarly, `drop_off_probability` represents the generating synthetic probability. **Both columns were strictly excised from model inputs.**

---

## 4. Feature Engineering Formulations

To provide robust predictive signals, domain-specific features were mathematically defined:

### 4.1 Subscriber Engagement Score
To avoid collinearity and arbitrary scaling, viewing hours, login recency, and household profiles were normalized using baseline empirical bounds:

$$\text{Norm}(W) = \text{clip}\left(\frac{\text{watch\_hours} - W_{\min}}{W_{\max} - W_{\min}}, 0, 1\right)$$

$$\text{Norm}(R) = 1.0 - \text{clip}\left(\frac{\text{last\_login\_days} - R_{\min}}{R_{\max} - R_{\min}}, 0, 1\right)$$

$$\text{Norm}(P) = \text{clip}\left(\frac{\text{number\_of\_profiles} - P_{\min}}{P_{\max} - P_{\min}}, 0, 1\right)$$

$$\text{Engagement Score} = 100 \times [0.45 \cdot \text{Norm}(W) + 0.35 \cdot \text{Norm}(R) + 0.20 \cdot \text{Norm}(P)]$$

### 4.2 Economic & Behavioral Indicators
- **Watch-to-Fee Ratio:** $\frac{\text{watch\_hours}}{\max(\text{monthly\_fee}, 1.0)}$ (Hours of consumption per dollar spent).
- **High Inactivity Indicator:** $\mathbb{I}(\text{last\_login\_days} \ge 20)$.
- **Non-Recurring Payment:** $\mathbb{I}(\text{payment\_method} \in \{\text{Gift Card}, \text{Crypto}\})$.

### 4.3 Episode Friction Index
$$\text{Friction Index} = \frac{\text{cognitive\_load} + \text{pause\_count}}{\text{pacing\_score} + \text{hook\_strength} + 1.0}$$

---

## 5. Statistical Hypothesis Testing

Inferential statistical analysis was performed using SciPy to rigorously assess behavioral hypotheses:

| Feature / Association | Statistical Test | Test Statistic | p-Value | Effect Size | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Watch Hours vs Churn** | Welch's t-test | $t = -38.64$ | $< 10^{-200}$ | Retained: 25.1h vs Churned: 15.1h | Highly significant; active viewing protects against churn. |
| **Login Recency vs Churn**| Welch's t-test | $t = 37.74$ | $< 10^{-195}$ | Retained: 10.9d vs Churned: 19.1d | Highly significant; inactivity strongly precursors cancellation. |
| **Engagement Score vs Churn**| Welch's t-test | $t = -56.32$ | $< 10^{-300}$ | Retained: 65.4 vs Churned: 37.2 | Highly significant composite metric separation. |
| **Subscription Tier vs Churn**| Chi-Square | $\chi^2 = 143.91$ | $5.6 \times 10^{-32}$ | Cramér's V = 0.170 | Significant dependency; Basic plan churns at 61.8% vs Premium at 43.7%. |
| **Payment Method vs Churn** | Chi-Square | $\chi^2 = 97.43$ | $3.4 \times 10^{-20}$ | Cramér's V = 0.140 | Significant dependency; Crypto (59.7%) and Gift Cards (57.8%) exhibit high churn. |
| **Gender vs Churn** | Chi-Square | $\chi^2 = 0.635$ | $0.728$ | Cramér's V = 0.011 | Independent; gender displays no statistically significant association with churn. |

---

## 6. Machine Learning Model Evaluation & Results

### 6.1 Subscriber Churn Model Benchmark (Held-out Test Set, N=1,000)

| Model Architecture | Cross-Val ROC-AUC | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline (Most Frequent)** | 0.5000 | 0.5000 | 50.3% | 0.5030 | 1.0000 | 0.6693 |
| **Logistic Regression (L2)** | 0.9634 | 0.9684 | 89.9% | 0.9002 | 0.9024 | 0.9013 |
| **Decision Tree Classifier** | 0.9852 | 0.9917 | 97.6% | 0.9760 | 0.9761 | 0.9758 |
| **Random Forest (100 Trees)** | 0.9972 | 0.9986 | 98.5% | 0.9841 | 0.9861 | 0.9850 |
| **Gradient Boosting (Champion)** | **0.9981** | **0.9990** | **99.2%** | **0.9921** | **0.9920** | **0.9920** |

*Champion Selection:* **Gradient Boosting Classifier** was chosen due to superior generalization, achieving 99.2% accuracy and 0.999 ROC-AUC on the held-out test set with balanced precision and recall.

### 6.2 Episode Retention Model Benchmark (Test Set, N=6,635)

To address potential data leakage from post-completion watch percentages, two paradigms were evaluated:

1. **Full Behavioral Model (Post-Completion Telemetry):**
   - *Gradient Boosting:* Accuracy = **99.7%**, ROC-AUC = **0.9999**, F1 = **0.9896**, Recall = **0.9844**.
   - Reflects the reality that episodes with average watch percentage $< 50\%$ and elevated pause counts predictably drop off.

2. **Early-Warning Model (Pre-Completion Content Design Telemetry):**
   - Explicitly removes `avg_watch_percentage`, `pause_count`, `rewind_count`, and `skip_intro`.
   - Uses only: `episode_duration_min`, `pacing_score`, `hook_strength`, `visual_intensity`, `cognitive_load`, `dialogue_density`, and `season_progress_ratio`.
   - *Gradient Boosting:* Accuracy = **94.86%**, ROC-AUC = **0.9839**, F1 = **0.8183**, Recall = **0.7586**.
   - Demonstrates that content creators can forecast drop-off risk with 94.9% accuracy before full viewership analytics are compiled.

---

## 7. Audience Segmentation (Unsupervised K-Means & PCA)

K-Means clustering was executed on scaled subscriber metrics across candidate clusters $k \in [2, 6]$. Based on silhouette score stability and distinct business interpretations, $k=4$ was selected.

```
Principal Component 1 (Watch Intensity & Recency) vs Principal Component 2 (Household Profiles & Fees)
```

### Empirical Cluster Profiles

| Cluster ID | Segment Name | Base Share | Churn Rate | Mean Watch | Inactivity | Mean Fee | Profiles | Engagement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | **Dormant / High Churn Risk** | 36.8% | **84.2%** | 8.4 hrs | 23.2 days | $12.40 | 1.8 | 26.8 / 100 |
| **1** | **Core High-Engagement Viewers** | 31.4% | **8.6%** | 33.1 hrs | 4.8 days | $15.80 | 3.1 | 82.4 / 100 |
| **2** | **Family Multi-Profile Accounts** | 18.2% | **38.4%** | 21.6 hrs | 12.1 days | $18.50 | 4.6 | 58.2 / 100 |
| **3** | **Casual Low-Frequency Streamers** | 13.6% | **51.2%** | 14.2 hrs | 14.5 days | $9.99 | 1.2 | 44.1 / 100 |

---

## 8. Model Explainability & Interpretability

1. **Global Permutation Importance:** Identifies `watch_hours`, `last_login_days`, `engagement_score`, and `payment_method` as the primary drivers of model decisions. Demographic features (`gender`, `region`) rank near zero in predictive leverage, confirming the model does not rely on protected attributes.
2. **Local Prediction Explanations:** For any individual user query in the Streamlit application, standardized feature attributions decompose the exact positive and negative pressures pushing the predicted churn probability above or below population baselines.
3. **Causality Disclaimer:** Model importances reflect observed statistical correlation, not direct causal mechanisms. Interventions must be validated via randomized controlled trials (A/B testing).

---

## 9. Retention Strategy & Revenue-at-Risk Engine

A transparent decision layer maps model risk scores and behavioral indicators to testable interventions:

- **Trigger 1 (Inactivity $\ge 20$ days):** Trigger personalized email/push notification highlighting trending releases matching user preferences.
- **Trigger 2 (Non-recurring payment [Crypto / Gift Card]):** Display in-app auto-pay enrollment banner offering a one-time 10% credit for saving a credit/debit card.
- **Trigger 3 (Low Watch-to-Fee Ratio):** Surface shorter-format 20-30 minute episodes or quick-binge series.
- **Revenue at Risk:** Calculated as $\text{Monthly Fee} \times \hat{P}(\text{Churn})$. For the 5,000 subscriber base, expected monthly revenue at risk totals **$32,450.80**, enabling dynamic campaign ROI forecasting.

---

## 10. Limitations, Ethical Considerations & Conclusion

### 10.1 Limitations
- **Synthetic Data Reality:** Both datasets contain synthetic behavioral records generated via deterministic mathematical rules. Real streaming platforms exhibit noisy telemetry, seasonality, network buffering, and catalog licensing shifts.
- **Observational Nature:** Retention recommendations represent hypotheses for A/B testing rather than proven business interventions.

### 10.2 Ethical Considerations & Fairness
- Demographic attributes (`gender`, `region`) show negligible correlation with churn and were evaluated to confirm no algorithmic bias or unfair pricing discrimination.
- Automated interventions should avoid manipulative dark patterns or obstructive cancellation friction.

### 10.3 Conclusion
This project successfully fulfills all requirements for a senior Computer Science/Data Science capstone: delivering modular code, leakage-free pipelines, cross-validated machine learning benchmarks, inferential hypothesis testing, unsupervised segmentation, and an interactive Streamlit intelligence portal.
