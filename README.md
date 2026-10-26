# Audience Retention & Churn Predictor for OTT Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

> **Third-Year Computer Science / Data Science Capstone Project**  
> An end-to-end, reproducible, and academically defensible Machine Learning system addressing subscriber churn and episodic content retention on streaming platforms.

---

## 📌 Executive Summary & Key Highlights

This project implements a complete, industry-standard machine learning system combining:
1. **Macro Subscriber Churn Prediction:** Classifying customer churn risk across 5,000 subscriber accounts.
2. **Micro Episode Retention Modeling:** Forecasting episodic drop-off across 33,171 episodes while preventing synthetic target leakage.
3. **Unsupervised Audience Segmentation:** K-Means clustering ($k=4$) with PCA 2D projections profiling behavioral subscriber cohorts.
4. **Transparent Retention Strategy Engine:** Rule-based decision layer calculating revenue at risk and suggesting testable interventions.
5. **Interactive Streamlit Web Dashboard:** A dark-themed, 5-page OTT-inspired operational cockpit connected live to trained models.

---

## ⚠️ Important Dataset Limitations Disclosure

> **SYNTHETIC BEHAVIORAL DATASET NOTICE:**  
> The datasets utilized (`netflix_customer_churn.csv` and `ott_episode_retention.csv`) contain synthetic behavioral telemetry generated to simulate streaming analytics patterns. They do **not** represent authentic corporate data from Netflix Inc. or any commercial streaming service. All retention interventions and campaign ROI projections represent academic hypotheses for controlled A/B testing rather than measured commercial outcomes.

---

## 🏗️ Project Architecture & Structure

```
DS CP/
├── data/
│   ├── netflix_customer_churn.csv       # 5,000 subscriber records
│   └── ott_episode_retention.csv        # 33,171 episode-level records
├── notebooks/
│   └── ott_retention_churn_analysis.ipynb # Executable Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Data ingestion, path resolution & schema validation
│   ├── preprocessing.py                 # Scikit-learn ColumnTransformers (zero leakage)
│   ├── feature_engineering.py           # Engagement score, friction index & domain indicators
│   ├── eda.py                           # SciPy hypothesis tests & figure generation
│   ├── metrics.py                       # Classification metrics, ROC/PR curves, t-tests
│   ├── churn_model.py                   # Stratified CV & hyperparameter tuning for churn
│   ├── retention_model.py               # Full vs Early-Warning episode drop-off pipelines
│   ├── segmentation.py                  # K-Means clustering & PCA 2D projections
│   ├── explainability.py                # Permutation importance & local explanations
│   └── recommendations.py               # Rule-based actions & revenue-at-risk calculations
├── models/
│   ├── churn_pipeline.joblib            # Champion Gradient Boosting churn model
│   ├── churn_metrics.json               # Full benchmark metrics across all models
│   ├── retention_pipeline.joblib        # Full behavioral episode drop-off pipeline
│   ├── retention_early_pipeline.joblib  # Early-Warning content-only drop-off pipeline
│   ├── retention_metrics.json           # Episode retention benchmarks
│   ├── segmentation_kmeans.joblib       # Fitted K-Means, Scaler & PCA
│   └── segmentation_metrics.json        # Cluster profiles & silhouette diagnostics
├── reports/
│   ├── figures/                         # 11 publication-grade PNG charts
│   ├── data_dictionary.md               # Complete attribute and feature specifications
│   ├── project_report.md                # Full academic capstone project report
│   └── statistical_tests.json           # SciPy Welch's t-test and Chi-square outputs
├── tests/
│   ├── test_data_loader.py              # Schema adherence & invalid input tests
│   ├── test_preprocessing.py            # Pipeline transformer & scaling tests
│   ├── test_feature_engineering.py      # Score boundedness & formula monotonicity tests
│   ├── test_models.py                   # Model serialization & probability bounds tests
│   └── test_recommendations.py          # Rule activation & revenue-at-risk tests
├── app.py                               # 5-Page Interactive Streamlit Web Application
├── train.py                             # Master training & pipeline execution script
├── requirements.txt                     # Pinned project dependencies
├── README.md                            # Comprehensive documentation & Viva Q&A
└── .gitignore                           # Ignored caches, environments, checkpoints
```

---

## ⚡ Quick Start & Execution Commands

### 1. Environment Setup
Clone or navigate to the project directory and install the requirements:
```bash
# Verify Python version (Python 3.10+ recommended)
python --version

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Train Models & Generate Artifacts
Run the master training pipeline to validate datasets, generate EDA charts, run SciPy statistical hypothesis tests, train all models with cross-validation, and export figures:
```bash
python train.py
```

### 3. Run Unit Test Suite
Execute the automated pytest suite (15 unit tests covering schemas, leakage prevention, bounds, and recommendations):
```bash
python -m pytest tests -v
```

### 4. Launch the Interactive Dashboard
Start the production Streamlit application:
```bash
streamlit run app.py
```
*The app will automatically launch at `http://localhost:8501`.*

### 5. Run the Jupyter Notebook
Open and execute the step-by-step academic analysis notebook:
```bash
jupyter notebook notebooks/ott_retention_churn_analysis.ipynb
```

---

## 📊 Measured Model Performance Results

*All metrics reported below were computed directly from held-out test sets without data leakage.*

### 1. Subscriber Churn Models (Held-out Test Set, N=1,000)

| Model Architecture | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline (Most Frequent)** | 0.5000 | 50.3% | 0.5030 | 1.0000 | 0.6693 |
| **Logistic Regression (L2)** | 0.9684 | 89.9% | 0.9002 | 0.9024 | 0.9013 |
| **Decision Tree Classifier** | 0.9917 | 97.6% | 0.9760 | 0.9761 | 0.9758 |
| **Random Forest (100 Trees)** | 0.9986 | 98.5% | 0.9841 | 0.9861 | 0.9850 |
| **Gradient Boosting (Champion)** | **0.9990** | **99.2%** | **0.9921** | **0.9920** | **0.9920** |

### 2. Episode Retention & Drop-Off Models (Test Set, N=6,635)

| Evaluation Framework | Champion Model | Test ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Behavioral Model** (Includes watch %, pauses, rewinds) | Gradient Boosting | **0.9999** | **99.70%** | 0.9948 | 0.9844 | **0.9896** |
| **Early-Warning Model** (Pre-completion content features only) | Gradient Boosting | **0.9839** | **94.86%** | 0.8880 | 0.7586 | **0.8183** |

> **Critical Academic Finding on Target Leakage:**  
> In synthetic streaming datasets, `avg_watch_percentage` strongly correlates with `drop_off` (mean watch of 37% for drop-offs vs 60% for completed). While the Full Model achieves 99.7% accuracy, in commercial reality post-completion telemetry is unavailable during early playback. The **Early-Warning Model** proves that content attributes alone (pacing score, hook strength, cognitive load, dialogue density, and season progress) predict drop-off with **94.9% accuracy** and **0.984 ROC-AUC**.

---

## 🔬 Statistical Hypothesis Testing (SciPy)

- **Watch Hours vs Churn:** Welch's t-test ($t = -38.64, p < 10^{-200}$). Retained subscribers average 25.1 watch hours vs 15.1 hours for churners.
- **Login Inactivity vs Churn:** Welch's t-test ($t = 37.74, p < 10^{-195}$). Churned subscribers average 19.1 days of inactivity vs 10.9 days for retained subscribers.
- **Subscription Tier vs Churn:** Pearson's Chi-Square ($\chi^2 = 143.91, p = 5.6 \times 10^{-32}$, Cramér's V = 0.170). Basic plan subscribers exhibit a 61.8% churn rate compared to 43.7% for Premium subscribers.
- **Payment Method vs Churn:** Pearson's Chi-Square ($\chi^2 = 97.43, p = 3.4 \times 10^{-20}$, Cramér's V = 0.140). Non-recurring manual methods (Crypto: 59.7%, Gift Cards: 57.8%) churn at significantly higher rates than auto-pay Credit Cards (43.6%).
- **Gender vs Churn:** Pearson's Chi-Square ($\chi^2 = 0.635, p = 0.728$). Confirms no statistically significant demographic bias in churn propensity.


## 🖥️ Live Demonstration 
When presenting this project:

1. **Step 1: Open the Terminal & Run the Test Suite**
   - Execute `python -m pytest tests -v`.
   - Show the evaluator that all 15 tests pass in ~3 seconds, verifying dataset schema integrity, pipeline loading, and mathematical boundedness.
2. **Step 2: Walk Through the Executive Overview in Streamlit**
   - Launch `streamlit run app.py`.
   - Point out the macro KPIs (5,000 active base, 50.3% observed churn, 20.1 avg watch hours).
   - Show the SciPy hypothesis testing tables proving that watch hours ($t = -38.64$) and subscription tier ($\chi^2 = 143.91$) are statistically significant drivers of churn.
3. **Step 3: Test the Live Churn Prediction Cockpit**
   - Navigate to **"2. Churn Prediction"**.
   - Input a low-engagement profile: `Watch Hours = 4.0`, `Last Login = 25 days`, `Payment = Gift Card`, `Profiles = 1`.
   - Click **"Evaluate Churn Risk"**.
   - Show the model outputting **High Risk (~85-90% probability)**, the waterfall attribution chart showing login recency pushing up risk, and the automated rule suggesting an **Inactivity Re-engagement Campaign** and **Auto-Pay Enrollment incentive**.
4. **Step 4: Demonstrate the Early-Warning Retention Model**
   - Navigate to **"3. Audience Retention"**.
   - Toggle between **"Full Model"** and **"Early-Warning Model"**.
   - Explain to the professor how the Early-Warning Model prevents post-completion data leakage by forecasting drop-off with 94.9% accuracy based solely on episode pacing, cognitive load, and opening hook strength.
