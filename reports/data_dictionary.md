# Data Dictionary & Schema Specification

**Project:** Audience Retention & Churn Predictor for OTT Platform  
**Target Course:** Third-Year B.Tech / B.E. Computer Science & Engineering (Data Science)  
**Dataset Provenance:** Synthetic behavioral analytics datasets representing OTT streaming subscriptions and episode engagement telemetry.

---

## 1. Subscriber Churn Dataset (`netflix_customer_churn.csv`)

- **Total Records:** 5,000 subscribers  
- **Raw Features:** 13 input features + 1 target (`churned`)  
- **Missing Values:** 0 across all columns  
- **Target Distribution:** 2,515 Churned (50.3%), 2,485 Retained (49.7%)

### Raw Attributes

| Column Name | Data Type | Value Range / Categories | Description | Measurement Unit |
| :--- | :--- | :--- | :--- | :--- |
| `customer_id` | String (UUID) | Unique 36-char string | Unique subscriber surrogate key identifier. | Categorical identifier |
| `age` | Integer | 18 – 70 | Subscriber age in completed years. | Years |
| `gender` | Categorical | `Male`, `Female`, `Other` | Self-reported subscriber gender. | Discrete categories |
| `subscription_type` | Categorical | `Basic`, `Standard`, `Premium` | Active subscription plan tier. | Discrete tiers |
| `watch_hours` | Float | 0.0 – 43.8 | Total viewing duration accumulated in the past 30 days. | Hours / month |
| `last_login_days` | Integer | 0 – 30 | Inactivity interval since last platform authentication. | Days |
| `region` | Categorical | `Africa`, `Asia`, `Europe`, `North America`, `Oceania`, `South America` | Primary billing geographic territory. | Regional territory |
| `device` | Categorical | `Desktop`, `Laptop`, `Mobile`, `Tablet`, `TV` | Most frequently utilized playback client device. | Hardware category |
| `monthly_fee` | Float | 8.99 – 19.99 | Recurring subscription billing fee. | USD ($) |
| `churned` | Binary Integer | `0` (Retained), `1` (Churned) | Primary binary prediction target. | Binary indicator |
| `payment_method` | Categorical | `Credit Card`, `Debit Card`, `PayPal`, `Gift Card`, `Crypto` | Payment instrument on billing file. | Instrument type |
| `number_of_profiles`| Integer | 1 – 5 | Active household user sub-profiles on account. | Count |
| `avg_watch_time_per_day` | Float | 0.0 – 1.46 | Calculated daily average viewing volume. | Hours / day |
| `favorite_genre` | Categorical | `Action`, `Comedy`, `Documentary`, `Drama`, `Horror`, `Romance`, `Sci-Fi` | Most watched genre tag. | Genre category |

### Engineered Features

| Feature Name | Derived From | Formula / Construction | Academic Rationale |
| :--- | :--- | :--- | :--- |
| `engagement_score` | `watch_hours`, `last_login_days`, `number_of_profiles` | $100 \times [0.45 \cdot \text{Norm}(W) + 0.35 \cdot (1 - \text{Norm}(R)) + 0.20 \cdot \text{Norm}(P)]$ | Composite metric evaluating platform consumption intensity and multi-user household stickiness. Scaled on $[0, 100]$. |
| `is_high_inactivity` | `last_login_days` | $\mathbb{I}(\text{last\_login\_days} \ge 20)$ | Binary marker flagging disengagement threshold (20+ days without session). |
| `watch_to_fee_ratio` | `watch_hours`, `monthly_fee` | $\frac{\text{watch\_hours}}{\max(\text{monthly\_fee}, 1.0)}$ | Economic value realization: hours of content enjoyed per dollar paid. |
| `is_single_profile` | `number_of_profiles` | $\mathbb{I}(\text{number\_of\_profiles} == 1)$ | Household dependency indicator; single accounts lack multi-user retention barriers. |
| `non_recurring_payment`| `payment_method` | $\mathbb{I}(\text{payment\_method} \in \{\text{Gift Card}, \text{Crypto}\})$ | Friction on renewal; non-recurring instruments require active subscriber payment every cycle. |
| `watch_hours_per_profile` | `watch_hours`, `number_of_profiles` | $\frac{\text{watch\_hours}}{\text{number\_of\_profiles}}$ | Individual viewing consumption normalized by household headcount. |

---

## 2. Episode Retention & Drop-Off Dataset (`ott_episode_retention.csv`)

- **Total Records:** 33,171 episode records  
- **Raw Features:** 22 input columns + 1 target (`drop_off`)  
- **Target Distribution:** 28,367 Completed (85.5%), 4,804 Drop-Off (14.5%)  
- **Target Leakage Warning:** Columns `drop_off_probability` and `retention_risk` are strictly excluded from all predictive pipelines because they represent deterministic target derivations.

### Raw Attributes

| Column Name | Data Type | Value Range / Categories | Description | Measurement Unit |
| :--- | :--- | :--- | :--- | :--- |
| `show_id` | Integer | Unique Show ID | Numerical identifier for TV series. | Identifier |
| `title` | String | E.g. Stranger Things, The Witcher | Series title. | Metadata |
| `platform` | Categorical | `Netflix`, `Amazon Prime`, `Disney+`, `Apple TV+`, `HBO Max` | Streaming platform service. | Streaming network |
| `genre` | Categorical | 16 discrete genres | Primary thematic classification. | Genre tag |
| `release_year` | Float | 2010 – 2024 | Original release year. | Year |
| `season_number` | Integer | 1 – 10 | Chronological season number. | Season index |
| `episode_number` | Integer | 1 – 24 | Episode sequence within season. | Episode index |
| `episode_duration_min`| Integer | 20 – 90 | Runtime duration of the episode. | Minutes |
| `pacing_score` | Integer | 1 – 10 | Narrative progression speed rating. | Scale (1-10) |
| `hook_strength` | Integer | 1 – 10 | First 5-minute narrative engagement score. | Scale (1-10) |
| `dialogue_density` | Categorical | `low`, `medium`, `high` | Frequency and volume of spoken dialogue. | Discrete level |
| `visual_intensity` | Integer | 1 – 10 | Scene complexity, CGI, and cuts tempo. | Scale (1-10) |
| `avg_watch_percentage`| Integer | 13 – 100 | Percentage of episode watched by audience. | Percentage (%) |
| `pause_count` | Integer | 0 – 10 | Number of user pause actions recorded. | Count |
| `rewind_count` | Integer | 0 – 6 | Number of backward scrubber seeks. | Count |
| `skip_intro` | Binary | `0` (Watched), `1` (Skipped) | Title intro sequence skipped. | Binary indicator |
| `cognitive_load` | Integer | 1 – 10 | Narrative complexity and comprehension difficulty. | Scale (1-10) |
| `attention_required` | Categorical | `low`, `medium`, `high` | Level of user visual/audio focus required. | Discrete level |
| `night_watch_safe` | Binary | `0` (No), `1` (Yes) | Absence of jarring audio spikes or sudden flash effects. | Binary indicator |
| `drop_off` | Binary | `0` (Retained), `1` (Dropped off) | Primary target: whether viewer abandoned prior to completion. | Binary indicator |
| `drop_off_probability`| Float | 0.05 – 0.95 | **SYNTHETIC LEAKAGE:** Underlying probability used in synthetic generation. **Excluded from models.** | Probability |
| `retention_risk` | Categorical | `low`, `medium`, `high` | **SYNTHETIC LEAKAGE:** Categorical mapping of drop-off. **Excluded from models.** | Risk tier |
| `dataset_version` | String | `v1.0` | Release tracking version. | Metadata |

### Engineered Retention Features

| Feature Name | Derived From | Formula / Construction | Academic Rationale |
| :--- | :--- | :--- | :--- |
| `season_progress_ratio`| `episode_number`, show max | $\frac{\text{episode\_number}}{\max(\text{episodes in season})}$ | Measures distance through season to capture progression fatigue. |
| `is_season_premiere` | `episode_number` | $\mathbb{I}(\text{episode\_number} == 1)$ | Flag for critical onboarding entry point of a season. |
| `friction_index` | `cognitive_load`, `pause_count`, `pacing_score`, `hook_strength` | $\frac{\text{cognitive\_load} + \text{pause\_count}}{\text{pacing\_score} + \text{hook\_strength} + 1.0}$ | Friction ratio quantifying cognitive impedance relative to narrative pull. |
| `cognitive_to_pacing_ratio` | `cognitive_load`, `pacing_score` | $\frac{\text{cognitive\_load}}{\text{pacing\_score} + 1.0}$ | Relative burden of mental effort vs narrative tempo. |
