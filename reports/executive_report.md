# Customer Churn — Executive Briefing
*Generated 26 Sep 2026 from 10,000 real bank customers (public Bank Customer Churn dataset).*

## 1. Headline
- **20.4%** of customers churned.
- A LightGBM model ranks customers by churn risk with a test AUC of **0.864** (5-fold CV 0.863 ± 0.011), versus 0.785 for a Logistic Regression baseline.
- Contacting only the riskiest **10%** of customers reaches **41%** of all churners — **4.1x** better than picking customers at random.
- Under the stated campaign assumptions, targeting by model score is worth **₹6.08 lakh** per 2,000 customers, versus ₹2.21 lakh for contacting everyone.

## 2. Who churns
| Segment | Churn rate | Compared with |
|---|---|---|
| Age 50+ | 45.4% | 16.3% for under 50 (2.8x) |
| Customers in Germany | 32.4% | 16.3% for other countries (2.0x) |
| Inactive members | 26.9% | 14.3% for active members (1.9x) |
| Female customers | 25.1% | 16.5% for male customers (1.5x) |

**Products held is the sharpest signal:** 1 product → 28% churn (n=5,084), 2 products → 8% churn (n=4,590), 3 products → 83% churn (n=266), 4 products → 100% churn (n=60). Customers with 3–4 products are a small group but churn at very high rates, which suggests they were cross-sold products they did not want.

The model's most important features are `age`, `num_products`, `balance`, `credit_score`, `balance_to_salary`.

## 3. Does the risk score hold up?
Actual churn in the held-out test set, by predicted risk segment:

| Predicted segment | Customers | Actual churn |
|---|---|---|
| High Risk | 462 | 60.0% |
| Medium Risk | 459 | 16.1% |
| Low Risk | 1,079 | 5.2% |

## 4. Recommended campaign
- Contact customers with a churn risk score ≥ **0.38** (threshold chosen on training data, results below are on the test set): 751 of 2,000 test customers, reaching 328 actual churners.
- Prioritise **inactive members aged 50+** (82% churn, n=484) and **customers with 3+ products** (86% churn, n=326) for relationship-manager calls.
- Investigate the **Germany** portfolio, which churns at 32.4% against 20.4% overall.
- Run the campaign as an A/B test (hold out a random control group) so the true save rate can be measured and fed back into the threshold.

## 5. Assumptions and limits
- Campaign economics assume ₹500 per contact, ₹10,000 per retained churner and a 30% save rate. These are placeholders, not figures from the data; change them in `churn_model.py`.
- Risk scores rank customers well but are not calibrated probabilities: class weighting lifts the average test score to 0.35 against an actual churn rate of 0.20.
- The dataset is a single snapshot, so the model predicts who churned, not when. A production model would need monthly behavioural history.
- Gender and age are legally sensitive in credit and marketing decisions; they are used here for analysis, and a deployed model would need a fairness review.