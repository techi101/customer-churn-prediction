# 📊 Customer Churn Prediction Model — Banking Use Case

> **Predictive Analytics | Logistic Regression + Random Forest | AUC 0.87**

A complete, end-to-end machine learning project that predicts the likelihood of
customer churn in a retail banking portfolio — from raw data generation to an
interactive analytics dashboard and an executive-level narrative report.

---

## 🎯 Project Highlights

| Metric | Value |
|---|---|
| **Dataset Size** | 10,500 synthetic banking records |
| **Best Model AUC** | **0.87** (Random Forest) |
| **Baseline AUC** | 0.83 (Logistic Regression) |
| **Key Features Engineered** | Spend Trend, Complaint Frequency, Reward Redemption Rate |
| **Dashboard** | Interactive Streamlit app with risk segmentation |
| **Report** | Automated executive narrative (GenAI-style) |

---

## 🗂️ Project Structure

```
customer-churn/
│
├── data_generator.py        # Synthetic dataset creation (10,500 records)
├── churn_model.py           # Full ML pipeline: EDA → preprocessing → training → evaluation
├── dashboard.py             # Interactive Streamlit dashboard
├── report_generator.py      # Automated executive report generator
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── banking_customers.csv    # Generated raw dataset
│   └── scored_customers.csv     # Test set with churn probabilities
│
├── models/
│   ├── logistic_regression.pkl  # Trained LR pipeline
│   └── random_forest.pkl        # Trained RF pipeline
│
└── reports/
    ├── eda_plots.png            # EDA visualisations
    ├── eda_summary.txt          # EDA statistics
    ├── model_evaluation.png     # ROC curves, confusion matrix, CV comparison
    ├── feature_importance.png   # Random Forest feature importances
    ├── classification_reports.txt
    ├── executive_report.txt     # Plain-text executive briefing
    └── executive_report.md      # Markdown executive briefing
```

---

## 🔧 Features Engineered

These domain-driven features mirror real banking analytics practice:

| Feature | Description | Churn Signal |
|---|---|---|
| `spend_trend` | Rate of change in monthly spend | Negative → churn risk ⬆️ |
| `complaint_frequency` | Avg complaints per year | High → churn risk ⬆️ |
| `reward_redemption_rate` | % of rewards redeemed | Low → disengaged → churn ⬆️ |
| `monthly_transactions` | Avg monthly transactions (last 6 mo) | Low → inactive → churn ⬆️ |
| `inactive_months` | Consecutive months with no activity | High → churn ⬆️ |
| `credit_score` | Customer creditworthiness | Low → higher risk |
| `balance` | Account balance | Low → lower loyalty |
| `tenure_years` | Years with the bank | High → more loyal |

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
```bash
python data_generator.py
```
Creates `data/banking_customers.csv` with 10,500 records.

### 3. Train the Models & Run the Pipeline
```bash
python churn_model.py
```
Outputs:
- Trained model artefacts in `models/`
- EDA plots, ROC curves, feature importance charts in `reports/`
- Scored test set: `data/scored_customers.csv`

### 4. Generate the Executive Report
```bash
python report_generator.py
```
Creates narrative business report in `reports/executive_report.md`.

### 5. Launch the Dashboard
```bash
streamlit run dashboard.py
```
Opens an interactive analytics dashboard at `http://localhost:8501`.

---

## 📈 Model Performance

| Model | Test AUC | 5-Fold CV AUC |
|---|---|---|
| **Random Forest** | **0.87** | **0.87 ± 0.01** |
| Logistic Regression | 0.83 | 0.83 ± 0.01 |

### Why Random Forest outperforms Logistic Regression here:
- Captures non-linear interactions between `spend_trend`, `complaint_frequency`, and `inactive_months`
- Handles correlated features (balance × num_products) more robustly
- `class_weight="balanced"` corrects for 80/20 class imbalance without synthetic oversampling

---

## 🏦 Business Application

This model enables a bank's analytics team to:

1. **Score** all customers monthly with churn probability
2. **Segment** the portfolio into High / Medium / Low Risk tiers
3. **Target** high-risk customers with proactive retention campaigns
4. **Track** the impact of interventions through a KPI dashboard
5. **Report** findings to leadership via auto-generated executive briefings

---

## 🤖 GenAI Integration

`report_generator.py` demonstrates a **GenAI-augmented analytics workflow**:
- Takes structured model outputs (AUC scores, segment stats, feature importances)
- Synthesises them into a narrative executive report
- Mimics the output of a large language model prompted with business context
- Illustrates **prompt design** and **responsible AI** principles:
  always grounded in verifiable data, never hallucinated statistics

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `scikit-learn` | ML models, preprocessing, evaluation |
| `pandas / numpy` | Data manipulation |
| `matplotlib / seaborn` | Static visualisations |
| `streamlit` | Interactive dashboard |
| `plotly` | Interactive Plotly charts |
| `joblib` | Model serialisation |

---

## 👨‍💻 Author

**Nitesh** · B.Tech ECE (AI & ML) · NSUT Delhi  
Built as a portfolio project targeting Decision Analytics roles in management consulting.

---

*All data is synthetically generated. No real customer information is used.*
