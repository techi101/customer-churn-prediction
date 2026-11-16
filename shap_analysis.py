# shap_analysis.py: explain the saved LightGBM churn model with SHAP.
# SHAP (SHapley Additive exPlanations) splits each customer's churn score into a push from every feature,
# so we can say WHY a customer scored high, not just which features the trees used most (gain importance).
# Same data, same feature engineering, same split (SEED 42) as churn_model.py; it reuses models/lightgbm.pkl.
# WHAT THIS FILE IS: explains the 2,000 test customers' LightGBM scores (test AUC 0.8642) feature by feature.
# Example: num_products has the biggest mean |SHAP| (0.7931), just above age (0.7886), though age leads gain (0.2917).
# Flow: read CSV -> same 5 features -> same 80/20 split -> load model -> SHAP values
#   -> reports/shap_importance.csv + reports/shap_summary.png -> print direction check
import joblib, pandas as pd, numpy as np, matplotlib  # load .pkl, tables, maths, plotting
matplotlib.use("Agg")  # draw to files, no window (before pyplot)
import matplotlib.pyplot as plt, shap  # drawing interface + SHAP explainer library
from sklearn.model_selection import train_test_split  # same 80/20 split as churn_model.py

# ── BLOCK: rebuild the exact test set of churn_model.py ──────────────────────
# Must match churn_model.py (SEED 42, 0.20, stratify) or rows differ from the model's test set.
SEED = 42  # same SEED as churn_model.py so the test split matches
raw = pd.read_csv("data/Churn_Modelling.csv")  # 10,000 customers
df = raw.rename(columns={  # same snake_case names as churn_model.py
    "CustomerId": "customer_id", "CreditScore": "credit_score", "Geography": "geography",  # ID, score, country
    "Gender": "gender", "Age": "age", "Tenure": "tenure_years", "Balance": "balance",  # person + account
    "NumOfProducts": "num_products", "HasCrCard": "has_credit_card",  # products, card 0/1
    "IsActiveMember": "is_active_member", "EstimatedSalary": "estimated_salary",  # active 0/1, salary
    "Exited": "churn"}).drop(columns=["RowNumber", "Surname"])  # target; drop useless columns

# ── Step 1: same 5 engineered features as add_features() ──
d = df  # same table, not a copy: edits change df too
d["zero_balance"] = d["balance"].eq(0).astype(int)  # 1 if balance is 0
d["balance_to_salary"] = d["balance"] / d["estimated_salary"].clip(lower=1)  # clip = never divide by ~0
d["products_per_tenure"] = d["num_products"] / (d["tenure_years"] + 1)  # +1 because tenure can be 0
d["inactive_multi_prod"] = ((d["is_active_member"] == 0) & (d["num_products"] >= 2)).astype(int)  # inactive, 2+ products
d["senior"] = d["age"].ge(50).astype(int)  # 1 if age 50+

# ── Step 2: same inputs and same 80/20 split ──
X = df.drop(columns=["customer_id", "churn"]); y = df["churn"]  # 15 inputs, 0/1 target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)  # 2,000 test

# ── BLOCK: load model and turn test rows into the model's 16 inputs ──────────
pipe = joblib.load("models/lightgbm.pkl")  # saved by churn_model.py
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]  # preprocessor, LightGBM model
CAT = ["geography", "gender"]  # text columns (one-hot encoded)
num = [c for c in X.columns if c not in CAT]  # 13 number columns, same order as NUM_FEATS
names = num + list(pre.named_transformers_["cat"].get_feature_names_out(CAT))  # + geography_Germany, ... = 16
Xt = pd.DataFrame(pre.transform(X_test), columns=names, index=X_test.index)  # scaled + one-hot test table

# Sanity check: the pipeline and our manual path give identical scores.
assert np.allclose(pipe.predict_proba(X_test)[:, 1], clf.predict_proba(Xt.values)[:, 1])  # stop if they differ

# ── BLOCK: SHAP values (one push per customer per feature) ───────────────────
explainer = shap.TreeExplainer(clf)  # fast exact SHAP for tree models
sv = explainer.shap_values(Xt.values)  # 2,000 x 16 array of pushes
sv = sv[1] if isinstance(sv, list) else sv          # older shap returns one array per class

# ── BLOCK: SHAP importance vs gain importance -> CSV ─────────────────────────
# avg push size, num_products 0.7931 first
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
gain = pd.Series(clf.booster_.feature_importance("gain"), index=names); gain = gain / gain.sum()  # gain as shares
out = pd.DataFrame({"mean_abs_shap": mean_abs, "gain_share": gain.reindex(mean_abs.index)})  # both side by side
out.to_csv("reports/shap_importance.csv", float_format="%.4f")  # 4 decimals

# ── BLOCK: summary chart -> reports/shap_summary.png ─────────────────────────
shap.summary_plot(sv, Xt, show=False, max_display=12)  # dot per customer, top 12 features
plt.title("What pushes churn risk up or down (SHAP, test set)")  # chart title
plt.tight_layout(); plt.savefig("reports/shap_summary.png", dpi=150); plt.close()  # fix spacing, save, free memory

# Direction check on raw values for the top features (positive = higher value raises churn risk)
# Raw X_test values used when the column exists there; one-hot columns come from Xt.
dirs = {f: np.corrcoef(X_test[f] if f in X_test else Xt[f], sv[:, names.index(f)])[0, 1] for f in mean_abs.index[:6]}
print(out.head(10).to_string())  # top 10 features table
# > 0: higher value, higher risk
print("direction (corr of feature value with its SHAP push):", {k: round(v, 2) for k, v in dirs.items()})
