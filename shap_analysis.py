# shap_analysis.py: explain the saved LightGBM churn model with SHAP.
# SHAP (SHapley Additive exPlanations) splits each customer's churn score into a push from every feature,
# so we can say WHY a customer scored high, not just which features the trees used most (gain importance).
# Same data, same feature engineering, same split (SEED 42) as churn_model.py; it reuses models/lightgbm.pkl.
import joblib, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap
from sklearn.model_selection import train_test_split

SEED = 42
raw = pd.read_csv("data/Churn_Modelling.csv")
df = raw.rename(columns={
    "CustomerId": "customer_id", "CreditScore": "credit_score", "Geography": "geography",
    "Gender": "gender", "Age": "age", "Tenure": "tenure_years", "Balance": "balance",
    "NumOfProducts": "num_products", "HasCrCard": "has_credit_card",
    "IsActiveMember": "is_active_member", "EstimatedSalary": "estimated_salary",
    "Exited": "churn"}).drop(columns=["RowNumber", "Surname"])
d = df
d["zero_balance"] = d["balance"].eq(0).astype(int)
d["balance_to_salary"] = d["balance"] / d["estimated_salary"].clip(lower=1)
d["products_per_tenure"] = d["num_products"] / (d["tenure_years"] + 1)
d["inactive_multi_prod"] = ((d["is_active_member"] == 0) & (d["num_products"] >= 2)).astype(int)
d["senior"] = d["age"].ge(50).astype(int)
X = df.drop(columns=["customer_id", "churn"]); y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)

pipe = joblib.load("models/lightgbm.pkl")
pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
CAT = ["geography", "gender"]
num = [c for c in X.columns if c not in CAT]
names = num + list(pre.named_transformers_["cat"].get_feature_names_out(CAT))
Xt = pd.DataFrame(pre.transform(X_test), columns=names, index=X_test.index)

# Sanity check: the pipeline and our manual path give identical scores.
assert np.allclose(pipe.predict_proba(X_test)[:, 1], clf.predict_proba(Xt.values)[:, 1])

explainer = shap.TreeExplainer(clf)
sv = explainer.shap_values(Xt.values)
sv = sv[1] if isinstance(sv, list) else sv          # older shap returns one array per class

mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
gain = pd.Series(clf.booster_.feature_importance("gain"), index=names); gain = gain / gain.sum()
out = pd.DataFrame({"mean_abs_shap": mean_abs, "gain_share": gain.reindex(mean_abs.index)})
out.to_csv("reports/shap_importance.csv", float_format="%.4f")

shap.summary_plot(sv, Xt, show=False, max_display=12)
plt.title("What pushes churn risk up or down (SHAP, test set)")
plt.tight_layout(); plt.savefig("reports/shap_summary.png", dpi=150); plt.close()

# Direction check on raw values for the top features (positive = higher value raises churn risk)
dirs = {f: np.corrcoef(X_test[f] if f in X_test else Xt[f], sv[:, names.index(f)])[0, 1] for f in mean_abs.index[:6]}
print(out.head(10).to_string())
print("direction (corr of feature value with its SHAP push):", {k: round(v, 2) for k, v in dirs.items()})
