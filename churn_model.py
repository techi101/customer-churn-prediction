"""
churn_model.py
--------------
End-to-end churn prediction pipeline on the public Bank Customer Churn dataset
(10,000 real, anonymised customers of a European retail bank).

Steps:
  1. Load & validate data             (data/Churn_Modelling.csv)
  2. Exploratory Data Analysis        -> reports/eda_summary.txt, reports/eda_plots.png
  3. Feature engineering              -> ratio / flag features derived from raw columns
  4. Model training                   -> Logistic Regression, Random Forest, LightGBM
  5. Evaluation                       -> ROC-AUC, PR-AUC, 5-fold CV, confusion matrix
  6. Business evaluation              -> top-decile capture, lift, profit-optimal threshold
  7. Artefact export                  -> models/, reports/metrics.json, data/scored_customers.csv

Run:
    python churn_model.py
"""

import os, json, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend - no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base             import clone
from sklearn.preprocessing    import StandardScaler, OneHotEncoder
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve,
)
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from lightgbm                 import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 42

# -- Paths ----------------------------------------------------------------------
DATA_PATH    = "data/Churn_Modelling.csv"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# -- Business assumptions (used only for the threshold / ROI analysis) -----------
# These are stated assumptions, not facts from the data. Change them to match a
# real bank's economics and re-run.
CONTACT_COST      = 500      # cost of one retention offer + outreach (INR)
CUSTOMER_VALUE    = 10_000   # value of keeping one customer who would have churned (INR)
SAVE_RATE         = 0.30     # share of contacted true churners the offer actually retains

# -- Palette --------------------------------------------------------------------
PALETTE   = {"churn": "#E63946", "retain": "#2A9D8F"}
BG_COLOR  = "#0F172A"
TEXT_CLR  = "#E2E8F0"
GRID_CLR  = "#1E293B"

def style_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5)
    ax.set_axisbelow(True)
    if title:   ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=10)

# ===============================================================================
# 1.  LOAD DATA
# ===============================================================================
print("\n" + "="*60)
print("  CUSTOMER CHURN PREDICTION - ML PIPELINE")
print("="*60)

raw = pd.read_csv(DATA_PATH)
df = raw.rename(columns={
    "CustomerId": "customer_id", "CreditScore": "credit_score", "Geography": "geography",
    "Gender": "gender", "Age": "age", "Tenure": "tenure_years", "Balance": "balance",
    "NumOfProducts": "num_products", "HasCrCard": "has_credit_card",
    "IsActiveMember": "is_active_member", "EstimatedSalary": "estimated_salary",
    "Exited": "churn",
}).drop(columns=["RowNumber", "Surname"])   # row index and surname carry no signal

assert df["customer_id"].is_unique, "duplicate customers in source data"
assert df.isnull().sum().sum() == 0, "unexpected missing values"
print(f"\nLoaded {len(df):,} real customers with {df.shape[1]} columns.")

# ===============================================================================
# 2.  EXPLORATORY DATA ANALYSIS
# ===============================================================================
print("\nRunning EDA ...")

def churn_by(col):
    return (df.groupby(col)["churn"].agg(["mean", "count"])
              .rename(columns={"mean": "churn_rate", "count": "n"})
              .assign(churn_rate=lambda t: (t["churn_rate"]*100).round(1)))

eda_lines = [
    "=" * 60, "  EDA SUMMARY - Bank Customer Churn Dataset", "=" * 60,
    f"\nTotal Records : {len(df):,}",
    f"Churn Rate    : {df['churn'].mean()*100:.1f}%",
    f"Retained      : {(df['churn']==0).sum():,}",
    f"Churned       : {(df['churn']==1).sum():,}",
    "\n--- Numeric Summary ---", df.describe(include="number").to_string(),
    "\n--- Churn by Geography ---",     churn_by("geography").to_string(),
    "\n--- Churn by Gender ---",        churn_by("gender").to_string(),
    "\n--- Churn by Num Products ---",  churn_by("num_products").to_string(),
    "\n--- Churn by Active Member ---", churn_by("is_active_member").to_string(),
    "\n--- Churn by Zero Balance ---",  churn_by(df["balance"].eq(0).rename("zero_balance")).to_string(),
]
with open(f"{REPORTS_DIR}/eda_summary.txt", "w") as f:
    f.write("\n".join(eda_lines))
print(f"   -> EDA saved to {REPORTS_DIR}/eda_summary.txt")

# -- EDA Plots ------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Customer Churn - Exploratory Data Analysis",
             color=TEXT_CLR, fontsize=16, fontweight="bold", y=1.01)

def rate_bar(ax, col, title, labels=None):
    cr = df.groupby(col)["churn"].mean().mul(100)
    ax.bar([labels.get(i, i) if labels else str(i) for i in cr.index], cr.values,
           color=["#E63946" if v > df["churn"].mean()*100 else "#2A9D8F" for v in cr.values],
           edgecolor=BG_COLOR)
    style_axes(ax, title, "", "Churn Rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

churn_counts = df["churn"].value_counts().sort_index()
axes[0,0].bar(["Retained", "Churned"], churn_counts.values,
              color=[PALETTE["retain"], PALETTE["churn"]], edgecolor=BG_COLOR, linewidth=1.5)
style_axes(axes[0,0], "Churn Distribution", "", "Count")

for label, grp in df.groupby("churn"):
    axes[0,1].hist(grp["age"], bins=25, alpha=0.7,
                   color=PALETTE["churn"] if label else PALETTE["retain"],
                   label="Churned" if label else "Retained", edgecolor=BG_COLOR)
axes[0,1].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR)
style_axes(axes[0,1], "Age Distribution by Churn", "Age", "Count")

rate_bar(axes[0,2], "geography", "Churn Rate by Geography")
rate_bar(axes[1,0], "num_products", "Churn Rate by Number of Products")
rate_bar(axes[1,1], "is_active_member", "Churn Rate by Activity", {0: "Inactive", 1: "Active"})
rate_bar(axes[1,2], df["balance"].eq(0).rename("zb"), "Churn Rate by Balance",
         {False: "Balance > 0", True: "Zero balance"})

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/eda_plots.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print(f"   -> EDA plots saved to {REPORTS_DIR}/eda_plots.png")

# ===============================================================================
# 3.  FEATURE ENGINEERING
# ===============================================================================
print("\nEngineering features ...")

def add_features(d):
    d = d.copy()
    d["zero_balance"]         = d["balance"].eq(0).astype(int)
    d["balance_to_salary"]    = d["balance"] / d["estimated_salary"].clip(lower=1)
    d["products_per_tenure"]  = d["num_products"] / (d["tenure_years"] + 1)
    d["inactive_multi_prod"]  = ((d["is_active_member"] == 0) & (d["num_products"] >= 2)).astype(int)
    d["senior"]               = d["age"].ge(50).astype(int)
    return d

df = add_features(df)

TARGET      = "churn"
CAT_FEATS   = ["geography", "gender"]
NUM_FEATS   = [c for c in df.columns if c not in ["customer_id", TARGET] + CAT_FEATS]

X = df.drop(columns=["customer_id", TARGET])
y = df[TARGET]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                  NUM_FEATS),
    ("cat", OneHotEncoder(drop="first",
                          sparse_output=False), CAT_FEATS),
], remainder="drop")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Class balance (train) - 0:{(y_train==0).sum()} | 1:{(y_train==1).sum()}")

# ===============================================================================
# 4.  MODEL TRAINING
# ===============================================================================
print("\nTraining models ...")

models = {
    "Logistic Regression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            max_iter=1000, C=0.5, class_weight="balanced", random_state=SEED
        ))
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ))
    ]),
    "LightGBM": Pipeline([
        ("pre", preprocessor),
        ("clf", LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            is_unbalance=True, importance_type="gain", random_state=SEED, verbose=-1
        ))
    ]),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for name, pipe in models.items():
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
    pipe.fit(X_train, y_train)
    y_prob  = pipe.predict_proba(X_test)[:, 1]
    y_pred  = (y_prob >= 0.5).astype(int)
    results[name] = {
        "pipe":     pipe,
        "y_prob":   y_prob,
        "y_pred":   y_pred,
        "auc":      roc_auc_score(y_test, y_prob),
        "pr_auc":   average_precision_score(y_test, y_prob),
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
    }
    joblib.dump(pipe, f"{MODELS_DIR}/{name.lower().replace(' ','_')}.pkl")
    print(f"   {name:<22}  Test AUC={results[name]['auc']:.4f}  "
          f"PR-AUC={results[name]['pr_auc']:.4f}  "
          f"CV AUC={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Model selection uses cross-validated AUC on the training set only, so the
# test set stays untouched until final reporting.
BEST = max(results, key=lambda n: results[n]["cv_mean"])
BASE = "Logistic Regression"
print(f"\n   Selected model (by CV AUC): {BEST}")

# ===============================================================================
# 5.  BUSINESS EVALUATION
# ===============================================================================
print("\nBusiness evaluation ...")
best_prob = results[BEST]["y_prob"]
y_true    = y_test.values

def capture_at(prob, frac):
    """Share of all churners found by contacting the top `frac` riskiest customers."""
    k = int(round(len(prob) * frac))
    top = np.argsort(-prob)[:k]
    return y_true[top].sum() / y_true.sum()

top10_capture = capture_at(best_prob, 0.10)
top20_capture = capture_at(best_prob, 0.20)
lift_top10    = top10_capture / 0.10

def net_value(prob, t, y):
    contact = prob >= t
    tp = (contact & (y == 1)).sum()
    return tp * SAVE_RATE * CUSTOMER_VALUE - contact.sum() * CONTACT_COST, int(contact.sum()), int(tp)

# The threshold is chosen on out-of-fold predictions for the TRAINING set, so the
# test set is only used to report the result, never to pick it.
oof_prob = cross_val_predict(clone(models[BEST]), X_train, y_train, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]
thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
train_curve = [(t, *net_value(oof_prob, t, y_train.values)) for t in thresholds]
best_t = max(train_curve, key=lambda r: r[1])[0]

curve = [(t, *net_value(best_prob, t, y_true)) for t in thresholds]   # test set, for reporting
best_net, best_n, best_tp = net_value(best_prob, best_t, y_true)
net_at_05, n_at_05, _ = net_value(best_prob, 0.5, y_true)
contact_all_net = y_true.sum() * SAVE_RATE * CUSTOMER_VALUE - len(y_true) * CONTACT_COST

print(f"   Top-10% riskiest customers contain {top10_capture:.1%} of churners "
      f"(lift {lift_top10:.1f}x vs random)")
print(f"   Profit-optimal threshold (chosen on training folds): {best_t:.2f} -> test set: contact {best_n} customers, "
      f"net value INR {best_net:,.0f} (vs INR {net_at_05:,.0f} at 0.50, "
      f"INR {contact_all_net:,.0f} if everyone is contacted)")

# ===============================================================================
# 6.  EVALUATION PLOTS
# ===============================================================================
print("\nGenerating evaluation charts ...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Model Evaluation (held-out test set)", color=TEXT_CLR, fontsize=15, fontweight="bold")

colors = {"Logistic Regression": "#F4A261", "Random Forest": "#457B9D", "LightGBM": "#2A9D8F"}

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})",
                 color=colors[name], linewidth=2.5)
axes[0].plot([0,1],[0,1],"--", color="#475569", linewidth=1)
axes[0].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
style_axes(axes[0], "ROC Curve", "False Positive Rate", "True Positive Rate")

cm = confusion_matrix(y_test, (best_prob >= best_t).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            ax=axes[1], linewidths=0.5, linecolor=BG_COLOR,
            cbar_kws={"shrink": 0.8})
axes[1].set_facecolor(BG_COLOR)
axes[1].set_title(f"Confusion Matrix - {BEST} @ t={best_t:.2f}", color=TEXT_CLR,
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted", color=TEXT_CLR)
axes[1].set_ylabel("Actual",    color=TEXT_CLR)
axes[1].tick_params(colors=TEXT_CLR)
axes[1].set_xticklabels(["Retained","Churned"], color=TEXT_CLR)
axes[1].set_yticklabels(["Retained","Churned"], color=TEXT_CLR, rotation=0)

axes[2].plot([c[0] for c in curve], [c[1]/1e5 for c in curve], color="#2A9D8F", linewidth=2.5)
axes[2].axvline(best_t, color="#E63946", linestyle="--", linewidth=1.5,
                label=f"chosen on train t={best_t:.2f}")
axes[2].axvline(0.5, color="#475569", linestyle=":", linewidth=1.5, label="default t=0.50")
axes[2].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
style_axes(axes[2], "Retention Campaign Net Value vs Threshold (test set)",
           "Churn risk-score threshold", "Net value (INR lakh)")

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/model_evaluation.png", dpi=150,
            bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"   -> {REPORTS_DIR}/model_evaluation.png")

# ===============================================================================
# 7.  FEATURE IMPORTANCE  (selected model)
# ===============================================================================
best_pipe = results[BEST]["pipe"]
pre_step  = best_pipe.named_steps["pre"]
cat_names = list(pre_step.named_transformers_["cat"].get_feature_names_out(CAT_FEATS))
feat_names = NUM_FEATS + cat_names

imp = best_pipe.named_steps["clf"].feature_importances_
fi_df = (pd.DataFrame({"feature": feat_names, "importance": imp / imp.sum()})
           .sort_values("importance", ascending=True))

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
bar_colors = [PALETTE["churn"] if v > fi_df["importance"].median()
              else "#457B9D" for v in fi_df["importance"]]
ax.barh(fi_df["feature"], fi_df["importance"], color=bar_colors, edgecolor=BG_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_CLR)
ax.tick_params(colors=TEXT_CLR, labelsize=9)
ax.set_title(f"Feature Importance - {BEST}",
             color=TEXT_CLR, fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Relative importance", color=TEXT_CLR, fontsize=10)
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=150,
            bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"   -> {REPORTS_DIR}/feature_importance.png")

# ===============================================================================
# 8.  SAVE REPORTS, METRICS AND SCORED TEST SET
# ===============================================================================
report_lines = []
for name, r in results.items():
    report_lines += [f"\n{'='*55}", f"  {name}", f"{'='*55}",
                     f"  Test AUC : {r['auc']:.4f}",
                     f"  PR-AUC   : {r['pr_auc']:.4f}",
                     f"  CV AUC   : {r['cv_mean']:.4f} +/- {r['cv_std']:.4f}",
                     "\n  (threshold 0.50)",
                     classification_report(y_test, r["y_pred"], target_names=["Retained","Churned"])]
with open(f"{REPORTS_DIR}/classification_reports.txt", "w") as f:
    f.write("\n".join(report_lines))

metrics = {
    "dataset": {"name": "Bank Customer Churn (Churn_Modelling.csv)", "rows": len(df),
                "churn_rate": round(float(df["churn"].mean()), 4), "test_rows": len(y_test)},
    "models": {n: {"test_auc": round(r["auc"], 4), "pr_auc": round(r["pr_auc"], 4),
                   "cv_auc_mean": round(r["cv_mean"], 4), "cv_auc_std": round(r["cv_std"], 4)}
               for n, r in results.items()},
    "best_model": BEST,
    "baseline_model": BASE,
    "business": {
        "assumptions": {"contact_cost_inr": CONTACT_COST, "customer_value_inr": CUSTOMER_VALUE,
                        "save_rate": SAVE_RATE},
        "top10_capture": round(float(top10_capture), 4),
        "top20_capture": round(float(top20_capture), 4),
        "lift_top10": round(float(lift_top10), 2),
        "optimal_threshold": float(best_t),
        "threshold_chosen_on": "out-of-fold predictions on the training set",
        "mean_risk_score_test": round(float(best_prob.mean()), 4),
        "customers_contacted": best_n,
        "churners_reached": best_tp,
        "net_value_optimal_inr": round(float(best_net)),
        "net_value_at_0_5_inr": float(net_at_05),
        "net_value_contact_all_inr": float(contact_all_net),
    },
    "feature_importance": fi_df.sort_values("importance", ascending=False)
                               .round(4).to_dict(orient="records"),
}
with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

test_out = X_test.copy()
test_out.insert(0, "customer_id", df.loc[test_out.index, "customer_id"])
test_out["actual_churn"]    = y_test.values
test_out["churn_prob"]      = best_prob
test_out["churn_prob_lr"]   = results[BASE]["y_prob"]
test_out["contact"]         = (best_prob >= best_t).astype(int)
test_out["risk_segment"]    = pd.cut(best_prob, bins=[-0.01, 0.30, 0.60, 1.0],
                                     labels=["Low Risk", "Medium Risk", "High Risk"])
test_out.to_csv("data/scored_customers.csv", index=False)

print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("="*60)
for name, r in results.items():
    print(f"  {name:<22} -> AUC {r['auc']:.4f}")
print(f"\n  Artifacts saved to models/, reports/ and data/scored_customers.csv")
print("="*60 + "\n")
