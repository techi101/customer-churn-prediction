"""
churn_model.py
--------------
End-to-end machine-learning pipeline for Customer Churn Prediction.

Steps:
  1. Load & validate data
  2. Exploratory Data Analysis  (EDA) – saved to reports/eda_summary.txt
  3. Preprocessing              – encoding, scaling, SMOTE for class balance
  4. Model training             – Logistic Regression + Random Forest
  5. Evaluation                 – ROC-AUC, Classification Report, Confusion Matrix
  6. Feature Importance         – saved chart to reports/feature_importance.png
  7. Artefact export            – models/  and  reports/

Run:
    python churn_model.py
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend – no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import OneHotEncoder

warnings.filterwarnings("ignore")

SEED = 42

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/banking_customers.csv"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Palette ─────────────────────────────────────────────────────────────────────
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
    if title:   ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:  ax.set_ylabel(ylabel, fontsize=10)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  CUSTOMER CHURN PREDICTION — ML PIPELINE")
print("═"*60)

df = pd.read_csv(DATA_PATH)
print(f"\n✅  Loaded {len(df):,} records with {df.shape[1]} features.")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📊  Running EDA …")

eda_lines = []
eda_lines.append("=" * 60)
eda_lines.append("  EDA SUMMARY — Customer Churn Dataset")
eda_lines.append("=" * 60)
eda_lines.append(f"\nTotal Records : {len(df):,}")
eda_lines.append(f"Features      : {df.shape[1]}")
eda_lines.append(f"\nChurn Rate    : {df['churn'].mean()*100:.1f}%")
eda_lines.append(f"Retained      : {(df['churn']==0).sum():,}")
eda_lines.append(f"Churned       : {(df['churn']==1).sum():,}")
eda_lines.append("\n--- Numeric Summary ---")
eda_lines.append(df.describe(include="number").to_string())
eda_lines.append("\n--- Missing Values ---")
eda_lines.append(df.isnull().sum().to_string())
eda_lines.append("\n--- Churn by Geography ---")
geo_churn = df.groupby("geography")["churn"].agg(["mean","count"]).rename(
    columns={"mean":"churn_rate","count":"n"})
geo_churn["churn_rate"] = (geo_churn["churn_rate"]*100).round(1)
eda_lines.append(geo_churn.to_string())
eda_lines.append("\n--- Churn by Num Products ---")
prod_churn = df.groupby("num_products")["churn"].mean().mul(100).round(1)
eda_lines.append(prod_churn.to_string())

with open(f"{REPORTS_DIR}/eda_summary.txt", "w") as f:
    f.write("\n".join(eda_lines))
print(f"   → EDA saved to {REPORTS_DIR}/eda_summary.txt")

# ── EDA Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Customer Churn – Exploratory Data Analysis",
             color=TEXT_CLR, fontsize=16, fontweight="bold", y=1.01)

# a) Churn distribution
churn_counts = df["churn"].value_counts()
axes[0,0].bar(["Retained", "Churned"], churn_counts.values,
              color=[PALETTE["retain"], PALETTE["churn"]], edgecolor=BG_COLOR, linewidth=1.5)
style_axes(axes[0,0], "Churn Distribution", "", "Count")

# b) Age distribution by churn
for label, grp in df.groupby("churn"):
    axes[0,1].hist(grp["age"], bins=25, alpha=0.7,
                   color=PALETTE["churn"] if label else PALETTE["retain"],
                   label="Churned" if label else "Retained", edgecolor=BG_COLOR)
axes[0,1].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR)
style_axes(axes[0,1], "Age Distribution by Churn", "Age", "Count")

# c) Churn rate by geography
geo_cr = df.groupby("geography")["churn"].mean().mul(100).sort_values(ascending=False)
axes[0,2].bar(geo_cr.index, geo_cr.values,
              color=["#E63946","#F4A261","#2A9D8F","#457B9D"], edgecolor=BG_COLOR)
style_axes(axes[0,2], "Churn Rate by Geography", "", "Churn Rate (%)")
axes[0,2].yaxis.set_major_formatter(mticker.PercentFormatter())

# d) Spend trend vs churn (box)
spend_box_data = pd.DataFrame({
    "Spend Trend": df["spend_trend"],
    "Status": df["churn"].map({0: "Retained", 1: "Churned"})
})
sns.boxplot(
    data=spend_box_data, x="Status", y="Spend Trend", ax=axes[1,0],
    palette={"Retained": PALETTE["retain"], "Churned": PALETTE["churn"]},
    linewidth=1.5
)
axes[1,0].set_facecolor(BG_COLOR)
style_axes(axes[1,0], "Spend Trend by Churn Status", "", "Spend Trend")

# e) Complaint frequency distribution
for label, grp in df.groupby("churn"):
    axes[1,1].hist(grp["complaint_frequency"], bins=20, alpha=0.75,
                   color=PALETTE["churn"] if label else PALETTE["retain"],
                   label="Churned" if label else "Retained", edgecolor=BG_COLOR)
axes[1,1].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR)
style_axes(axes[1,1], "Complaint Frequency by Churn", "Complaints/Year", "Count")

# f) Reward redemption rate
for label, grp in df.groupby("churn"):
    axes[1,2].hist(grp["reward_redemption_rate"], bins=20, alpha=0.75,
                   color=PALETTE["churn"] if label else PALETTE["retain"],
                   label="Churned" if label else "Retained", edgecolor=BG_COLOR)
axes[1,2].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR)
style_axes(axes[1,2], "Reward Redemption Rate", "Rate", "Count")

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/eda_plots.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print(f"   → EDA plots saved to {REPORTS_DIR}/eda_plots.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🔧  Preprocessing …")

DROP_COLS   = ["customer_id"]
TARGET      = "churn"
CAT_FEATS   = ["geography"]
NUM_FEATS   = [c for c in df.columns if c not in DROP_COLS + [TARGET] + CAT_FEATS]

X = df.drop(columns=DROP_COLS + [TARGET])
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
print(f"   Class balance (train) — 0:{(y_train==0).sum()} | 1:{(y_train==1).sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🤖  Training models …")

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
            n_estimators=200, max_depth=12, min_samples_leaf=8,
            class_weight="balanced", random_state=SEED, n_jobs=-1
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
    y_pred  = pipe.predict(X_test)
    auc     = roc_auc_score(y_test, y_prob)
    results[name] = {
        "pipe":     pipe,
        "y_prob":   y_prob,
        "y_pred":   y_pred,
        "auc":      auc,
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
    }
    joblib.dump(pipe, f"{MODELS_DIR}/{name.lower().replace(' ','_')}.pkl")
    print(f"   {name:<22}  Test AUC={auc:.4f}  "
          f"CV AUC={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n📈  Generating evaluation charts …")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Model Evaluation", color=TEXT_CLR, fontsize=15, fontweight="bold")

colors = {"Logistic Regression": "#F4A261", "Random Forest": "#2A9D8F"}

# a) ROC curves
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.4f})",
                 color=colors[name], linewidth=2.5)
axes[0].plot([0,1],[0,1],"--", color="#475569", linewidth=1)
axes[0].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
style_axes(axes[0], "ROC Curve", "False Positive Rate", "True Positive Rate")

# b) Confusion Matrix – Random Forest
cm = confusion_matrix(y_test, results["Random Forest"]["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            ax=axes[1], linewidths=0.5, linecolor=BG_COLOR,
            cbar_kws={"shrink": 0.8})
axes[1].set_facecolor(BG_COLOR)
axes[1].set_title("Confusion Matrix — Random Forest", color=TEXT_CLR,
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted", color=TEXT_CLR)
axes[1].set_ylabel("Actual",    color=TEXT_CLR)
axes[1].tick_params(colors=TEXT_CLR)
axes[1].set_xticklabels(["Retained","Churned"], color=TEXT_CLR)
axes[1].set_yticklabels(["Retained","Churned"], color=TEXT_CLR, rotation=0)

# c) CV AUC comparison bar
names_  = list(results.keys())
means_  = [results[n]["cv_mean"] for n in names_]
stds_   = [results[n]["cv_std"]  for n in names_]
bar_clr = [colors[n] for n in names_]
axes[2].bar(names_, means_, yerr=stds_, color=bar_clr,
            capsize=8, edgecolor=BG_COLOR, linewidth=1.5)
axes[2].set_ylim(0.5, 1.0)
for i,(m,s) in enumerate(zip(means_, stds_)):
    axes[2].text(i, m + s + 0.005, f"{m:.4f}", ha="center",
                 color=TEXT_CLR, fontsize=10, fontweight="bold")
style_axes(axes[2], "5-Fold CV AUC Comparison", "", "AUC Score")

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/model_evaluation.png", dpi=150,
            bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"   → {REPORTS_DIR}/model_evaluation.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  FEATURE IMPORTANCE  (Random Forest)
# ═══════════════════════════════════════════════════════════════════════════════
rf_pipe  = results["Random Forest"]["pipe"]
rf_model = rf_pipe.named_steps["clf"]
pre_step = rf_pipe.named_steps["pre"]

# Rebuild feature names after OHE
num_names  = NUM_FEATS
cat_names  = list(pre_step.named_transformers_["cat"]
                  .get_feature_names_out(CAT_FEATS))
feat_names = num_names + cat_names

importances = rf_model.feature_importances_
fi_df = (
    pd.DataFrame({"feature": feat_names, "importance": importances})
    .sort_values("importance", ascending=True)
)

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

bar_colors = [PALETTE["churn"] if v > fi_df["importance"].median()
              else "#457B9D" for v in fi_df["importance"]]
ax.barh(fi_df["feature"], fi_df["importance"], color=bar_colors, edgecolor=BG_COLOR)
for spine in ax.spines.values():
    spine.set_edgecolor(GRID_CLR)
ax.tick_params(colors=TEXT_CLR, labelsize=9)
ax.set_title("Feature Importance — Random Forest",
             color=TEXT_CLR, fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Importance Score", color=TEXT_CLR, fontsize=10)
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)

plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=150,
            bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"   → {REPORTS_DIR}/feature_importance.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  SAVE CLASSIFICATION REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
report_lines = []
for name, r in results.items():
    report_lines.append(f"\n{'='*55}")
    report_lines.append(f"  {name}")
    report_lines.append(f"{'='*55}")
    report_lines.append(f"  Test AUC : {r['auc']:.4f}")
    report_lines.append(f"  CV AUC   : {r['cv_mean']:.4f} ± {r['cv_std']:.4f}")
    report_lines.append("\n" + classification_report(
        y_test, r["y_pred"], target_names=["Retained","Churned"]
    ))

with open(f"{REPORTS_DIR}/classification_reports.txt", "w") as f:
    f.write("\n".join(report_lines))

# ── Save scored test set for dashboard ─────────────────────────────────────────
test_out = X_test.copy()
test_out["customer_id"]     = df.loc[test_out.index, "customer_id"]
test_out["actual_churn"]    = y_test.values
test_out["churn_prob_rf"]   = results["Random Forest"]["y_prob"]
test_out["churn_prob_lr"]   = results["Logistic Regression"]["y_prob"]
test_out["predicted_churn"] = results["Random Forest"]["y_pred"]
test_out["risk_segment"]    = pd.cut(
    test_out["churn_prob_rf"],
    bins=[0, 0.30, 0.60, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)
test_out.to_csv("data/scored_customers.csv", index=False)

print("\n" + "═"*60)
print("  PIPELINE COMPLETE")
print("═"*60)
for name, r in results.items():
    print(f"  {name:<22} → AUC {r['auc']:.4f}")
print(f"\n  Artifacts saved to:")
print(f"    models/   — trained model pipelines (.pkl)")
print(f"    reports/  — charts & evaluation reports")
print(f"    data/scored_customers.csv — test set with churn probabilities")
print("═"*60 + "\n")
