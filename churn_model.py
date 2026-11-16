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

# WHAT THIS FILE IS: the "brain" of the project. Learns who leaves the bank (CHURN = Exited 1) from 10,000
# real customers and gives each of the 2,000 test customers a churn risk score between 0 and 1.
# Example row 1: CustomerId 15634602, France, Female, Age 42, Balance 0, 1 product, Exited 1 (left).
# Results: LightGBM test AUC 0.8642 vs 0.7847 Logistic Regression; threshold 0.38 -> contact 751, net INR 6.08 lakh.
# Flow: read CSV -> checks -> EDA -> 5 new features -> 80/20 split -> 3 models + 5-fold CV
#   -> pick best -> business numbers -> charts -> save models/, reports/metrics.json, data/scored_customers.csv
import os, json, warnings, joblib  # folders, save JSON, hide warnings, save models as .pkl files
import numpy as np  # fast maths on arrays of numbers
import pandas as pd  # tables (DataFrame = rows x columns)
import matplotlib  # plotting library for the PNG charts
matplotlib.use("Agg")          # headless backend - no display needed
import matplotlib.pyplot as plt  # easy drawing interface (figures, bars, lines)
import matplotlib.ticker as mticker  # axis label format, e.g. "20%"
import seaborn as sns  # used only for the confusion-matrix heatmap

# CV = cross-validation: 5 folds, train on 4, score the 5th, repeat
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base             import clone  # fresh untrained copy of a model
from sklearn.preprocessing    import StandardScaler, OneHotEncoder  # rescale numbers, text -> 0/1 columns
from sklearn.linear_model     import LogisticRegression  # simple linear model = the baseline
from sklearn.ensemble         import RandomForestClassifier  # many decision trees, votes averaged
from sklearn.metrics          import (  # scoring functions
    roc_auc_score, average_precision_score, classification_report,  # AUC, PR-AUC, precision/recall table
    confusion_matrix, roc_curve,  # 2x2 count table, ROC curve points
)
from sklearn.pipeline         import Pipeline  # chain "preprocess -> model" into one object
from sklearn.compose          import ColumnTransformer  # different preprocessing per column
from lightgbm                 import LGBMClassifier  # gradient boosting: trees fix earlier trees' mistakes

warnings.filterwarnings("ignore")  # hide ALL warnings (useful ones too)

SEED = 42  # fixed randomness: same split, folds, trees every run

# ── BLOCK: Paths ─────────────────────────────────────────────────────────────
# Relative paths: run the script from the project folder.
DATA_PATH    = "data/Churn_Modelling.csv"  # 10,000 customers x 14 columns
MODELS_DIR   = "models"  # trained .pkl files go here
REPORTS_DIR  = "reports"  # charts, text reports, metrics.json
os.makedirs(MODELS_DIR,  exist_ok=True)  # create folder, no error if it exists
os.makedirs(REPORTS_DIR, exist_ok=True)  # same for reports/

# ── BLOCK: Business assumptions (used only for the threshold / ROI analysis) ──
# Hand-picked placeholders, not measured. Churner contacted: 0.30 x 10,000 - 500 = +INR 2,500; stayer: -INR 500.
CONTACT_COST      = 500      # cost of one retention offer + outreach (INR)
CUSTOMER_VALUE    = 10_000   # value of keeping one customer who would have churned (INR)
SAVE_RATE         = 0.30     # share of contacted true churners the offer actually retains

# ── BLOCK: Palette ───────────────────────────────────────────────────────────
# Same colours as the Streamlit dashboard, so charts and dashboard match.
PALETTE   = {"churn": "#E63946", "retain": "#2A9D8F"}  # red = churn, teal = retained
BG_COLOR  = "#0F172A"  # dark navy background
TEXT_CLR  = "#E2E8F0"  # light grey text
GRID_CLR  = "#1E293B"  # grid and border lines

# IN: one chart area (ax) + optional title/labels -> OUT: nothing, restyled in place (dark theme)
# e.g. style_axes(axes[0,0], "Churn Distribution", "", "Count")
def style_axes(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_COLOR)  # dark chart background
    ax.tick_params(colors=TEXT_CLR)  # light tick labels
    for spine in ax.spines.values():  # spines = 4 border lines
        spine.set_edgecolor(GRID_CLR)  # colour them grid-grey
    ax.xaxis.label.set_color(TEXT_CLR)  # light x label
    ax.yaxis.label.set_color(TEXT_CLR)  # light y label
    ax.title.set_color(TEXT_CLR)  # light title
    ax.grid(color=GRID_CLR, linewidth=0.5)  # thin grid lines
    ax.set_axisbelow(True)  # grid drawn behind the bars
    if title:   ax.set_title(title, fontsize=13, fontweight="bold", pad=10)  # only if given ("" skips)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=10)  # x label, only if given
    if ylabel:  ax.set_ylabel(ylabel, fontsize=10)  # y label, only if given

# ===============================================================================
# 1.  LOAD DATA
# ===============================================================================
print("\n" + "="*60)  # banner line of 60 "="
print("  CUSTOMER CHURN PREDICTION - ML PIPELINE")  # banner title
print("="*60)  # banner line

raw = pd.read_csv(DATA_PATH)  # 10,000 rows x 14 columns
# ── Step 1: rename to snake_case, Exited -> churn (the TARGET to predict) ──
df = raw.rename(columns={
    "CustomerId": "customer_id", "CreditScore": "credit_score", "Geography": "geography",  # ID, score, country
    "Gender": "gender", "Age": "age", "Tenure": "tenure_years", "Balance": "balance",  # person + account
    "NumOfProducts": "num_products", "HasCrCard": "has_credit_card",  # products, card 0/1
    "IsActiveMember": "is_active_member", "EstimatedSalary": "estimated_salary",  # active 0/1, salary
    "Exited": "churn",  # 1 = left the bank
}).drop(columns=["RowNumber", "Surname"])   # row index and surname carry no signal

# ── Step 2: data checks (assert = stop with message if False) ──
assert df["customer_id"].is_unique, "duplicate customers in source data"  # no duplicate customers
assert df.isnull().sum().sum() == 0, "unexpected missing values"  # zero empty cells in the table
print(f"\nLoaded {len(df):,} real customers with {df.shape[1]} columns.")  # "Loaded 10,000 ... 12 columns."

# ===============================================================================
# 2.  EXPLORATORY DATA ANALYSIS
# ===============================================================================
# EDA = simple counts and charts before modelling, to spot patterns.
print("\nRunning EDA ...")  # progress message

# IN: column name (or 0/1 Series) -> OUT: table of churn_rate (%) and n per group
# e.g. churn_by("geography") -> France 16.2% (n=5014), Germany 32.4% (n=2509), Spain 16.7%
def churn_by(col):
    return (df.groupby(col)["churn"].agg(["mean", "count"])  # mean of 0/1 = churn share, count = size
              .rename(columns={"mean": "churn_rate", "count": "n"})  # friendlier column names
              .assign(churn_rate=lambda t: (t["churn_rate"]*100).round(1)))  # 0.324 -> 32.4

# ── Step 1: build the lines of reports/eda_summary.txt ──
eda_lines = [
    "=" * 60, "  EDA SUMMARY - Bank Customer Churn Dataset", "=" * 60,  # header
    f"\nTotal Records : {len(df):,}",  # 10,000
    f"Churn Rate    : {df['churn'].mean()*100:.1f}%",  # 20.4%
    f"Retained      : {(df['churn']==0).sum():,}",  # 7,963
    f"Churned       : {(df['churn']==1).sum():,}",  # 2,037
    "\n--- Numeric Summary ---", df.describe(include="number").to_string(),  # mean/std/min/max per column
    "\n--- Churn by Geography ---",     churn_by("geography").to_string(),  # Germany 32.4%
    "\n--- Churn by Gender ---",        churn_by("gender").to_string(),  # per gender
    "\n--- Churn by Num Products ---",  churn_by("num_products").to_string(),  # 3 products -> 82.7%
    "\n--- Churn by Active Member ---", churn_by("is_active_member").to_string(),  # inactive vs active
    # 13.8% vs 24.1%
    "\n--- Churn by Zero Balance ---",  churn_by(df["balance"].eq(0).rename("zero_balance")).to_string(),
]
with open(f"{REPORTS_DIR}/eda_summary.txt", "w") as f:  # "w" = overwrite, auto-close
    f.write("\n".join(eda_lines))  # one line per list item
print(f"   -> EDA saved to {REPORTS_DIR}/eda_summary.txt")  # progress message

# ── BLOCK: EDA Plots (2 x 3 grid of charts) ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 6 chart slots, axes[row, col]
fig.patch.set_facecolor(BG_COLOR)  # dark figure background
fig.suptitle("Customer Churn - Exploratory Data Analysis",  # main title above all 6
             color=TEXT_CLR, fontsize=16, fontweight="bold", y=1.01)  # y=1.01 = just above the top

# IN: chart slot, column, title, optional {value: label} -> OUT: churn-rate bar chart
# e.g. {0: "Inactive", 1: "Active"} -> Inactive 26.9% (red), Active 14.3% (teal)
def rate_bar(ax, col, title, labels=None):
    cr = df.groupby(col)["churn"].mean().mul(100)  # churn % per group
    ax.bar([labels.get(i, i) if labels else str(i) for i in cr.index], cr.values,  # bar names + heights
           color=["#E63946" if v > df["churn"].mean()*100 else "#2A9D8F" for v in cr.values],  # red if above 20.4% avg
           edgecolor=BG_COLOR)  # bar edge = background colour
    style_axes(ax, title, "", "Churn Rate (%)")  # dark theme + labels
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())  # y axis shows "20%"

# ── Step 1: chart 1, retained vs churned counts (7,963 vs 2,037) ──
# Shows CLASS IMBALANCE: stayers about 4x churners.
churn_counts = df["churn"].value_counts().sort_index()  # counts of 0 then 1
axes[0,0].bar(["Retained", "Churned"], churn_counts.values,  # two bars
              color=[PALETTE["retain"], PALETTE["churn"]], edgecolor=BG_COLOR, linewidth=1.5)  # teal, red
style_axes(axes[0,0], "Churn Distribution", "", "Count")  # dark theme + labels

# ── Step 2: chart 2, age histograms for stayers vs churners, overlaid ──
for label, grp in df.groupby("churn"):  # runs twice: label 0, label 1
    axes[0,1].hist(grp["age"], bins=25, alpha=0.7,  # 25 age bands, 70% opaque
                   color=PALETTE["churn"] if label else PALETTE["retain"],  # 1 -> red, 0 -> teal
                   label="Churned" if label else "Retained", edgecolor=BG_COLOR)  # legend name
axes[0,1].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR)  # dark legend box
style_axes(axes[0,1], "Age Distribution by Churn", "Age", "Count")  # dark theme + labels

# ── Step 3: charts 3 to 6, churn rate per group ──
rate_bar(axes[0,2], "geography", "Churn Rate by Geography")  # Germany 32.4% stands out
rate_bar(axes[1,0], "num_products", "Churn Rate by Number of Products")  # 3 -> 82.7%, 4 -> 100%
rate_bar(axes[1,1], "is_active_member", "Churn Rate by Activity", {0: "Inactive", 1: "Active"})  # 0/1 -> words
rate_bar(axes[1,2], df["balance"].eq(0).rename("zb"), "Churn Rate by Balance",  # True if balance is 0
         {False: "Balance > 0", True: "Zero balance"})  # True/False -> words

# ── Step 4: save the PNG ──
plt.tight_layout()  # fix spacing so labels don't overlap
plt.savefig(f"{REPORTS_DIR}/eda_plots.png", dpi=150, bbox_inches="tight",  # 150 dpi, trim border
            facecolor=BG_COLOR)  # keep dark background in the file
plt.close()  # free memory
print(f"   -> EDA plots saved to {REPORTS_DIR}/eda_plots.png")  # progress message

# ===============================================================================
# 3.  FEATURE ENGINEERING
# ===============================================================================
# FEATURE = one input column; feature engineering = new useful columns from raw ones.
print("\nEngineering features ...")  # progress message

# IN: customer DataFrame -> OUT: copy with 5 new columns
# e.g. row 1 (Balance 0, 1 product, Tenure 2, Age 42): zero_balance 1, products_per_tenure 0.33, senior 0
def add_features(d):
    d = d.copy()  # don't change the caller's table
    d["zero_balance"]         = d["balance"].eq(0).astype(int)  # 1 if balance is 0 (36% of customers)
    d["balance_to_salary"]    = d["balance"] / d["estimated_salary"].clip(lower=1)  # clip = never divide by ~0
    d["products_per_tenure"]  = d["num_products"] / (d["tenure_years"] + 1)  # +1 because tenure can be 0
    # inactive, 2+ products
    d["inactive_multi_prod"]  = ((d["is_active_member"] == 0) & (d["num_products"] >= 2)).astype(int)
    d["senior"]               = d["age"].ge(50).astype(int)  # age 50+ churns 45% vs 16% (README)
    return d  # table with 5 extra columns

df = add_features(df)  # now 17 columns

# ── Step 1: name the target and feature groups ──
TARGET      = "churn"  # column to predict (1 = left)
CAT_FEATS   = ["geography", "gender"]  # CATEGORICAL = text categories
NUM_FEATS   = [c for c in df.columns if c not in ["customer_id", TARGET] + CAT_FEATS]  # 13 number columns

# ── Step 2: X = inputs, y = answer (customer_id dropped: an ID can't cause churn) ──
X = df.drop(columns=["customer_id", TARGET])  # 15 input columns
y = df[TARGET]  # 0/1 churn

# ── Step 3: preprocessing recipe, 13 scaled + 3 one-hot = 16 model inputs ──
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                  NUM_FEATS),  # mean 0, spread 1 (needed by LR)
    ("cat", OneHotEncoder(drop="first",  # drop France/Female, implied by zeros
                          sparse_output=False), CAT_FEATS),  # normal table, not sparse
], remainder="drop")  # unlisted columns thrown away

# ── Step 4: 80/20 split, stratified = both parts keep ~20% churners ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y  # 8,000 train, 2,000 test (407 churners)
)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")  # "Train: 8,000 | Test: 2,000"
print(f"   Class balance (train) - 0:{(y_train==0).sum()} | 1:{(y_train==1).sum()}")  # stayers vs churners

# ===============================================================================
# 4.  MODEL TRAINING
# ===============================================================================
print("\nTraining models ...")  # progress message

# ── BLOCK: 3 candidate models, each Pipeline "pre -> clf" ─────────────────────
# Hyperparameters set by hand, not tuned. Class weights fix imbalance (churners count more).
# All three share one preprocessor object; each fit() re-learns it on the same X_train, so no leak.
models = {
    "Logistic Regression": Pipeline([  # baseline the others must beat
        ("pre", preprocessor),  # scale + one-hot
        ("clf", LogisticRegression(
            max_iter=1000, C=0.5, class_weight="balanced", random_state=SEED  # C=0.5 = stronger regularisation
        ))
    ]),
    "Random Forest": Pipeline([
        ("pre", preprocessor),  # scale + one-hot
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,  # 300 trees, depth 10, leaf >= 5 rows
            class_weight="balanced", random_state=SEED, n_jobs=-1  # n_jobs=-1 = all CPU cores
        ))
    ]),
    "LightGBM": Pipeline([
        ("pre", preprocessor),  # scale + one-hot
        ("clf", LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,  # 400 small trees, small steps
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,  # each tree sees 80% rows, 80% columns
            is_unbalance=True, importance_type="gain", random_state=SEED, verbose=-1  # up-weight churners, quiet
        ))
    ]),
}

results = {}  # one entry per model: pipe, scores
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)  # same 5 folds reused for threshold

# ── BLOCK: CV on train -> fit on all train -> score test -> save .pkl ────────
for name, pipe in models.items():  # 3 models
    cv_scores = cross_val_score(pipe, X_train, y_train,  # 5 AUCs, training data only
                                cv=cv, scoring="roc_auc", n_jobs=-1)  # AUC per fold
    pipe.fit(X_train, y_train)  # train on all 8,000
    y_prob  = pipe.predict_proba(X_test)[:, 1]  # churn score column, 2,000 values
    y_pred  = (y_prob >= 0.5).astype(int)  # default threshold 0.5, report only
    results[name] = {  # LightGBM real: auc 0.8642, pr_auc 0.7159, cv 0.8631 +/- 0.0105
        "pipe":     pipe,  # fitted pipeline
        "y_prob":   y_prob,  # test scores
        "y_pred":   y_pred,  # test 0/1 at 0.5
        "auc":      roc_auc_score(y_test, y_prob),  # AUC: churner ranked above stayer how often
        "pr_auc":   average_precision_score(y_test, y_prob),  # PR-AUC, random ~0.20
        "cv_mean":  cv_scores.mean(),  # average of 5 fold AUCs
        "cv_std":   cv_scores.std(),  # spread of 5 fold AUCs
    }
    joblib.dump(pipe, f"{MODELS_DIR}/{name.lower().replace(' ','_')}.pkl")  # e.g. models/lightgbm.pkl
    print(f"   {name:<22}  Test AUC={results[name]['auc']:.4f}  "  # name padded to 22 chars
          f"PR-AUC={results[name]['pr_auc']:.4f}  "  # PR-AUC
          f"CV AUC={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")  # CV mean +/- std

# ── Step: pick best by CV AUC on train only (test stays untouched) ──
BEST = max(results, key=lambda n: results[n]["cv_mean"])  # LightGBM 0.8631 beats RF 0.8594
BASE = "Logistic Regression"  # baseline kept for comparison
print(f"\n   Selected model (by CV AUC): {BEST}")  # "Selected model ...: LightGBM"

# ===============================================================================
# 5.  BUSINESS EVALUATION
# ===============================================================================
# Scores -> business answers: how many churners reached, how much money.
print("\nBusiness evaluation ...")  # progress message
best_prob = results[BEST]["y_prob"]  # LightGBM scores, 2,000 test customers
y_true    = y_test.values  # real 0/1 as numpy array

# IN: scores + fraction like 0.10 -> OUT: share of all churners in the top fraction
# e.g. capture_at(best_prob, 0.10) -> top 200 hold 41.28% of the 407 churners
def capture_at(prob, frac):
    """Share of all churners found by contacting the top `frac` riskiest customers."""
    k = int(round(len(prob) * frac))  # 2000 x 0.10 = 200
    top = np.argsort(-prob)[:k]  # positions of the k highest scores
    return y_true[top].sum() / y_true.sum()  # uses global y_true (test set)

top10_capture = capture_at(best_prob, 0.10)  # 0.4128
top20_capture = capture_at(best_prob, 0.20)  # 0.629
lift_top10    = top10_capture / 0.10  # LIFT vs random = 4.13x

# IN: scores, threshold t, true 0/1 -> OUT: (net INR, contacted, churners reached)
# e.g. t=0.38 on test: 328 x 0.30 x 10,000 - 751 x 500 = INR 608,500
def net_value(prob, t, y):
    contact = prob >= t  # True = we contact this customer
    tp = (contact & (y == 1)).sum()  # true positives: contacted and churned
    return tp * SAVE_RATE * CUSTOMER_VALUE - contact.sum() * CONTACT_COST, int(contact.sum()), int(tp)  # money, n, tp

# ── Step 1: choose threshold on TRAINING out-of-fold scores (test never used to pick) ──
# out-of-fold = each training row scored by the model that did not train on it
oof_prob = cross_val_predict(clone(models[BEST]), X_train, y_train, cv=cv,  # fresh LightGBM, same 5 folds
                             method="predict_proba", n_jobs=-1)[:, 1]  # churn score column
thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)  # 0.05 to 0.95, 91 values
train_curve = [(t, *net_value(oof_prob, t, y_train.values)) for t in thresholds]  # (t, net, n, tp) each
best_t = max(train_curve, key=lambda r: r[1])[0]  # highest net value -> 0.38

# ── Step 2: report results on the test set ──
curve = [(t, *net_value(best_prob, t, y_true)) for t in thresholds]   # test set, for reporting
best_net, best_n, best_tp = net_value(best_prob, best_t, y_true)  # INR 608,500, 751, 328
# At 0.50: INR 626,000, slightly more; test curve is flat 0.3 to 0.6, 0.38 is the fair (train-picked) number
net_at_05, n_at_05, _ = net_value(best_prob, 0.5, y_true)  # default threshold, comparison
contact_all_net = y_true.sum() * SAVE_RATE * CUSTOMER_VALUE - len(y_true) * CONTACT_COST  # call all: INR 221,000

print(f"   Top-10% riskiest customers contain {top10_capture:.1%} of churners "  # "41.3%"
      f"(lift {lift_top10:.1f}x vs random)")  # "4.1x"
print(f"   Profit-optimal threshold (chosen on training folds): {best_t:.2f} -> test set: contact {best_n} customers, "
      f"net value INR {best_net:,.0f} (vs INR {net_at_05:,.0f} at 0.50, "  # 608,500 vs 626,000
      f"INR {contact_all_net:,.0f} if everyone is contacted)")  # 221,000

# ===============================================================================
# 6.  EVALUATION PLOTS
# ===============================================================================
print("\nGenerating evaluation charts ...")  # progress message

fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 3 charts side by side
fig.patch.set_facecolor(BG_COLOR)  # dark figure background
fig.suptitle("Model Evaluation (held-out test set)", color=TEXT_CLR, fontsize=15, fontweight="bold")  # main title

colors = {"Logistic Regression": "#F4A261", "Random Forest": "#457B9D", "LightGBM": "#2A9D8F"}  # line colour per model

# ── Step 1: ROC curves (false positive rate vs true positive rate) ──
for name, r in results.items():  # one line per model
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])  # points at every threshold
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})",  # legend shows AUC
                 color=colors[name], linewidth=2.5)  # model colour
axes[0].plot([0,1],[0,1],"--", color="#475569", linewidth=1)  # dashed diagonal = random (AUC 0.5)
axes[0].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)  # dark legend
style_axes(axes[0], "ROC Curve", "False Positive Rate", "True Positive Rate")  # dark theme + labels

# ── Step 2: confusion matrix of LightGBM at t=0.38 (rows actual, cols predicted) ──
cm = confusion_matrix(y_test, (best_prob >= best_t).astype(int))  # 2x2 counts, bottom-right 328
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",  # write counts, whole numbers, blue scale
            ax=axes[1], linewidths=0.5, linecolor=BG_COLOR,  # middle slot, thin cell borders
            cbar_kws={"shrink": 0.8})  # smaller colour bar
axes[1].set_facecolor(BG_COLOR)  # styled by hand, not style_axes
axes[1].set_title(f"Confusion Matrix - {BEST} @ t={best_t:.2f}", color=TEXT_CLR,  # "... @ t=0.38"
                  fontsize=12, fontweight="bold")  # title font
axes[1].set_xlabel("Predicted", color=TEXT_CLR)  # columns = predicted
axes[1].set_ylabel("Actual",    color=TEXT_CLR)  # rows = actual
axes[1].tick_params(colors=TEXT_CLR)  # light ticks
axes[1].set_xticklabels(["Retained","Churned"], color=TEXT_CLR)  # 0/1 -> words
axes[1].set_yticklabels(["Retained","Churned"], color=TEXT_CLR, rotation=0)  # horizontal labels

# ── Step 3: test net value vs threshold, in INR lakh (1 lakh = 100,000) ──
axes[2].plot([c[0] for c in curve], [c[1]/1e5 for c in curve], color="#2A9D8F", linewidth=2.5)  # t vs lakh
axes[2].axvline(best_t, color="#E63946", linestyle="--", linewidth=1.5,  # red dashed at 0.38
                label=f"chosen on train t={best_t:.2f}")  # legend text
axes[2].axvline(0.5, color="#475569", linestyle=":", linewidth=1.5, label="default t=0.50")  # grey dotted at 0.5
axes[2].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)  # dark legend
style_axes(axes[2], "Retention Campaign Net Value vs Threshold (test set)",  # title
           "Churn risk-score threshold", "Net value (INR lakh)")  # x, y labels

# ── Step 4: save the PNG ──
plt.tight_layout()  # fix spacing
plt.savefig(f"{REPORTS_DIR}/model_evaluation.png", dpi=150,  # 150 dpi
            bbox_inches="tight", facecolor=BG_COLOR)  # trim border, dark background
plt.close()  # free memory
print(f"   -> {REPORTS_DIR}/model_evaluation.png")  # progress message

# ===============================================================================
# 7.  FEATURE IMPORTANCE  (selected model)
# ===============================================================================
# Gain importance: top 5 = age 0.29, num_products 0.20, balance 0.10, credit_score 0.07, balance_to_salary 0.07.
best_pipe = results[BEST]["pipe"]  # fitted LightGBM pipeline
pre_step  = best_pipe.named_steps["pre"]  # its fitted preprocessor
cat_names = list(pre_step.named_transformers_["cat"].get_feature_names_out(CAT_FEATS))  # geography_Germany, ...
feat_names = NUM_FEATS + cat_names  # 16 names, same order as model inputs

imp = best_pipe.named_steps["clf"].feature_importances_  # raw gain per feature
fi_df = (pd.DataFrame({"feature": feat_names, "importance": imp / imp.sum()})  # shares summing to 1
           .sort_values("importance", ascending=True))  # smallest first -> biggest bar on top

# ── Step 1: horizontal bar chart ──
fig, ax = plt.subplots(figsize=(10, 7))  # one chart, 10 x 7 inches
fig.patch.set_facecolor(BG_COLOR)  # dark figure background
ax.set_facecolor(BG_COLOR)  # dark chart background
bar_colors = [PALETTE["churn"] if v > fi_df["importance"].median()  # above median -> red
              else "#457B9D" for v in fi_df["importance"]]  # rest -> blue
ax.barh(fi_df["feature"], fi_df["importance"], color=bar_colors, edgecolor=BG_COLOR)  # barh = horizontal bars
for spine in ax.spines.values():  # 4 border lines
    spine.set_edgecolor(GRID_CLR)  # grid-grey
ax.tick_params(colors=TEXT_CLR, labelsize=9)  # light, small ticks
ax.set_title(f"Feature Importance - {BEST}",  # "Feature Importance - LightGBM"
             color=TEXT_CLR, fontsize=14, fontweight="bold", pad=12)  # title style
ax.set_xlabel("Relative importance", color=TEXT_CLR, fontsize=10)  # x label
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5)  # vertical grid lines only
plt.tight_layout()  # fix spacing
plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=150,  # 150 dpi
            bbox_inches="tight", facecolor=BG_COLOR)  # trim border, dark background
plt.close()  # free memory
print(f"   -> {REPORTS_DIR}/feature_importance.png")  # progress message

# ===============================================================================
# 8.  SAVE REPORTS, METRICS AND SCORED TEST SET
# ===============================================================================
# ── Step 1: classification_reports.txt (LightGBM churn row at 0.50: precision 0.53, recall 0.75) ──
report_lines = []  # text lines to write
for name, r in results.items():  # one section per model
    report_lines += [f"\n{'='*55}", f"  {name}", f"{'='*55}",  # section header
                     f"  Test AUC : {r['auc']:.4f}",  # e.g. 0.8642
                     f"  PR-AUC   : {r['pr_auc']:.4f}",  # e.g. 0.7159
                     f"  CV AUC   : {r['cv_mean']:.4f} +/- {r['cv_std']:.4f}",  # e.g. 0.8631 +/- 0.0105
                     "\n  (threshold 0.50)",  # report uses default cut-off
                     # precision/recall/F1
                     classification_report(y_test, r["y_pred"], target_names=["Retained","Churned"])]
with open(f"{REPORTS_DIR}/classification_reports.txt", "w") as f:  # overwrite file
    f.write("\n".join(report_lines))  # write all lines

# ── Step 2: metrics.json, read by dashboard and report_generator.py ──
# float()/int() turn numpy numbers into plain Python (json can't save numpy types).
metrics = {
    "dataset": {"name": "Bank Customer Churn (Churn_Modelling.csv)", "rows": len(df),  # 10,000 rows
                "churn_rate": round(float(df["churn"].mean()), 4), "test_rows": len(y_test)},  # 0.2037, 2,000
    "models": {n: {"test_auc": round(r["auc"], 4), "pr_auc": round(r["pr_auc"], 4),  # per-model scores
                   "cv_auc_mean": round(r["cv_mean"], 4), "cv_auc_std": round(r["cv_std"], 4)}  # CV mean, std
               for n, r in results.items()},  # one entry per model
    "best_model": BEST,  # "LightGBM"
    "baseline_model": BASE,  # "Logistic Regression"
    # mean score 0.3506 > 20% churn rate (class weighting), so "risk scores", not calibrated probabilities
    "business": {
        "assumptions": {"contact_cost_inr": CONTACT_COST, "customer_value_inr": CUSTOMER_VALUE,  # 500, 10,000
                        "save_rate": SAVE_RATE},  # 0.30
        "top10_capture": round(float(top10_capture), 4),  # 0.4128
        "top20_capture": round(float(top20_capture), 4),  # 0.629
        "lift_top10": round(float(lift_top10), 2),  # 4.13
        "optimal_threshold": float(best_t),  # 0.38
        "threshold_chosen_on": "out-of-fold predictions on the training set",  # honesty note
        "mean_risk_score_test": round(float(best_prob.mean()), 4),  # 0.3506
        "customers_contacted": best_n,  # 751
        "churners_reached": best_tp,  # 328
        "net_value_optimal_inr": round(float(best_net)),  # 608,500
        "net_value_at_0_5_inr": float(net_at_05),  # 626,000
        "net_value_contact_all_inr": float(contact_all_net),  # 221,000
    },
    "feature_importance": fi_df.sort_values("importance", ascending=False)  # biggest first
                               .round(4).to_dict(orient="records"),  # [{"feature": "age", "importance": 0.2917}, ...]
}
with open(f"{REPORTS_DIR}/metrics.json", "w") as f:  # overwrite file
    json.dump(metrics, f, indent=2)  # indent=2 = human-readable

# ── Step 3: data/scored_customers.csv, the dashboard's data (2,000 test rows) ──
test_out = X_test.copy()  # start from test inputs
test_out.insert(0, "customer_id", df.loc[test_out.index, "customer_id"])  # ID back as first column, matched by index
test_out["actual_churn"]    = y_test.values  # real outcome 0/1
test_out["churn_prob"]      = best_prob  # LightGBM risk score
test_out["churn_prob_lr"]   = results[BASE]["y_prob"]  # LR score, for comparison
test_out["contact"]         = (best_prob >= best_t).astype(int)  # 1 if score >= 0.38
# Bins (-0.01,0.30] Low, (0.30,0.60] Medium, (0.60,1.0] High; -0.01 so score 0 is Low. High 462 (60.0% churned)
test_out["risk_segment"]    = pd.cut(best_prob, bins=[-0.01, 0.30, 0.60, 1.0],  # cut scores into 3 bins
                                     labels=["Low Risk", "Medium Risk", "High Risk"])  # bin names
test_out.to_csv("data/scored_customers.csv", index=False)  # no pandas row-number column

# ── Step 4: final banner ──
print("\n" + "="*60)  # banner line
print("  PIPELINE COMPLETE")  # banner title
print("="*60)  # banner line
for name, r in results.items():  # one line per model
    print(f"  {name:<22} -> AUC {r['auc']:.4f}")  # e.g. "LightGBM -> AUC 0.8642"
print(f"\n  Artifacts saved to models/, reports/ and data/scored_customers.csv")  # where outputs went
print("="*60 + "\n")  # banner line
