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

# WHAT THIS FILE IS: the "brain" of the project. It reads 10,000 real bank customers, learns the pattern of who
# leaves the bank ("churns"), and gives every test customer a churn RISK SCORE between 0 and 1.
# Analogy: like a doctor who has seen 8,000 old patient files (training data), and is then tested on 2,000 new
# patients (test data) whose outcome the doctor never saw, to prove the skill is real and not memorised.
# CHURN = a customer closing their account / leaving the bank. In the CSV it is the column Exited (1 = left, 0 = stayed).
# Real example row (row 1 of data/Churn_Modelling.csv): CustomerId 15634602, CreditScore 619, Geography France,
#   Gender Female, Age 42, Tenure 2, Balance 0, NumOfProducts 1, HasCrCard 1, IsActiveMember 1,
#   EstimatedSalary 101348.88, Exited 1  -> this customer left the bank.
# Real results (README.md / reports/metrics.json): 20.4% of customers churned. LightGBM test AUC 0.864
#   (CV 0.863 +/- 0.011) vs 0.785 for Logistic Regression. The top 10% of scores contain 41% of all churners
#   (lift 4.1x). Contact threshold 0.38 picked on the training folds; on the test set it contacts 751 customers,
#   reaches 328 churners and is worth INR 6.08 lakh vs INR 2.21 lakh if everyone is contacted.
# Overall flow: read CSV -> check data -> EDA summary + charts -> add 5 new features -> split 80/20 train/test
#   -> train 3 models (5-fold CV on train) -> pick best by CV AUC -> business numbers (capture, lift, threshold)
#   -> charts -> feature importance -> save models/, reports/metrics.json, data/scored_customers.csv
#
# os = talk to the operating system (make folders). json = save the metrics dict as a .json text file.
# warnings = Python's warning messages (we switch them off below). joblib = save a trained model to a .pkl file
# on disk (".pkl" = pickle = a Python object frozen into a file) so the model can be loaded later without retraining.
import os, json, warnings, joblib
# numpy (np) = fast maths on arrays (lists of numbers). Used for sorting scores, thresholds, counting.
import numpy as np
# pandas (pd) = tables in Python. A DataFrame is a table: rows = customers, columns = CreditScore, Age, Balance...
# A Series is one column of a DataFrame.
import pandas as pd
# matplotlib = the standard Python plotting library. It makes the PNG charts in reports/.
import matplotlib
# "Agg" backend = draw charts straight into image files, without opening a window. Needed when there is no screen
# (a server) and makes the script never pop up windows. It must be set BEFORE pyplot is imported.
matplotlib.use("Agg")          # headless backend - no display needed
# pyplot (plt) = the easy drawing interface of matplotlib: figures, bars, lines, save as PNG.
import matplotlib.pyplot as plt
# ticker = controls axis labels; we use PercentFormatter so the y axis shows "20%" instead of "20".
import matplotlib.ticker as mticker
# seaborn (sns) = a plotting library built on matplotlib. Used here only for the confusion-matrix heatmap.
import seaborn as sns

# scikit-learn (sklearn) = the main Python machine-learning library. From model_selection we take:
#   train_test_split = cut the data into a TRAIN part (the model learns from it) and a TEST part (kept hidden,
#     used only once at the end to measure how good the model really is).
#   StratifiedKFold = cross-validation splitter. CROSS-VALIDATION (CV) = cut the training data into 5 equal parts
#     called FOLDS; train on 4, score on the 5th, repeat 5 times so every fold is scored once, then average.
#     STRATIFIED = every fold keeps the same churn share (about 20%) as the whole data.
#   cross_val_score = run that 5-fold loop and return one score per fold.
#   cross_val_predict = run the 5-fold loop and return, for every training row, the prediction made by the model
#     that did NOT see that row ("out-of-fold" prediction). Used to pick the threshold honestly.
from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
# clone = make a fresh, untrained copy of a model with the same settings.
from sklearn.base             import clone
# StandardScaler = rescale each number column to mean 0 and spread (standard deviation) 1, so Balance (up to
#   250,898) and NumOfProducts (1 to 4) are on the same scale. Logistic Regression needs this; trees do not care.
# OneHotEncoder = ONE-HOT ENCODING turns a text column into 0/1 columns, because models only understand numbers.
#   Example: Geography France/Germany/Spain -> geography_Germany (0/1) and geography_Spain (0/1).
from sklearn.preprocessing    import StandardScaler, OneHotEncoder
# LogisticRegression = a simple linear model: adds up (weight x feature) and squashes it to 0..1.
#   It is the BASELINE here = the simple model the fancy models must beat.
from sklearn.linear_model     import LogisticRegression
# RandomForestClassifier = many decision trees, each trained on a random sample of rows; their votes are averaged.
#   A DECISION TREE asks yes/no questions like "Age > 45?" then "NumOfProducts >= 3?" until it reaches an answer.
from sklearn.ensemble         import RandomForestClassifier
# Scoring functions:
#   roc_auc_score = AUC (Area Under the ROC Curve). It answers: "pick one random churner and one random stayer;
#     how often does the model give the churner the higher score?" 0.5 = coin toss, 1.0 = perfect. LightGBM = 0.864.
#   average_precision_score = PR-AUC, area under the precision-recall curve. PRECISION = of the customers we flag,
#     how many really churn. RECALL = of all real churners, how many we flag. A random model scores about 0.20 here
#     (the churn rate); LightGBM scores 0.716.
#   classification_report = a text table of precision, recall and F1 per class (Retained / Churned).
#   confusion_matrix = a 2x2 count table: actual (rows) vs predicted (columns).
#   roc_curve = the points of the ROC curve (false positive rate vs true positive rate at every threshold).
from sklearn.metrics          import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve,
)
# Pipeline = chain steps into one object: "preprocess -> model". fit() and predict() then run both steps in order,
# so the same cleaning is always applied to train, test and new customers.
from sklearn.pipeline         import Pipeline
# ColumnTransformer = apply different preprocessing to different columns (scaler on numbers, one-hot on text).
from sklearn.compose          import ColumnTransformer
# LightGBM = a fast GRADIENT BOOSTING library from Microsoft. Gradient boosting = build trees one after another,
# where each new small tree tries to fix the mistakes the previous trees still make. LGBMClassifier is the
# scikit-learn-style version, so it fits into a Pipeline. This is the model that wins here.
from lightgbm                 import LGBMClassifier

# Hide all warning messages (for example library "future change" notices) so the console output stays clean.
# Note: this hides ALL warnings, including useful ones.
warnings.filterwarnings("ignore")

# SEED = a fixed random number. Every random step (split, folds, trees) uses it, so every run gives the same
# result. 42 is just a common convention; any fixed number works.
SEED = 42

# -- Paths ----------------------------------------------------------------------
# Where things live, relative to the folder you run the script from (run it from the project folder).
DATA_PATH    = "data/Churn_Modelling.csv"
MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
# Create the output folders if missing. exist_ok=True = no error if the folder is already there.
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# -- Business assumptions (used only for the threshold / ROI analysis) -----------
# These are stated assumptions, not facts from the data. Change them to match a
# real bank's economics and re-run.
# All three numbers were chosen by hand as placeholders (README says so). Not measured, not tuned.
# Example: contact 1 real churner -> expected gain 0.30 x 10,000 = INR 3,000, cost INR 500 -> net INR 2,500.
#          contact 1 customer who would have stayed -> net -INR 500 (wasted call).
CONTACT_COST      = 500      # cost of one retention offer + outreach (INR)
CUSTOMER_VALUE    = 10_000   # value of keeping one customer who would have churned (INR)
SAVE_RATE         = 0.30     # share of contacted true churners the offer actually retains

# -- Palette --------------------------------------------------------------------
# Chart colours as hex codes: red for churn, teal for retained, dark navy background, light grey text.
# Same colours as the Streamlit dashboard theme, so charts and dashboard look like one product.
PALETTE   = {"churn": "#E63946", "retain": "#2A9D8F"}
BG_COLOR  = "#0F172A"
TEXT_CLR  = "#E2E8F0"
GRID_CLR  = "#1E293B"

# IN: one chart area (ax) + optional title and axis labels  ->  OUT: nothing; the chart is restyled in place.
# WHY: every chart needs the same dark look; one helper avoids repeating 10 lines for each of the 9 charts.
# Example: style_axes(axes[0,0], "Churn Distribution", "", "Count") -> dark background, grey grid, light text.
def style_axes(ax, title="", xlabel="", ylabel=""):
    # Dark background and light tick labels.
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_CLR)
    # Loop over the 4 border lines (spines: top, bottom, left, right) and colour them grid-grey.
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    # Light colour for axis labels and title text.
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    # Thin grid lines (0.5 width, chosen by hand), drawn behind the bars (set_axisbelow).
    ax.grid(color=GRID_CLR, linewidth=0.5)
    ax.set_axisbelow(True)
    # Each label is set only if the caller passed one (an empty string "" counts as False and is skipped).
    # Font sizes 13 / 10 and pad 10 are chosen by hand for readability.
    if title:   ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    # x-axis label, only if given.
    if xlabel:  ax.set_xlabel(xlabel, fontsize=10)
    # y-axis label, only if given.
    if ylabel:  ax.set_ylabel(ylabel, fontsize=10)

# ===============================================================================
# 1.  LOAD DATA
# ===============================================================================
# Print a banner so the person running the script sees where the run starts. "="*60 = a line of 60 "=" signs.
print("\n" + "="*60)
print("  CUSTOMER CHURN PREDICTION - ML PIPELINE")
print("="*60)

# Read the CSV file into a DataFrame: 10,000 rows (customers) x 14 columns.
raw = pd.read_csv(DATA_PATH)
# Rename columns to lower_snake_case (CreditScore -> credit_score) so the code reads easily. Exited -> churn,
# because that column IS the answer we want to predict (the TARGET).
# Then drop RowNumber (just 1, 2, 3...) and Surname (a name like "Hargrave"): they cannot predict churn,
# and a model could wrongly memorise them.
df = raw.rename(columns={
    "CustomerId": "customer_id", "CreditScore": "credit_score", "Geography": "geography",
    "Gender": "gender", "Age": "age", "Tenure": "tenure_years", "Balance": "balance",
    "NumOfProducts": "num_products", "HasCrCard": "has_credit_card",
    "IsActiveMember": "is_active_member", "EstimatedSalary": "estimated_salary",
    "Exited": "churn",
}).drop(columns=["RowNumber", "Surname"])   # row index and surname carry no signal

# Data checks. assert = "stop the program with this message if the condition is False".
# 1) Every customer_id appears once (no duplicate customers). 2) No empty cells anywhere
#    (isnull().sum().sum() = count of missing values in the whole table; must be 0).
assert df["customer_id"].is_unique, "duplicate customers in source data"
assert df.isnull().sum().sum() == 0, "unexpected missing values"
# Prints "Loaded 10,000 real customers with 12 columns." ({:,} adds the thousands comma).
print(f"\nLoaded {len(df):,} real customers with {df.shape[1]} columns.")

# ===============================================================================
# 2.  EXPLORATORY DATA ANALYSIS
# ===============================================================================
# EDA (Exploratory Data Analysis) = look at the data with simple counts and charts BEFORE modelling,
# to find patterns, like "customers with 3-4 products churn far more".
print("\nRunning EDA ...")

# IN: a column name (or a 0/1 Series)  ->  OUT: a small table: churn rate (%) and customer count per group.
# WHY: "churn rate by group" is the basic EDA question; one helper serves 5 groupings.
# Example: churn_by("geography") -> France 16.2% (n=5014), Germany 32.4% (n=2509), Spain 16.7% (n=2477).
def churn_by(col):
    # groupby = split rows into groups by the column's value. The mean of a 0/1 churn column = churn share
    # (e.g. 0.324), and count = group size. Rename to churn_rate / n, then turn 0.324 into 32.4 (percent, 1 decimal).
    return (df.groupby(col)["churn"].agg(["mean", "count"])
              .rename(columns={"mean": "churn_rate", "count": "n"})
              .assign(churn_rate=lambda t: (t["churn_rate"]*100).round(1)))

# Build the text of reports/eda_summary.txt as a list of lines: totals, churn rate (20.4%), retained (7,963),
# churned (2,037), describe() = count/mean/std/min/quartiles/max of every number column, then 5 churn_by tables.
# The last one groups by "balance is exactly 0" (True/False): zero balance churns at 13.8% vs 24.1%.
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
# Write the lines to the text file. "w" = overwrite the old file. with = close the file automatically.
with open(f"{REPORTS_DIR}/eda_summary.txt", "w") as f:
    f.write("\n".join(eda_lines))
print(f"   -> EDA saved to {REPORTS_DIR}/eda_summary.txt")

# -- EDA Plots ------------------------------------------------------------------
# One figure with a grid of 2 rows x 3 columns = 6 charts. figsize 18x10 inches, chosen by hand to fit 6 charts.
# axes[row, col] picks one chart slot, e.g. axes[0,0] = top-left.
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor(BG_COLOR)
# Main title above all 6 charts. y=1.01 puts it just above the top edge so it does not overlap the charts.
fig.suptitle("Customer Churn - Exploratory Data Analysis",
             color=TEXT_CLR, fontsize=16, fontweight="bold", y=1.01)

# IN: chart slot, column to group by, chart title, optional {value: label} names  ->  OUT: a churn-rate bar chart.
# WHY: 4 of the 6 EDA charts are "churn rate per group"; one helper draws them all the same way.
# Example: rate_bar(ax, "is_active_member", "...", {0: "Inactive", 1: "Active"}) -> bars "Inactive" 26.9% (red,
#   above the 20.4% average) and "Active" 14.3% (teal, below average).
def rate_bar(ax, col, title, labels=None):
    # Churn rate in percent per group (mean of 0/1 churn x 100).
    cr = df.groupby(col)["churn"].mean().mul(100)
    # Bar names: use the friendly label if a labels dict was given, else the raw value as text.
    # Bar colour: red if this group churns MORE than the overall average (20.4%), teal if not.
    ax.bar([labels.get(i, i) if labels else str(i) for i in cr.index], cr.values,
           color=["#E63946" if v > df["churn"].mean()*100 else "#2A9D8F" for v in cr.values],
           edgecolor=BG_COLOR)
    style_axes(ax, title, "", "Churn Rate (%)")
    # Show y-axis numbers with a "%" sign.
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())

# Chart 1 (top-left): how many customers stayed vs left. value_counts counts 0s and 1s; sort_index puts 0
# (Retained) first so the bar labels match. Real counts: 7,963 retained, 2,037 churned.
# This shows CLASS IMBALANCE = one class (stayers) is about 4x bigger than the other (churners).
churn_counts = df["churn"].value_counts().sort_index()
axes[0,0].bar(["Retained", "Churned"], churn_counts.values,
              color=[PALETTE["retain"], PALETTE["churn"]], edgecolor=BG_COLOR, linewidth=1.5)
style_axes(axes[0,0], "Churn Distribution", "", "Count")

# Chart 2 (top-middle): age histograms, one for stayers and one for churners, drawn on top of each other.
# Loop runs twice: label 0 (retained group) and label 1 (churned group).
# A HISTOGRAM counts how many customers fall in each age band. bins=25 bands, alpha=0.7 = 70% opaque so the
# overlap stays visible (both chosen by hand). Colour/label: label 1 is truthy -> red "Churned", 0 -> teal "Retained".
for label, grp in df.groupby("churn"):
    axes[0,1].hist(grp["age"], bins=25, alpha=0.7,
                   color=PALETTE["churn"] if label else PALETTE["retain"],
                   label="Churned" if label else "Retained", edgecolor=BG_COLOR)
axes[0,1].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR)
style_axes(axes[0,1], "Age Distribution by Churn", "Age", "Count")

# Charts 3-6: churn rate by Geography (Germany 32.4% stands out), by number of products (3 -> 82.7%,
# 4 -> 100%), by active/inactive, and by zero balance vs positive balance.
rate_bar(axes[0,2], "geography", "Churn Rate by Geography")
rate_bar(axes[1,0], "num_products", "Churn Rate by Number of Products")
rate_bar(axes[1,1], "is_active_member", "Churn Rate by Activity", {0: "Inactive", 1: "Active"})
rate_bar(axes[1,2], df["balance"].eq(0).rename("zb"), "Churn Rate by Balance",
         {False: "Balance > 0", True: "Zero balance"})

# tight_layout = auto-fix spacing so titles and labels do not overlap. Save as PNG at 150 dpi (dots per inch:
# sharp enough for a README, chosen by hand). bbox_inches="tight" trims empty border. close() frees memory.
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/eda_plots.png", dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR)
plt.close()
print(f"   -> EDA plots saved to {REPORTS_DIR}/eda_plots.png")

# ===============================================================================
# 3.  FEATURE ENGINEERING
# ===============================================================================
# FEATURE = one input column the model learns from (age, balance...). FEATURE ENGINEERING = making new, more
# useful columns out of the raw ones, using business sense.
print("\nEngineering features ...")

# IN: a customer DataFrame  ->  OUT: a copy with 5 new columns added.
# WHY: give the model ready-made signals that EDA suggested (zero balance, older age, inactive with many products).
# Example (row 1, CustomerId 15634602: Balance 0, EstimatedSalary 101348.88, NumOfProducts 1, Tenure 2, Age 42, active):
#   zero_balance 1, balance_to_salary 0.0, products_per_tenure 1/(2+1) = 0.33, inactive_multi_prod 0, senior 0.
def add_features(d):
    # Work on a copy so the caller's table is not changed by surprise.
    d = d.copy()
    # 1 if balance is exactly 0, else 0. (36% of customers have a zero balance, per README.)
    d["zero_balance"]         = d["balance"].eq(0).astype(int)
    # Balance divided by salary. clip(lower=1) turns any salary below 1 into 1, so we never divide by ~0
    # (the smallest salary in the data is 11.58, per eda_summary.txt).
    d["balance_to_salary"]    = d["balance"] / d["estimated_salary"].clip(lower=1)
    # Products per year with the bank. +1 because tenure can be 0 years, and dividing by 0 is not allowed.
    d["products_per_tenure"]  = d["num_products"] / (d["tenure_years"] + 1)
    # 1 if the customer is NOT active AND holds 2 or more products (a possibly unhappy cross-sold customer).
    d["inactive_multi_prod"]  = ((d["is_active_member"] == 0) & (d["num_products"] >= 2)).astype(int)
    # 1 if age >= 50. The cut-off 50 comes from EDA: age 50+ churns 45% vs 16% under 50 (README).
    d["senior"]               = d["age"].ge(50).astype(int)
    return d

df = add_features(df)

# TARGET = the column to predict (churn: 1 = left). CAT_FEATS = CATEGORICAL features = text categories
# (geography, gender). NUM_FEATS = every other column except customer_id and the target: 13 number columns
# (credit_score, age, tenure_years, balance, num_products, has_credit_card, is_active_member, estimated_salary
# + the 5 new ones).
TARGET      = "churn"
CAT_FEATS   = ["geography", "gender"]
NUM_FEATS   = [c for c in df.columns if c not in ["customer_id", TARGET] + CAT_FEATS]

# X = the inputs (all columns except the ID and the answer). y = the answer column (0/1).
# customer_id is dropped from X because an ID number cannot cause churn; the model would only memorise it.
X = df.drop(columns=["customer_id", TARGET])
y = df[TARGET]

# Preprocessing recipe: scale the 13 number columns, one-hot encode the 2 text columns.
# drop="first" = drop the first category of each text column (France, Female), because it is implied when the
# others are 0. Result: geography_Germany, geography_Spain, gender_Male. sparse_output=False = return a normal
# table, not a compressed (sparse) one. remainder="drop" = any column not listed is thrown away.
# Total columns going into the model: 13 + 3 = 16.
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),                  NUM_FEATS),
    ("cat", OneHotEncoder(drop="first",
                          sparse_output=False), CAT_FEATS),
], remainder="drop")

# TRAIN/TEST SPLIT: 80% train (8,000 customers), 20% test (2,000 customers). test_size=0.20 is a common default.
# random_state=SEED = the same split every run. stratify=y = both parts keep the same ~20% churn share
# (the test set has 407 churners out of 2,000). The test set is not touched again until final scoring.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"   Class balance (train) - 0:{(y_train==0).sum()} | 1:{(y_train==1).sum()}")

# ===============================================================================
# 4.  MODEL TRAINING
# ===============================================================================
print("\nTraining models ...")

# Three candidate models, each wrapped in a Pipeline "pre (preprocessor) -> clf (classifier)".
# A CLASSIFIER predicts a class (churn yes/no) and also a probability-like score.
# Note: all three Pipelines share the SAME preprocessor object. Each fit() re-learns it on the same X_train,
# so the result is the same every time and nothing leaks.
# CLASS IMBALANCE (20% churners vs 80% stayers) is handled with class weights: mistakes on the rare churners
# count more during training, so the model does not just predict "stays" for everyone.
# None of the hyperparameters below were searched/tuned in this repo; they were set by hand.
# (HYPERPARAMETER = a setting you choose before training, like the number of trees; not learned from data.)
models = {
    # Baseline. max_iter=1000 = allow up to 1000 solver steps so it finishes (converges).
    # C=0.5 = REGULARISATION strength (smaller C = stronger penalty on big weights = simpler model, less
    # OVERFITTING; overfitting = memorising the training rows instead of learning patterns that generalise).
    # class_weight="balanced" = weight each class by 1 / its frequency (churners count about 4x more).
    "Logistic Regression": Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(
            max_iter=1000, C=0.5, class_weight="balanced", random_state=SEED
        ))
    ]),
    # Random Forest: n_estimators=300 trees. max_depth=10 = each tree asks at most 10 questions in a row.
    # min_samples_leaf=5 = every final leaf must hold at least 5 customers. Both limits fight overfitting.
    # n_jobs=-1 = use all CPU cores to train in parallel.
    "Random Forest": Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ))
    ]),
    # LightGBM (gradient boosting):
    #   n_estimators=400 = 400 small trees built one after another.
    #   learning_rate=0.03 = each new tree corrects only 3% of the remaining error; small steps + many trees = steadier.
    #   num_leaves=15 = each tree has at most 15 end points (LightGBM default is 31), so trees stay small.
    #   min_child_samples=30 = a leaf needs at least 30 customers (default 20), so it cannot fit tiny groups.
    #   subsample=0.8 + subsample_freq=1 = each tree sees a random 80% of the rows (re-drawn every tree).
    #   colsample_bytree=0.8 = each tree sees a random 80% of the 16 columns. Randomness reduces overfitting.
    #   is_unbalance=True = LightGBM's built-in fix for class imbalance (up-weights the rare churners).
    #   importance_type="gain" = feature importance = total error reduction a feature gives (not split counts).
    #   verbose=-1 = no training log messages.
    "LightGBM": Pipeline([
        ("pre", preprocessor),
        ("clf", LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            is_unbalance=True, importance_type="gain", random_state=SEED, verbose=-1
        ))
    ]),
}

# results = one entry per model with its fitted pipeline, test scores and CV scores.
# cv = the 5-fold stratified splitter. shuffle=True mixes rows before cutting; random_state fixes the mix, so the
# same 5 folds are reused later for the threshold. 5 folds is a common default: each fold = 1,600 of the 8,000
# training rows, so each round trains on 6,400 rows and scores the other 1,600.
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Loop over the 3 models. For each: CV on train -> fit on all of train -> score the test set -> save to disk.
for name, pipe in models.items():
    # 5 AUC scores, one per fold, using ONLY the training data. n_jobs=-1 = all CPU cores.
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                cv=cv, scoring="roc_auc", n_jobs=-1)
    # Train on all 8,000 training customers.
    pipe.fit(X_train, y_train)
    # predict_proba gives 2 columns per customer: [score of class 0, score of class 1]. [:, 1] = churn score.
    y_prob  = pipe.predict_proba(X_test)[:, 1]
    # Yes/no prediction with the default THRESHOLD 0.5: score >= 0.5 -> predict "churn" (1).
    # A THRESHOLD is the cut-off score above which we act on a customer. Used only for the classification report.
    y_pred  = (y_prob >= 0.5).astype(int)
    # Store everything for this model. auc = test ROC-AUC, pr_auc = test PR-AUC, cv_mean / cv_std = the
    # average and spread of the 5 CV AUCs. Real LightGBM values: 0.8642, 0.7159, 0.8631, 0.0105.
    results[name] = {
        "pipe":     pipe,
        "y_prob":   y_prob,
        "y_pred":   y_pred,
        "auc":      roc_auc_score(y_test, y_prob),
        "pr_auc":   average_precision_score(y_test, y_prob),
        "cv_mean":  cv_scores.mean(),
        "cv_std":   cv_scores.std(),
    }
    # Save the trained pipeline: "LightGBM" -> models/lightgbm.pkl, "Random Forest" -> models/random_forest.pkl.
    joblib.dump(pipe, f"{MODELS_DIR}/{name.lower().replace(' ','_')}.pkl")
    # One summary line per model. {name:<22} = pad the name to 22 characters so the columns line up.
    print(f"   {name:<22}  Test AUC={results[name]['auc']:.4f}  "
          f"PR-AUC={results[name]['pr_auc']:.4f}  "
          f"CV AUC={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Model selection uses cross-validated AUC on the training set only, so the
# test set stays untouched until final reporting.
# max(..., key=...) = the model name with the highest CV mean AUC. Real result: LightGBM 0.8631 beats
# Random Forest 0.8594 (their test AUCs tie at 0.864). BASE = the simple baseline kept for comparison.
BEST = max(results, key=lambda n: results[n]["cv_mean"])
BASE = "Logistic Regression"
print(f"\n   Selected model (by CV AUC): {BEST}")

# ===============================================================================
# 5.  BUSINESS EVALUATION
# ===============================================================================
# Turn model scores into business answers: "if we call the riskiest customers, how many churners do we reach,
# and how much money does that make?"
print("\nBusiness evaluation ...")
# best_prob = LightGBM's churn score for each of the 2,000 test customers. y_true = their real 0/1 outcomes
# as a numpy array (.values), so we can index it by position.
best_prob = results[BEST]["y_prob"]
y_true    = y_test.values

# IN: scores + a fraction like 0.10  ->  OUT: share of ALL test churners found in the top `frac` riskiest customers.
# WHY: a retention team works down a list from the top; this is "how good is the top of the list".
# Example: capture_at(best_prob, 0.10) -> top 200 of 2,000 customers contain 41.28% of the 407 churners.
def capture_at(prob, frac):
    """Share of all churners found by contacting the top `frac` riskiest customers."""
    # k = how many customers to take: 2000 x 0.10 = 200.
    k = int(round(len(prob) * frac))
    # argsort(-prob) = positions sorted from highest score to lowest (minus sign flips the order). Keep the first k.
    top = np.argsort(-prob)[:k]
    # Churners inside the top k, divided by all churners. Note: it reads y_true (the test set) from outside the function.
    return y_true[top].sum() / y_true.sum()

# Real values: top 10% -> 41.28%, top 20% -> 62.9% of churners.
# LIFT = how many times better than random. Random top 10% would catch 10% of churners, so lift = 0.4128 / 0.10 = 4.13x.
top10_capture = capture_at(best_prob, 0.10)
top20_capture = capture_at(best_prob, 0.20)
lift_top10    = top10_capture / 0.10

# IN: scores, threshold t, true 0/1 outcomes  ->  OUT: (net value in INR, customers contacted, churners reached).
# WHY: puts a money value on every possible threshold, so we pick the threshold by profit, not by accuracy.
# Formula: net = churners_reached x SAVE_RATE x CUSTOMER_VALUE - contacted x CONTACT_COST.
# Example (test set, t=0.38): 328 x 0.30 x 10,000 - 751 x 500 = 984,000 - 375,500 = INR 608,500 (6.08 lakh).
def net_value(prob, t, y):
    # True/False per customer: do we contact them (score >= t)?
    contact = prob >= t
    # tp = TRUE POSITIVES = contacted AND really churned.
    tp = (contact & (y == 1)).sum()
    return tp * SAVE_RATE * CUSTOMER_VALUE - contact.sum() * CONTACT_COST, int(contact.sum()), int(tp)

# The threshold is chosen on out-of-fold predictions for the TRAINING set, so the
# test set is only used to report the result, never to pick it.
# clone(models[BEST]) = a fresh untrained LightGBM pipeline. cross_val_predict trains it 5 times (same 5 folds)
# and gives each of the 8,000 training customers a score from the model that did NOT train on them.
# method="predict_proba" + [:, 1] = the churn score column.
oof_prob = cross_val_predict(clone(models[BEST]), X_train, y_train, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]
# Candidate thresholds 0.05, 0.06, ..., 0.95 (91 values, step 0.01; the end 0.96 is not included).
# np.round(..., 2) removes float noise like 0.38000000000000006.
thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
# For every threshold: (t, net value, contacted, reached) on the TRAINING out-of-fold scores.
# The * unpacks the 3 returned values into the same tuple.
train_curve = [(t, *net_value(oof_prob, t, y_train.values)) for t in thresholds]
# Pick the threshold with the highest training net value (r[1] = net value; [0] = its t). Real result: 0.38.
best_t = max(train_curve, key=lambda r: r[1])[0]

# Same curve on the TEST set, only for the chart (never used to choose t).
curve = [(t, *net_value(best_prob, t, y_true)) for t in thresholds]   # test set, for reporting
# Test-set results at the chosen 0.38: INR 608,500, 751 contacted, 328 churners reached.
best_net, best_n, best_tp = net_value(best_prob, best_t, y_true)
# Test-set result at the default 0.50, for comparison. Real value: INR 626,000, slightly MORE than at 0.38.
# Honest reading (README): the test curve is nearly flat between 0.3 and 0.6, so the gain comes from targeting
# at all, not from the exact cut-off; 0.38 was picked on training data, so its test number is the fair one to quote.
net_at_05, n_at_05, _ = net_value(best_prob, 0.5, y_true)
# "Call everyone" option: all 407 test churners reached, all 2,000 customers contacted.
# 407 x 3,000 - 2,000 x 500 = 1,221,000 - 1,000,000 = INR 221,000 (2.21 lakh).
contact_all_net = y_true.sum() * SAVE_RATE * CUSTOMER_VALUE - len(y_true) * CONTACT_COST

# Print the business summary. {:.1%} shows 0.4128 as "41.3%"; {:,.0f} shows 608500 as "608,500".
print(f"   Top-10% riskiest customers contain {top10_capture:.1%} of churners "
      f"(lift {lift_top10:.1f}x vs random)")
print(f"   Profit-optimal threshold (chosen on training folds): {best_t:.2f} -> test set: contact {best_n} customers, "
      f"net value INR {best_net:,.0f} (vs INR {net_at_05:,.0f} at 0.50, "
      f"INR {contact_all_net:,.0f} if everyone is contacted)")

# ===============================================================================
# 6.  EVALUATION PLOTS
# ===============================================================================
print("\nGenerating evaluation charts ...")

# One figure, 1 row x 3 charts: ROC curves, confusion matrix, net value vs threshold. Size chosen by hand.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle("Model Evaluation (held-out test set)", color=TEXT_CLR, fontsize=15, fontweight="bold")

# One line colour per model on the ROC chart.
colors = {"Logistic Regression": "#F4A261", "Random Forest": "#457B9D", "LightGBM": "#2A9D8F"}

# Chart 1: ROC curve for each model. ROC = for every threshold, plot the FALSE POSITIVE RATE (share of stayers
# wrongly flagged) on x against the TRUE POSITIVE RATE (= recall, share of churners caught) on y.
# The area under this curve is the AUC shown in the legend.
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})",
                 color=colors[name], linewidth=2.5)
# Dashed diagonal = a random model (AUC 0.5). A good model's curve bends far above it.
axes[0].plot([0,1],[0,1],"--", color="#475569", linewidth=1)
axes[0].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
style_axes(axes[0], "ROC Curve", "False Positive Rate", "True Positive Rate")

# Chart 2: CONFUSION MATRIX of LightGBM at the chosen threshold (0.38) on the test set.
# Rows = actual (Retained, Churned), columns = predicted. E.g. bottom-right = churners we would contact (328).
# annot=True writes the counts in the cells; fmt="d" = whole numbers; cmap="Blues" = blue colour scale.
cm = confusion_matrix(y_test, (best_prob >= best_t).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            ax=axes[1], linewidths=0.5, linecolor=BG_COLOR,
            cbar_kws={"shrink": 0.8})
# Style the heatmap by hand (style_axes is not used here because the heatmap has its own grid and colour bar).
axes[1].set_facecolor(BG_COLOR)
axes[1].set_title(f"Confusion Matrix - {BEST} @ t={best_t:.2f}", color=TEXT_CLR,
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel("Predicted", color=TEXT_CLR)
axes[1].set_ylabel("Actual",    color=TEXT_CLR)
axes[1].tick_params(colors=TEXT_CLR)
# Replace the default 0/1 tick labels with words. rotation=0 keeps the y labels horizontal.
axes[1].set_xticklabels(["Retained","Churned"], color=TEXT_CLR)
axes[1].set_yticklabels(["Retained","Churned"], color=TEXT_CLR, rotation=0)

# Chart 3: test-set net value for every threshold. c[0] = threshold, c[1] = net value; / 1e5 converts INR to lakh
# (1 lakh = 100,000). Red dashed line = the threshold chosen on training data (0.38); grey dotted line = default 0.50.
axes[2].plot([c[0] for c in curve], [c[1]/1e5 for c in curve], color="#2A9D8F", linewidth=2.5)
axes[2].axvline(best_t, color="#E63946", linestyle="--", linewidth=1.5,
                label=f"chosen on train t={best_t:.2f}")
axes[2].axvline(0.5, color="#475569", linestyle=":", linewidth=1.5, label="default t=0.50")
axes[2].legend(facecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=9)
style_axes(axes[2], "Retention Campaign Net Value vs Threshold (test set)",
           "Churn risk-score threshold", "Net value (INR lakh)")

# Save as reports/model_evaluation.png (same settings as the EDA chart) and free memory.
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/model_evaluation.png", dpi=150,
            bbox_inches="tight", facecolor=BG_COLOR)
plt.close()
print(f"   -> {REPORTS_DIR}/model_evaluation.png")

# ===============================================================================
# 7.  FEATURE IMPORTANCE  (selected model)
# ===============================================================================
# FEATURE IMPORTANCE = how much each input column helped the model. Real top 5 (gain-based, metrics.json):
# age 0.29, num_products 0.20, balance 0.10, credit_score 0.07, balance_to_salary 0.07.
# named_steps["pre"] = the fitted preprocessor inside the best pipeline.
best_pipe = results[BEST]["pipe"]
pre_step  = best_pipe.named_steps["pre"]
# Names of the one-hot columns: ["geography_Germany", "geography_Spain", "gender_Male"].
cat_names = list(pre_step.named_transformers_["cat"].get_feature_names_out(CAT_FEATS))
# Full list of the 16 model inputs, in the same order the ColumnTransformer outputs them (numbers first, then one-hot).
feat_names = NUM_FEATS + cat_names

# Raw importances from the trained model (gain for LightGBM; Random Forest also has feature_importances_).
# Divide by the total so they add up to 1 (shares). Sort smallest first, so the biggest bar ends up at the top
# of a horizontal bar chart.
imp = best_pipe.named_steps["clf"].feature_importances_
fi_df = (pd.DataFrame({"feature": feat_names, "importance": imp / imp.sum()})
           .sort_values("importance", ascending=True))

# Draw the horizontal bar chart (10 x 7 inches, chosen by hand).
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)
# Bars above the median importance in red, the rest in blue, so the top half stands out.
bar_colors = [PALETTE["churn"] if v > fi_df["importance"].median()
              else "#457B9D" for v in fi_df["importance"]]
ax.barh(fi_df["feature"], fi_df["importance"], color=bar_colors, edgecolor=BG_COLOR)
# Colour the 4 border lines grid-grey (same as in style_axes).
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
# Build reports/classification_reports.txt: for each model, its AUCs and the precision/recall table at the default
# threshold 0.50. Real LightGBM churn row at 0.50: precision 0.53, recall 0.75 (it flags 75% of churners, and
# 53% of the flagged customers really churn).
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

# All key numbers in one dict -> reports/metrics.json. The dashboard and report_generator.py read this file,
# so no number is typed by hand anywhere else. round(..., 4) keeps 4 decimals. float()/int() turn numpy numbers
# into plain Python numbers, because json cannot save numpy types.
metrics = {
    # Dataset facts: 10,000 rows, churn rate 0.2037, 2,000 test rows.
    "dataset": {"name": "Bank Customer Churn (Churn_Modelling.csv)", "rows": len(df),
                "churn_rate": round(float(df["churn"].mean()), 4), "test_rows": len(y_test)},
    # Per-model scores (a dict comprehension: one entry per model name).
    "models": {n: {"test_auc": round(r["auc"], 4), "pr_auc": round(r["pr_auc"], 4),
                   "cv_auc_mean": round(r["cv_mean"], 4), "cv_auc_std": round(r["cv_std"], 4)}
               for n, r in results.items()},
    "best_model": BEST,
    "baseline_model": BASE,
    # Business numbers, all on the test set except the threshold choice itself.
    # mean_risk_score_test = average LightGBM score (0.3506) - higher than the real 20% churn rate, because class
    # weighting pushes scores up. That is why the project calls them RISK SCORES, not calibrated probabilities.
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
    # Feature importance list, biggest first, as [{"feature": "age", "importance": 0.2917}, ...].
    "feature_importance": fi_df.sort_values("importance", ascending=False)
                               .round(4).to_dict(orient="records"),
}
# indent=2 = pretty-print the JSON so a human can read it.
with open(f"{REPORTS_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Scored test set -> data/scored_customers.csv (the dashboard's data). Start from the 2,000 test rows.
test_out = X_test.copy()
# Put customer_id back as the first column. X_test kept the original row numbers (index), so df.loc[...] finds
# the matching IDs.
test_out.insert(0, "customer_id", df.loc[test_out.index, "customer_id"])
# Add: the real outcome, the LightGBM score, the Logistic Regression score (for comparison in the dashboard),
# and contact = 1 if the LightGBM score >= 0.38 (this customer is in the retention campaign).
test_out["actual_churn"]    = y_test.values
test_out["churn_prob"]      = best_prob
test_out["churn_prob_lr"]   = results[BASE]["y_prob"]
test_out["contact"]         = (best_prob >= best_t).astype(int)
# Risk segment by score: (-0.01, 0.30] Low, (0.30, 0.60] Medium, (0.60, 1.0] High. Each bin includes its right edge.
# The lower edge is -0.01 (not 0) so a score of exactly 0 still falls in "Low Risk". 0.30/0.60 are chosen by hand.
# Real check (README): High Risk 462 customers, 60.0% really churned; Medium 459, 16.1%; Low 1,079, 5.2%.
test_out["risk_segment"]    = pd.cut(best_prob, bins=[-0.01, 0.30, 0.60, 1.0],
                                     labels=["Low Risk", "Medium Risk", "High Risk"])
# index=False = do not write pandas' row numbers as an extra column.
test_out.to_csv("data/scored_customers.csv", index=False)

# Final banner: one "AUC" line per model, then where the outputs were saved.
print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("="*60)
for name, r in results.items():
    print(f"  {name:<22} -> AUC {r['auc']:.4f}")
print(f"\n  Artifacts saved to models/, reports/ and data/scored_customers.csv")
print("="*60 + "\n")
