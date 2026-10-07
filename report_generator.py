"""
report_generator.py
--------------------
Turns model outputs into an executive briefing a retention manager can act on.

Every number in the report is computed from the pipeline outputs; the prose
around them is a fixed template. Nothing is hard-coded or invented.

This module reads:
  - data/Churn_Modelling.csv      (full portfolio, for segment churn rates)
  - data/scored_customers.csv     (held-out test set with churn risk scores)
  - reports/metrics.json          (model scores and campaign economics)

And produces:
  - reports/executive_report.md   — markdown (renders on GitHub)
  - reports/executive_report.txt  — plain-text version

Run:
    python report_generator.py
"""

# WHAT THIS FILE IS: the "report writer". It reads the outputs of churn_model.py and writes a one-page briefing
# for a bank manager who will never open Python. No model is trained here.
# Analogy: a fill-in-the-blanks letter. The sentences are fixed; every blank (a percentage, a count, a rupee value)
# is filled from the data files, so no number is ever typed by hand.
# Real example output (reports/executive_report.md): "20.4% of customers churned", "test AUC of 0.864 (5-fold CV
# 0.863 ± 0.011), versus 0.785 for a Logistic Regression baseline", "the riskiest 10% ... reaches 41% of all
# churners — 4.1x better than ... random", "Age 50+ | 45.4% | 16.3% for under 50 (2.8x)",
# "Contact customers with a churn risk score ≥ 0.38 ... 751 of 2,000 test customers, reaching 328 actual churners".
# Overall flow: check files exist -> load CSV + scored CSV + metrics.json -> compute segment churn rates
#   -> fill the Markdown template -> save .md -> strip Markdown symbols -> save .txt -> print the report
#
# json = read reports/metrics.json into a Python dict.
import json
# os = check that input files exist and create the reports folder.
import os
# re = REGULAR EXPRESSIONS: text patterns. Used to strip Markdown symbols like **bold** for the plain-text copy.
import re
# datetime = today's date, printed at the top of the report ("Generated 26 Sep 2026 ...").
from datetime import datetime

# pandas (pd) = tables (DataFrames). Used to read the CSVs and compute churn rates per group.
import pandas as pd

# Output folder. Create it if missing; exist_ok=True = no error if it already exists.
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Load data ───────────────────────────────────────────────────────────────────
# IN: nothing  ->  OUT: (full 10,000-row table, 2,000-row scored test table, metrics dict).
# WHY: the report needs churn rates for the WHOLE bank (full), the test-set segment check (scored), and the model
# and money numbers (metrics).
# Only the 7 columns the report uses are renamed to lower_snake_case (Exited -> churn, Age -> age ...);
# other columns like CreditScore keep their original names because they are not used here.
# Example: full.loc[0] = customer 15634602, age 42, France, Female, churn 1.
def load_data():
    full = pd.read_csv("data/Churn_Modelling.csv").rename(columns={
        "Geography": "geography", "Gender": "gender", "Age": "age",
        "NumOfProducts": "num_products", "IsActiveMember": "is_active_member",
        "Balance": "balance", "Exited": "churn",
    })
    scored = pd.read_csv("data/scored_customers.csv")
    # with = open the file and close it automatically. json.load = JSON text -> dict.
    with open(f"{REPORTS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    return full, scored, metrics


# IN: a table + a True/False mask (one True/False per row)  ->  OUT: churn rate among the rows where the mask is True.
# WHY: the report compares many groups ("age 50+" vs "under 50"); this is the one-line helper for each rate.
# Example: rate(full, full["age"] >= 50) -> 0.454 (45.4% of customers aged 50+ churned).
def rate(d, mask):
    # The mean of a 0/1 column = the share of 1s = the churn rate.
    return d.loc[mask, "churn"].mean()


# ── Analytics engine ────────────────────────────────────────────────────────────
# IN: full table, scored test table, metrics dict  ->  OUT: one dict "ins" (insights) with every number the report needs.
# WHY: keep all the maths in one place, so build_markdown only fills in text.
# Example: ins["overall"] = 0.2037, ins["geo"] = Germany 0.324, Spain 0.167, France 0.162,
#   ins["old_inactive"] = mean 0.82, count 484.
def compute_insights(full, scored, metrics):
    # Overall churn rate of all 10,000 customers (0.2037 = 20.4%).
    overall = full["churn"].mean()
    # Churn rate per country, highest first (Germany 32.4% on top).
    geo = full.groupby("geography")["churn"].mean().sort_values(ascending=False)
    # Churn rate and count per number of products held (1, 2, 3, 4).
    prod = full.groupby("num_products")["churn"].agg(["mean", "count"])
    # On the TEST set: real churn rate and count inside each predicted risk segment (High / Medium / Low).
    seg_actual = scored.groupby("risk_segment")["actual_churn"].agg(["mean", "count"])

    # Churn "drivers": each entry is (group name, churn rate inside the group, churn rate of the comparison group,
    # comparison label). The country entry is not hard-coded: geo.index[0] is whichever country churns most
    # (Germany here), compared with all other countries.
    drivers = [
        ("Age 50+",                    rate(full, full["age"] >= 50),
                                       rate(full, full["age"] < 50), "under 50"),
        ("Inactive members",           rate(full, full["is_active_member"] == 0),
                                       rate(full, full["is_active_member"] == 1), "active members"),
        (f"Customers in {geo.index[0]}", geo.iloc[0],
                                       rate(full, full["geography"] != geo.index[0]), "other countries"),
        ("Female customers",           rate(full, full["gender"] == "Female"),
                                       rate(full, full["gender"] == "Male"), "male customers"),
    ]
    # Sort by the ratio inside/outside (d[1] / d[2]), biggest first.
    # Real order: Age 50+ 2.8x, Germany 2.0x, Inactive 1.9x, Female 1.5x.
    drivers.sort(key=lambda d: d[1] / d[2], reverse=True)

    # Pack everything into one dict. Two extra groups for the campaign advice:
    #   old_inactive = customers aged 50+ AND inactive -> churn 82%, n=484.
    #   multi_prod   = customers with 3 or more products -> churn 86%, n=326.
    # .agg(["mean", "count"]) returns both the churn rate and the group size.
    return {
        "overall": overall,
        "n": len(full),
        "geo": geo,
        "prod": prod,
        "seg_actual": seg_actual,
        "drivers": drivers,
        "m": metrics,
        "old_inactive": full.loc[(full["age"] >= 50) & (full["is_active_member"] == 0), "churn"].agg(["mean", "count"]),
        "multi_prod": full.loc[full["num_products"] >= 3, "churn"].agg(["mean", "count"]),
    }


# ── Report builder ──────────────────────────────────────────────────────────────
# IN: the insights dict  ->  OUT: the full report as one Markdown text.
# MARKDOWN = a simple text format: "# Title" = heading, "**bold**", "| a | b |" = table rows. GitHub renders it nicely.
# WHY: a fixed template with the numbers dropped in, so the report always matches the latest model run.
# Example line: "- **20.4%** of customers churned."
def build_markdown(ins):
    # Short names: m = metrics, biz = its business block, best/base = "LightGBM" / "Logistic Regression",
    # bm/lm = their score dicts, a = the cost assumptions, prod = churn by products.
    m, biz = ins["m"], ins["m"]["business"]
    best, base = m["best_model"], m["baseline_model"]
    bm, lm = m["models"][best], m["models"][base]
    a = biz["assumptions"]
    prod = ins["prod"]
    # The 5 most important features as `code` text: `age`, `num_products`, `balance`, `credit_score`, `balance_to_salary`.
    top_feats = ", ".join(f"`{f['feature']}`" for f in m["feature_importance"][:5])

    # Sections 1-2 of the report. Number formats: {x:.1%} -> "20.4%", {x:.3f} -> "0.864", {x:.0%} -> "41%",
    # {x:,} -> "10,000". /1e5 converts rupees to lakh (608,500 -> 6.08).
    lines = [
        "# Customer Churn — Executive Briefing",
        f"*Generated {datetime.now():%d %b %Y} from {ins['n']:,} real bank customers "
        f"(public Bank Customer Churn dataset).*",
        "",
        "## 1. Headline",
        f"- **{ins['overall']:.1%}** of customers churned.",
        f"- A {best} model ranks customers by churn risk with a test AUC of **{bm['test_auc']:.3f}** "
        f"(5-fold CV {bm['cv_auc_mean']:.3f} ± {bm['cv_auc_std']:.3f}), versus {lm['test_auc']:.3f} for a "
        f"{base} baseline.",
        f"- Contacting only the riskiest **10%** of customers reaches **{biz['top10_capture']:.0%}** of all "
        f"churners — **{biz['lift_top10']:.1f}x** better than picking customers at random.",
        f"- Under the stated campaign assumptions, targeting by model score is worth "
        f"**₹{biz['net_value_optimal_inr']/1e5:.2f} lakh** per 2,000 customers, versus "
        f"₹{biz['net_value_contact_all_inr']/1e5:.2f} lakh for contacting everyone.",
        "",
        "## 2. Who churns",
        "| Segment | Churn rate | Compared with |",
        "|---|---|---|",
    ]
    # One table row per driver, e.g. "| Age 50+ | 45.4% | 16.3% for under 50 (2.8x) |".
    for name, r_in, r_out, other in ins["drivers"]:
        lines.append(f"| {name} | {r_in:.1%} | {r_out:.1%} for {other} ({r_in/r_out:.1f}x) |")
    # Products paragraph: one "k product(s) → v% churn (n=count)" item per row of prod. iterrows() gives
    # (k = number of products, (v = churn rate, c = count)). Adds "s" when k > 1.
    # Real text: "1 product → 28% churn (n=5,084), 2 products → 8% ..., 3 products → 83% ..., 4 products → 100% (n=60)".
    # Then the top-5 features sentence and the header of section 3.
    lines += [
        "",
        "**Products held is the sharpest signal:** "
        + ", ".join(f"{int(k)} product{'s' if k > 1 else ''} → {v:.0%} churn (n={int(c):,})"
                    for k, (v, c) in prod.iterrows())
        + ". Customers with 3–4 products are a small group but churn at very high rates, "
          "which suggests they were cross-sold products they did not want.",
        "",
        f"The model's most important features are {top_feats}.",
        "",
        "## 3. Does the risk score hold up?",
        "Actual churn in the held-out test set, by predicted risk segment:",
        "",
        "| Predicted segment | Customers | Actual churn |",
        "|---|---|---|",
    ]
    # One row per segment, highest risk first. Real rows: High 462 / 60.0%, Medium 459 / 16.1%, Low 1,079 / 5.2%.
    for seg in ["High Risk", "Medium Risk", "Low Risk"]:
        # Skip a segment that has no customers at all (it would not be in the index, and .loc would fail).
        if seg in ins["seg_actual"].index:
            v, c = ins["seg_actual"].loc[seg]
            lines.append(f"| {seg} | {int(c):,} | {v:.1%} |")
    # Sections 4-5: campaign advice and honest limits.
    # The last number in section 5 is the test-set churn rate, computed as a weighted average of the segment rates
    # (rate x count, summed, divided by total count) = 407 / 2,000 = 0.20. It is compared with the average risk
    # score 0.35 to show the scores are pushed up (not calibrated probabilities).
    lines += [
        "",
        "## 4. Recommended campaign",
        f"- Contact customers with a churn risk score ≥ **{biz['optimal_threshold']:.2f}** (threshold chosen on "
        f"training data, results below are on the test set): "
        f"{biz['customers_contacted']:,} of 2,000 test customers, reaching {biz['churners_reached']} "
        f"actual churners.",
        f"- Prioritise **inactive members aged 50+** ({ins['old_inactive']['mean']:.0%} churn, "
        f"n={int(ins['old_inactive']['count']):,}) and **customers with 3+ products** "
        f"({ins['multi_prod']['mean']:.0%} churn, n={int(ins['multi_prod']['count']):,}) for "
        "relationship-manager calls.",
        f"- Investigate the **{ins['geo'].index[0]}** portfolio, which churns at "
        f"{ins['geo'].iloc[0]:.1%} against {ins['overall']:.1%} overall.",
        "- Run the campaign as an A/B test (hold out a random control group) so the true save "
        "rate can be measured and fed back into the threshold.",
        "",
        "## 5. Assumptions and limits",
        f"- Campaign economics assume ₹{a['contact_cost_inr']:,} per contact, ₹{a['customer_value_inr']:,} "
        f"per retained churner and a {a['save_rate']:.0%} save rate. These are placeholders, not "
        "figures from the data; change them in `churn_model.py`.",
        f"- Risk scores rank customers well but are not calibrated probabilities: class weighting lifts the "
        f"average test score to {biz['mean_risk_score_test']:.2f} against an actual churn rate of "
        f"{ins['seg_actual']['mean'].mul(ins['seg_actual']['count']).sum() / ins['seg_actual']['count'].sum():.2f}.",
        "- The dataset is a single snapshot, so the model predicts who churned, not when. A production "
        "model would need monthly behavioural history.",
        "- Gender and age are legally sensitive in credit and marketing decisions; they are used here "
        "for analysis, and a deployed model would need a fairness review.",
    ]
    # Join all lines with newlines into one text.
    return "\n".join(lines)


# IN: Markdown text  ->  OUT: the same text without Markdown symbols (for the .txt copy).
# WHY: some readers open the report in Notepad or email, where "**" and "#" look like clutter.
# Example: "## 1. Headline" -> "1. Headline"; "**20.4%** of customers" -> "20.4% of customers"; "`age`" -> "age".
def to_plain_text(md):
    # Regex \*\*(.+?)\*\* = "**", then the shortest text, then "**". Replace with just the inner text (\1).
    txt = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    # Same for single "*italic*" (run AFTER the bold step, so "**" pairs are already gone).
    txt = re.sub(r"\*(.+?)\*", r"\1", txt)
    # Remove backticks around feature names.
    txt = txt.replace("`", "")
    # ^#+ = one or more "#" at the start of a line, then a space. flags=re.M makes ^ match at the start of EVERY line.
    txt = re.sub(r"^#+ ", "", txt, flags=re.M)
    return txt


# ── Main ────────────────────────────────────────────────────────────────────────
# This block runs only when the file is run directly (python report_generator.py), not when another file imports it.
if __name__ == "__main__":
    # Stop with a clear message if churn_model.py has not produced its outputs yet.
    # (data/Churn_Modelling.csv is not checked here; it is the source data and ships with the repo.)
    for path in ["data/scored_customers.csv", f"{REPORTS_DIR}/metrics.json"]:
        if not os.path.exists(path):
            raise SystemExit("Run `python churn_model.py` first.")

    # load_data() returns 3 things; * unpacks them as the 3 arguments of compute_insights. Then build the text.
    md = build_markdown(compute_insights(*load_data()))
    # Save the Markdown and plain-text versions. encoding="utf-8" is needed for "₹", "—" and "≥".
    with open(f"{REPORTS_DIR}/executive_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{REPORTS_DIR}/executive_report.txt", "w", encoding="utf-8") as f:
        f.write(to_plain_text(md))
    # Print the report to the console. On Windows this print can fail on "₹" unless PYTHONIOENCODING=utf-8 is set
    # (run_all.bat and the README set it).
    print(md)
    print(f"\n-> {REPORTS_DIR}/executive_report.md and executive_report.txt")
