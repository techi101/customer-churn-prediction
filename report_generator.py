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

# WHAT THIS FILE IS: the report writer. Fills a fixed text template with numbers from churn_model.py's outputs,
# for a bank manager who never opens Python. Trains nothing.
# Example output line: "Contact customers with a churn risk score ≥ 0.38 ... 751 of 2,000 test customers, reaching 328".
# Flow: check files -> load CSV + scored CSV + metrics.json -> segment churn rates
#   -> fill Markdown -> save .md -> strip symbols -> save .txt -> print
import json  # read reports/metrics.json
import os  # check files exist, make folder
import re  # regex = text patterns, strips **bold** etc.
from datetime import datetime  # today's date for the report header

import pandas as pd  # tables, churn rates per group

REPORTS_DIR = "reports"  # output folder
os.makedirs(REPORTS_DIR, exist_ok=True)  # create it, no error if it exists


# ── BLOCK: Load data ─────────────────────────────────────────────────────────
# IN: nothing -> OUT: (full 10,000 rows, scored 2,000 test rows, metrics dict)
# e.g. full.loc[0] = customer 15634602, age 42, France, Female, churn 1
def load_data():
    full = pd.read_csv("data/Churn_Modelling.csv").rename(columns={  # only the 7 used columns renamed
        "Geography": "geography", "Gender": "gender", "Age": "age",  # country, gender, age
        "NumOfProducts": "num_products", "IsActiveMember": "is_active_member",  # products, active 0/1
        "Balance": "balance", "Exited": "churn",  # balance, 1 = left
    })
    scored = pd.read_csv("data/scored_customers.csv")  # test set with risk scores
    with open(f"{REPORTS_DIR}/metrics.json") as f:  # auto-close file
        metrics = json.load(f)  # JSON text -> dict
    return full, scored, metrics  # 3 inputs for compute_insights


# IN: table + True/False mask per row -> OUT: churn rate where mask is True
# e.g. rate(full, full["age"] >= 50) -> 0.454
def rate(d, mask):
    return d.loc[mask, "churn"].mean()  # mean of 0/1 = churn share


# ── BLOCK: Analytics engine ──────────────────────────────────────────────────
# IN: full, scored, metrics -> OUT: dict "ins" with every number the report needs
# e.g. ins["overall"] = 0.2037, ins["geo"] Germany 0.324 first, ins["old_inactive"] mean 0.82, count 484
def compute_insights(full, scored, metrics):
    overall = full["churn"].mean()  # 0.2037 = 20.4%
    geo = full.groupby("geography")["churn"].mean().sort_values(ascending=False)  # Germany 32.4% first
    prod = full.groupby("num_products")["churn"].agg(["mean", "count"])  # rate + count per 1 to 4 products
    seg_actual = scored.groupby("risk_segment")["actual_churn"].agg(["mean", "count"])  # test set, per segment

    # ── Step 1: drivers = (group, rate inside, rate of comparison group, comparison label) ──
    drivers = [
        ("Age 50+",                    rate(full, full["age"] >= 50),  # 45.4%
                                       rate(full, full["age"] < 50), "under 50"),  # 16.3%
        ("Inactive members",           rate(full, full["is_active_member"] == 0),  # inactive rate
                                       rate(full, full["is_active_member"] == 1), "active members"),  # active rate
        (f"Customers in {geo.index[0]}", geo.iloc[0],  # top country, not hard-coded (Germany here)
                                       rate(full, full["geography"] != geo.index[0]), "other countries"),  # all others
        ("Female customers",           rate(full, full["gender"] == "Female"),  # female rate
                                       rate(full, full["gender"] == "Male"), "male customers"),  # male rate
    ]
    drivers.sort(key=lambda d: d[1] / d[2], reverse=True)  # by ratio: Age 2.8x, Germany 2.0x, Inactive 1.9x, Female 1.5x

    # ── Step 2: pack all numbers into one dict ──
    return {
        "overall": overall,  # 0.2037
        "n": len(full),  # 10,000
        "geo": geo,  # rate per country
        "prod": prod,  # rate + count per products
        "seg_actual": seg_actual,  # test churn per segment
        "drivers": drivers,  # sorted driver list
        "m": metrics,  # whole metrics dict
        # age 50+ AND inactive: churn 82%, n=484
        "old_inactive": full.loc[(full["age"] >= 50) & (full["is_active_member"] == 0), "churn"].agg(["mean", "count"]),
        "multi_prod": full.loc[full["num_products"] >= 3, "churn"].agg(["mean", "count"]),  # 3+ products: 86%, n=326
    }


# ── BLOCK: Report builder ────────────────────────────────────────────────────
# IN: insights dict -> OUT: full report as one Markdown text, e.g. "- **20.4%** of customers churned."
# MARKDOWN = simple text format: "# Title" heading, "**bold**", "| a | b |" table rows.
def build_markdown(ins):
    m, biz = ins["m"], ins["m"]["business"]  # metrics, business block
    best, base = m["best_model"], m["baseline_model"]  # "LightGBM", "Logistic Regression"
    bm, lm = m["models"][best], m["models"][base]  # their score dicts
    a = biz["assumptions"]  # ₹500, ₹10,000, 30%
    prod = ins["prod"]  # churn by products
    top_feats = ", ".join(f"`{f['feature']}`" for f in m["feature_importance"][:5])  # `age`, `num_products`, ...

    # ── Step 1: sections 1 and 2 ({x:.1%} -> "20.4%", /1e5 -> lakh) ──
    lines = [
        "# Customer Churn — Executive Briefing",  # report title
        f"*Generated {datetime.now():%d %b %Y} from {ins['n']:,} real bank customers "  # date, 10,000
        f"(public Bank Customer Churn dataset).*",
        "",  # blank line
        "## 1. Headline",  # section heading
        f"- **{ins['overall']:.1%}** of customers churned.",  # 20.4%
        f"- A {best} model ranks customers by churn risk with a test AUC of **{bm['test_auc']:.3f}** "  # 0.864
        # CV 0.863 +/- 0.011, baseline 0.785
        f"(5-fold CV {bm['cv_auc_mean']:.3f} ± {bm['cv_auc_std']:.3f}), versus {lm['test_auc']:.3f} for a "
        f"{base} baseline.",
        f"- Contacting only the riskiest **10%** of customers reaches **{biz['top10_capture']:.0%}** of all "  # 41%
        f"churners — **{biz['lift_top10']:.1f}x** better than picking customers at random.",  # 4.1x
        f"- Under the stated campaign assumptions, targeting by model score is worth "
        f"**₹{biz['net_value_optimal_inr']/1e5:.2f} lakh** per 2,000 customers, versus "  # ₹6.08 lakh
        f"₹{biz['net_value_contact_all_inr']/1e5:.2f} lakh for contacting everyone.",  # ₹2.21 lakh
        "",  # blank line
        "## 2. Who churns",  # section heading
        "| Segment | Churn rate | Compared with |",  # table header
        "|---|---|---|",  # table divider
    ]
    for name, r_in, r_out, other in ins["drivers"]:  # one row per driver
        # "| Age 50+ | 45.4% | 16.3% ... (2.8x) |"
        lines.append(f"| {name} | {r_in:.1%} | {r_out:.1%} for {other} ({r_in/r_out:.1f}x) |")
    # ── Step 2: products paragraph, top features, section 3 header ──
    lines += [
        "",  # blank line
        "**Products held is the sharpest signal:** "
        # k = products, v = churn rate, c = count; e.g. "1 product → 28% churn (n=5,084)"
        + ", ".join(f"{int(k)} product{'s' if k > 1 else ''} → {v:.0%} churn (n={int(c):,})"
                    for k, (v, c) in prod.iterrows())  # one item per products row
        + ". Customers with 3–4 products are a small group but churn at very high rates, "
          "which suggests they were cross-sold products they did not want.",
        "",  # blank line
        f"The model's most important features are {top_feats}.",  # top 5 features
        "",  # blank line
        "## 3. Does the risk score hold up?",  # section heading
        "Actual churn in the held-out test set, by predicted risk segment:",
        "",  # blank line
        "| Predicted segment | Customers | Actual churn |",  # table header
        "|---|---|---|",  # table divider
    ]
    # ── Step 3: segment rows (High 462 / 60.0%, Medium 459 / 16.1%, Low 1,079 / 5.2%) ──
    for seg in ["High Risk", "Medium Risk", "Low Risk"]:  # highest risk first
        if seg in ins["seg_actual"].index:  # skip empty segment (.loc would fail)
            v, c = ins["seg_actual"].loc[seg]  # rate, count
            lines.append(f"| {seg} | {int(c):,} | {v:.1%} |")  # one table row
    # ── Step 4: sections 4 and 5, campaign advice and honest limits ──
    lines += [
        "",  # blank line
        "## 4. Recommended campaign",  # section heading
        f"- Contact customers with a churn risk score ≥ **{biz['optimal_threshold']:.2f}** (threshold chosen on "  # 0.38
        f"training data, results below are on the test set): "
        f"{biz['customers_contacted']:,} of 2,000 test customers, reaching {biz['churners_reached']} "  # 751, 328
        f"actual churners.",
        f"- Prioritise **inactive members aged 50+** ({ins['old_inactive']['mean']:.0%} churn, "  # 82%
        f"n={int(ins['old_inactive']['count']):,}) and **customers with 3+ products** "  # 484
        f"({ins['multi_prod']['mean']:.0%} churn, n={int(ins['multi_prod']['count']):,}) for "  # 86%, 326
        "relationship-manager calls.",
        f"- Investigate the **{ins['geo'].index[0]}** portfolio, which churns at "  # Germany
        f"{ins['geo'].iloc[0]:.1%} against {ins['overall']:.1%} overall.",  # 32.4% vs 20.4%
        "- Run the campaign as an A/B test (hold out a random control group) so the true save "
        "rate can be measured and fed back into the threshold.",
        "",  # blank line
        "## 5. Assumptions and limits",  # section heading
        # ₹500, ₹10,000
        f"- Campaign economics assume ₹{a['contact_cost_inr']:,} per contact, ₹{a['customer_value_inr']:,} "
        f"per retained churner and a {a['save_rate']:.0%} save rate. These are placeholders, not "  # 30%
        "figures from the data; change them in `churn_model.py`.",
        f"- Risk scores rank customers well but are not calibrated probabilities: class weighting lifts the "
        f"average test score to {biz['mean_risk_score_test']:.2f} against an actual churn rate of "  # 0.35
        # weighted average of segment rates = 407 / 2,000 = 0.20
        f"{ins['seg_actual']['mean'].mul(ins['seg_actual']['count']).sum() / ins['seg_actual']['count'].sum():.2f}.",
        "- The dataset is a single snapshot, so the model predicts who churned, not when. A production "
        "model would need monthly behavioural history.",
        "- Gender and age are legally sensitive in credit and marketing decisions; they are used here "
        "for analysis, and a deployed model would need a fairness review.",
    ]
    return "\n".join(lines)  # one text, lines joined by newlines


# IN: Markdown text -> OUT: same text without Markdown symbols (for the .txt copy)
# e.g. "## 1. Headline" -> "1. Headline"; "**20.4%**" -> "20.4%"; "`age`" -> "age"
def to_plain_text(md):
    txt = re.sub(r"\*\*(.+?)\*\*", r"\1", md)  # **bold** -> bold
    txt = re.sub(r"\*(.+?)\*", r"\1", txt)  # *italic* -> italic (after bold step)
    txt = txt.replace("`", "")  # drop backticks
    txt = re.sub(r"^#+ ", "", txt, flags=re.M)  # drop "# " at every line start (re.M)
    return txt  # plain text


# ── BLOCK: Main ──────────────────────────────────────────────────────────────
# Runs only when started directly, not when imported.
if __name__ == "__main__":
    # ── Step 1: stop if churn_model.py outputs are missing (source CSV ships with repo) ──
    for path in ["data/scored_customers.csv", f"{REPORTS_DIR}/metrics.json"]:  # 2 required files
        if not os.path.exists(path):  # missing?
            raise SystemExit("Run `python churn_model.py` first.")  # clear message, exit

    # ── Step 2: build and save the report ──
    md = build_markdown(compute_insights(*load_data()))  # * unpacks the 3 loaded items
    with open(f"{REPORTS_DIR}/executive_report.md", "w", encoding="utf-8") as f:  # utf-8 needed for ₹, —, ≥
        f.write(md)  # Markdown version
    with open(f"{REPORTS_DIR}/executive_report.txt", "w", encoding="utf-8") as f:  # plain-text copy
        f.write(to_plain_text(md))  # symbols stripped
    # Windows console may fail on "₹" unless PYTHONIOENCODING=utf-8 (run_all.bat sets it)
    print(md)  # show report in console
    print(f"\n-> {REPORTS_DIR}/executive_report.md and executive_report.txt")  # where files went
