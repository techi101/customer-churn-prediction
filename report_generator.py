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

import json
import os
import re
from datetime import datetime

import pandas as pd

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── Load data ───────────────────────────────────────────────────────────────────
def load_data():
    full = pd.read_csv("data/Churn_Modelling.csv").rename(columns={
        "Geography": "geography", "Gender": "gender", "Age": "age",
        "NumOfProducts": "num_products", "IsActiveMember": "is_active_member",
        "Balance": "balance", "Exited": "churn",
    })
    scored = pd.read_csv("data/scored_customers.csv")
    with open(f"{REPORTS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    return full, scored, metrics


def rate(d, mask):
    return d.loc[mask, "churn"].mean()


# ── Analytics engine ────────────────────────────────────────────────────────────
def compute_insights(full, scored, metrics):
    overall = full["churn"].mean()
    geo = full.groupby("geography")["churn"].mean().sort_values(ascending=False)
    prod = full.groupby("num_products")["churn"].agg(["mean", "count"])
    seg_actual = scored.groupby("risk_segment")["actual_churn"].agg(["mean", "count"])

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
    drivers.sort(key=lambda d: d[1] / d[2], reverse=True)

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
def build_markdown(ins):
    m, biz = ins["m"], ins["m"]["business"]
    best, base = m["best_model"], m["baseline_model"]
    bm, lm = m["models"][best], m["models"][base]
    a = biz["assumptions"]
    prod = ins["prod"]
    top_feats = ", ".join(f"`{f['feature']}`" for f in m["feature_importance"][:5])

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
    for name, r_in, r_out, other in ins["drivers"]:
        lines.append(f"| {name} | {r_in:.1%} | {r_out:.1%} for {other} ({r_in/r_out:.1f}x) |")
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
    for seg in ["High Risk", "Medium Risk", "Low Risk"]:
        if seg in ins["seg_actual"].index:
            v, c = ins["seg_actual"].loc[seg]
            lines.append(f"| {seg} | {int(c):,} | {v:.1%} |")
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
    return "\n".join(lines)


def to_plain_text(md):
    txt = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    txt = re.sub(r"\*(.+?)\*", r"\1", txt)
    txt = txt.replace("`", "")
    txt = re.sub(r"^#+ ", "", txt, flags=re.M)
    return txt


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for path in ["data/scored_customers.csv", f"{REPORTS_DIR}/metrics.json"]:
        if not os.path.exists(path):
            raise SystemExit("Run `python churn_model.py` first.")

    md = build_markdown(compute_insights(*load_data()))
    with open(f"{REPORTS_DIR}/executive_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(f"{REPORTS_DIR}/executive_report.txt", "w", encoding="utf-8") as f:
        f.write(to_plain_text(md))
    print(md)
    print(f"\n-> {REPORTS_DIR}/executive_report.md and executive_report.txt")
