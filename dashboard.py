"""
dashboard.py
------------
Interactive Streamlit dashboard for the Customer Churn Prediction project.

Features:
  ─ KPI headline cards  (total customers, churn rate, avg AUC, high-risk count)
  ─ Retention campaign economics  (capture, lift, profit-optimal threshold)
  ─ Risk segments vs actual churn  (bar + pie charts)
  ─ Churn drivers: age, products held, activity, feature importance
  ─ Individual customer look-up with churn risk-score gauge
  ─ Raw scored data table with search & download

Run:
    streamlit run dashboard.py
"""

# WHAT THIS FILE IS: the "shop window" of the project. It does NOT train anything. It only reads the two files
# churn_model.py wrote (data/scored_customers.csv and reports/metrics.json) and shows them as a web page.
# Analogy: churn_model.py is the kitchen that cooks; this dashboard is the menu board that shows what was cooked.
# STREAMLIT = a Python library that turns a plain Python script into a web page. Every widget (dropdown, slider)
# is one line of Python. IMPORTANT: every time the user changes a widget, Streamlit RE-RUNS this whole file from
# top to bottom with the new widget values (that is called a "rerun").
# Real example: the top-ranked test customer (data/scored_customers.csv) is customer 15703793: Germany, Male, age 58,
# 4 products, inactive, balance 133,745. LightGBM risk score 99.70%, Logistic Regression 89.70%, in the campaign,
# and in reality this customer DID churn.
# Real numbers shown (reports/metrics.json): 2,000 test customers, LightGBM test AUC 0.864 (+0.080 vs Logistic
# Regression 0.785), top 10% of scores catch 41% of churners (4.1x), threshold 0.38, 751 to contact, 328 churners
# reached, INR 6.08 lakh vs INR 2.21 lakh contacting everyone.
# Overall flow: page setup -> CSS -> load CSV + JSON (cached) -> sidebar filters -> filter the table
#   -> KPI cards -> campaign economics -> segment charts -> driver charts -> importance + histogram
#   -> one-customer look-up with gauge -> data table + CSV download
#
# streamlit (st) = the web-page library described above.
import streamlit as st
# pandas (pd) = tables (DataFrames). The scored customers are loaded as a DataFrame and filtered with it.
import pandas as pd
# numpy (np) = number arrays. Imported here but not used anywhere in this file.
import numpy as np
# plotly.express (px) = Plotly's quick chart maker: one call gives a bar/pie/scatter/histogram.
# PLOTLY = an interactive chart library: charts in the browser can be hovered, zoomed and saved.
import plotly.express as px
# plotly.graph_objects (go) = Plotly's detailed builder; used only for the speedometer-style gauge.
import plotly.graph_objects as go
# os = operating-system helpers; used to check that the input files exist.
import os
# json = read reports/metrics.json into a Python dict.
import json

# ── Page config ────────────────────────────────────────────────────────────────
# Browser tab title and icon, full-width layout ("wide"), sidebar open at start.
# It is called before any other st.* command (older Streamlit versions insist on that order).
st.set_page_config(
    page_title="Churn Analytics | Banking Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
# CSS = the language that styles a web page (colours, borders, fonts). This block injects custom CSS into the page:
# dark background, styled KPI ("metric") cards, a blue ".section-title" heading style, coloured risk badges,
# and loads the Inter font from Google Fonts. unsafe_allow_html=True lets Streamlit render raw HTML/CSS
# (it is "unsafe" only if the text came from users; here it is our own fixed text).
st.markdown("""
<style>
  /* Base */
  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
  .stApp { background: #0F172A; color: #E2E8F0; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #1E293B; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1E293B, #0F172A);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  div[data-testid="metric-container"] label { color: #94A3B8 !important; font-size: 12px !important; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #F1F5F9 !important; font-size: 28px !important; font-weight: 700 !important;
  }

  /* Section headers */
  .section-title {
    font-size: 18px; font-weight: 700; color: #38BDF8;
    border-left: 4px solid #38BDF8; padding-left: 10px;
    margin: 28px 0 14px 0;
  }

  /* Risk badge */
  .badge-high   { background:#EF4444;color:#fff;padding:2px 10px;border-radius:999px;font-size:12px; }
  .badge-medium { background:#F59E0B;color:#fff;padding:2px 10px;border-radius:999px;font-size:12px; }
  .badge-low    { background:#10B981;color:#fff;padding:2px 10px;border-radius:999px;font-size:12px; }

  /* Plotly charts background */
  .js-plotly-plot .plotly .main-svg { border-radius: 12px; }

  /* Divider */
  hr { border-color: #334155; }
</style>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ── Colour constants ────────────────────────────────────────────────────────────
# Hex colour codes reused by every chart: dark background, card background, border, and the 3 risk colours
# (green = low, amber = medium, red = high) plus blue for neutral charts.
BG       = "#0F172A"
CARD_BG  = "#1E293B"
BORDER   = "#334155"
RED      = "#E63946"
GREEN    = "#2A9D8F"
AMBER    = "#F4A261"
BLUE     = "#38BDF8"
# Shared Plotly look, passed to every chart with **PLOTLY_LAYOUT (** = unpack the dict as keyword arguments).
# Margins 20px and title size 14 are chosen by hand.
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    font_color="#CBD5E1", font_family="Inter",
    margin=dict(l=20, r=20, t=40, b=20),
    title_font_size=14, title_font_color="#E2E8F0",
)

# ── Data loader ────────────────────────────────────────────────────────────────
# IN: nothing  ->  OUT: (scored DataFrame of 2,000 test customers, metrics dict from metrics.json).
# WHY @st.cache_data: because Streamlit re-runs the whole file on every click, the files would be re-read every
# time. st.cache_data remembers the function's result after the first call, so later reruns reuse it instantly.
# Side effect to know: if you re-run churn_model.py while the dashboard is open, clear the cache (or restart
# Streamlit) to see the new files.
# Example: scored has 2,000 rows with columns like customer_id, churn_prob, risk_segment, contact, actual_churn;
#   metrics["best_model"] == "LightGBM".
@st.cache_data
def load_data():
    scored  = pd.read_csv("data/scored_customers.csv")
    # with = open the file and close it automatically; json.load turns the JSON text into a dict.
    with open("reports/metrics.json") as f:
        metrics = json.load(f)
    return scored, metrics

# ── Guard: run pipeline first ──────────────────────────────────────────────────
# If either input file is missing (the model was never trained), show a red error and stop the script here,
# instead of crashing with a confusing "file not found" error.
if not os.path.exists("data/scored_customers.csv") or not os.path.exists("reports/metrics.json"):
    st.error("⚠️  Run `python churn_model.py` first to generate scored data.")
    st.stop()

# Load the data, then pull out the values used everywhere below.
# BEST = "LightGBM", BASE = "Logistic Regression", biz = the business numbers block,
# best_auc = 0.8642, base_auc = 0.7847 (test AUCs from metrics.json).
scored, metrics = load_data()
BEST     = metrics["best_model"]
BASE     = metrics["baseline_model"]
biz      = metrics["business"]
best_auc = metrics["models"][BEST]["test_auc"]
base_auc = metrics["models"][BASE]["test_auc"]
# Fixed order and colour for the three risk segments, so every chart shows Low -> Medium -> High in the same colours.
SEG_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
SEG_COLORS = {"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED}

# ── Sidebar filters ────────────────────────────────────────────────────────────
# The left sidebar: a title and a divider line ("---" in Markdown).
st.sidebar.markdown("## 🏦 Churn Intelligence")
st.sidebar.markdown("---")

# Dropdown choices: "All" + the countries found in the data (France, Germany, Spain), and "All" + the 3 segments.
geo_options  = ["All"] + sorted(scored["geography"].unique().tolist())
seg_options  = ["All"] + SEG_ORDER

# The three filters. selectbox = a dropdown. slider with a tuple default (0.0, 1.0) = a RANGE slider with two handles,
# from 0.0 to 1.0 in steps of 0.01. It returns (low, high), e.g. (0.38, 1.0) to see only campaign-level scores.
sel_geo  = st.sidebar.selectbox("Geography",     geo_options)
sel_seg  = st.sidebar.selectbox("Risk Segment",  seg_options)
prob_min, prob_max = st.sidebar.slider(
    "Churn Risk Score Range", 0.0, 1.0, (0.0, 1.0), 0.01
)

# Small grey note under the filters: these are test-set customers the model never trained on.
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#64748B'>Scores are for the 2,000-customer held-out test set, "
    "which the model never saw during training.</small>",
    unsafe_allow_html=True
)

# ── Apply filters ──────────────────────────────────────────────────────────────
# Start from a copy of all 2,000 rows, then narrow it down. df[condition] keeps only rows where the condition is True.
df = scored.copy()
# Country filter, only if a specific country is picked.
if sel_geo != "All":
    df = df[df["geography"] == sel_geo]
# Risk-segment filter, only if a specific segment is picked.
if sel_seg != "All":
    df = df[df["risk_segment"] == sel_seg]
# Score-range filter is always applied (default 0.0-1.0 keeps everyone).
df = df[(df["churn_prob"] >= prob_min) & (df["churn_prob"] <= prob_max)]

# No rows left (e.g. "Low Risk" + score range 0.9-1.0): show a yellow warning and stop, so no chart crashes on an
# empty table.
if df.empty:
    st.warning("No customers match these filters.")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
# A big banner (HTML) with the title and "Real bank data (10,000 customers) · LightGBM · test AUC 0.864".
# It is an f-string, so the {...} parts are filled from metrics.json.
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1E3A5F,#0F172A);
            border:1px solid #334155;border-radius:16px;padding:28px 36px;margin-bottom:24px;">
  <h1 style="margin:0;font-size:28px;color:#F1F5F9;">
    📊 Customer Churn Prediction Dashboard
  </h1>
  <p style="margin:8px 0 0;color:#94A3B8;font-size:15px;">
    Real bank data ({metrics['dataset']['rows']:,} customers) · {BEST} · test AUC {best_auc:.3f}
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
# KPI = Key Performance Indicator = a headline number. st.columns(4) = 4 side-by-side boxes.
k1, k2, k3, k4 = st.columns(4)

# Cards 1-3 follow the filters: number of customers shown, their real churn rate (mean of 0/1 x 100;
# about 20% with no filter: 407 churners of 2,000), and how many are "High Risk" (462 with no filter).
# Card 4 is fixed: the model's test AUC 0.864, with a delta (small arrow text) of +0.080 vs Logistic Regression.
k1.metric("👥 Customers (test set)", f"{len(df):,}")
k2.metric("🔴 Actual Churn Rate",   f"{df['actual_churn'].mean()*100:.1f}%")
k3.metric("⚠️ High-Risk Customers", f"{(df['risk_segment'] == 'High Risk').sum():,}")
k4.metric(f"🎯 Model AUC ({BEST})", f"{best_auc:.3f}",
          delta=f"{best_auc - base_auc:+.3f} vs {BASE}")

# ═══════════════════════════════════════════════════════════════════════════════
# RETENTION CAMPAIGN ECONOMICS
# ═══════════════════════════════════════════════════════════════════════════════
# These 4 cards come straight from metrics.json and do NOT change with the sidebar filters (whole test set).
st.markdown('<div class="section-title">Retention Campaign Economics</div>',
            unsafe_allow_html=True)
a = biz["assumptions"]
b1, b2, b3, b4 = st.columns(4)
# Card 1: "41%" of churners are in the top 10% of scores; delta "4.1x better than random" (LIFT).
b1.metric("Churners in top 10% of scores", f"{biz['top10_capture']:.0%}",
          delta=f"{biz['lift_top10']:.1f}x better than random")
# Card 2: the contact threshold 0.38 (chosen on training folds in churn_model.py).
b2.metric("Contact threshold", f"{biz['optimal_threshold']:.2f}")
# Card 3: 751 customers to contact, reaching 328 churners. delta_color="off" = grey text, not a green/red arrow,
# because this delta is information, not "good" or "bad".
b3.metric("Customers to contact", f"{biz['customers_contacted']:,}",
          delta=f"reaches {biz['churners_reached']} churners", delta_color="off")
# Card 4: net value in lakh (/1e5; 1 lakh = 100,000): "₹6.08 L" vs "₹2.21 L contacting everyone".
b4.metric("Campaign net value", f"₹{biz['net_value_optimal_inr']/1e5:.2f} L",
          delta=f"vs ₹{biz['net_value_contact_all_inr']/1e5:.2f} L contacting everyone",
          delta_color="off")
# Two honest fine-print lines: scores are risk scores (not calibrated probabilities: the average test score is 0.35
# while real churn is 0.20), and the stated cost assumptions (₹500 per contact, ₹10,000 per saved churner, 30% save rate).
st.caption("Risk scores rank customers; they are not calibrated probabilities (class weighting pushes them up). "
           "The contact threshold was chosen on training data and is reported here on the test set.")
st.caption(f"Assumptions (editable in churn_model.py): contacting a customer costs ₹{a['contact_cost_inr']:,}, "
           f"a retained churner is worth ₹{a['customer_value_inr']:,}, and the offer retains "
           f"{a['save_rate']:.0%} of the churners it reaches.")

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Risk Segment + Geo breakdown
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Portfolio Risk Segmentation</div>',
            unsafe_allow_html=True)

# Three columns; the numbers are relative widths (1.2 : 1.2 : 1), chosen by hand so the pie chart is a bit narrower.
col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

# Count customers per segment in the filtered table. reindex(SEG_ORDER, fill_value=0) forces the order
# Low, Medium, High and puts 0 for a segment with nobody in it. reset_index turns it into a 2-column table.
# With no filter: Low 1,079, Medium 459, High 462.
seg_counts = (df["risk_segment"].value_counts()
                .reindex(SEG_ORDER, fill_value=0).reset_index())
seg_counts.columns = ["segment", "count"]

# "with col_a:" = everything inside is drawn in the first column.
# Chart: bar of customers per segment, coloured green/amber/red. showlegend=False because the x labels already
# name the segments. width="stretch" = fill the column width.
with col_a:
    fig_seg = px.bar(
        seg_counts, x="segment", y="count", color="segment",
        color_discrete_map=SEG_COLORS,
        title="Customers by Risk Segment",
        labels={"count": "Customers", "segment": ""},
    )
    fig_seg.update_layout(**PLOTLY_LAYOUT, showlegend=False)
    fig_seg.update_traces(marker_line_width=0)
    st.plotly_chart(fig_seg, width="stretch")

# Chart: the "does the score hold up?" check. Real churn rate (%) inside each PREDICTED segment.
# With no filter: High 60.0%, Medium 16.1%, Low 5.2% (README) - higher predicted risk really means more churn.
# observed=False only matters for category-type columns; risk_segment is plain text when read from CSV, so it
# changes nothing here. reindex(SEG_ORDER) keeps the Low -> High order (a missing segment becomes empty).
with col_b:
    seg_actual = (df.groupby("risk_segment", observed=False)["actual_churn"].mean()
                    .reindex(SEG_ORDER).mul(100).round(1).reset_index())
    fig_cal = px.bar(
        seg_actual, x="risk_segment", y="actual_churn", color="risk_segment",
        color_discrete_map=SEG_COLORS,
        title="Actual Churn Rate per Predicted Segment (%)",
        labels={"actual_churn": "Actual churn (%)", "risk_segment": ""},
    )
    fig_cal.update_layout(**PLOTLY_LAYOUT, showlegend=False)
    fig_cal.update_traces(marker_line_width=0)
    st.plotly_chart(fig_cal, width="stretch")

# Chart: donut (pie with a hole) of the same segment counts. hole=0.55 = the hole is 55% of the radius
# (chosen by hand for the donut look).
with col_c:
    fig_pie = px.pie(
        seg_counts, names="segment", values="count",
        color="segment", color_discrete_map=SEG_COLORS,
        title="Risk Distribution",
        hole=0.55,
    )
    fig_pie.update_layout(**PLOTLY_LAYOUT)
    fig_pie.update_traces(textfont_color="#E2E8F0")
    st.plotly_chart(fig_pie, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Drivers of churn
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Key Churn Drivers</div>',
            unsafe_allow_html=True)
col_d, col_e = st.columns(2)

# Scatter plot: one dot per customer, x = age, y = LightGBM risk score, colour = segment.
# Shows at most 1,000 random customers (min(1000, len(df)) avoids an error when fewer rows are left) so the chart
# stays readable and fast. random_state=42 = the same sample on every rerun. opacity 0.65 lets overlapping dots show.
# category_orders keeps the legend in Low, Medium, High order.
with col_d:
    fig_scatter = px.scatter(
        df.sample(min(1000, len(df)), random_state=42),
        x="age", y="churn_prob",
        color="risk_segment",
        color_discrete_map=SEG_COLORS,
        category_orders={"risk_segment": SEG_ORDER},
        opacity=0.65,
        title="Age vs Churn Risk Score",
        labels={"age": "Age", "churn_prob": f"Churn risk score ({BEST})"},
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_scatter, width="stretch")

# Grouped bar chart: average risk score (x100) for every (number of products, active?) pair.
# How to read it (no filter): the 3- and 4-product bars are the tallest, matching the EDA finding that 3-4 products
# churn at 86% (README), and in every product group the Inactive bar is taller than the Active bar.
# map({0: "Inactive", 1: "Active"}) turns the 0/1 column into words for the legend.
# barmode="group" = put the Inactive and Active bars side by side.
with col_e:
    prod = (df.groupby(["num_products", "is_active_member"])["churn_prob"].mean()
              .mul(100).round(1).reset_index())
    prod["member"] = prod["is_active_member"].map({0: "Inactive", 1: "Active"})
    fig_prod = px.bar(
        prod, x="num_products", y="churn_prob", color="member", barmode="group",
        color_discrete_map={"Inactive": RED, "Active": GREEN},
        title="Avg Risk Score by Products Held and Activity",
        labels={"num_products": "Number of products", "churn_prob": "Avg risk score (x100)", "member": ""},
    )
    fig_prod.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_prod, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Feature importance + risk-score histogram
# ═══════════════════════════════════════════════════════════════════════════════
col_f, col_g = st.columns(2)

# Top 10 features from metrics.json (already sorted biggest first: age 0.29, num_products 0.20, balance 0.10 ...).
# .iloc[::-1] reverses the order because a horizontal Plotly bar chart draws the first row at the bottom;
# reversed, the biggest feature (age) ends up on top. Fixed numbers: they do not follow the filters.
with col_f:
    fi = pd.DataFrame(metrics["feature_importance"]).head(10).iloc[::-1]
    fig_fi = px.bar(
        fi, x="importance", y="feature", orientation="h",
        color_discrete_sequence=[BLUE],
        title=f"Top 10 Features ({BEST})",
        labels={"importance": "Relative importance", "feature": ""},
    )
    fig_fi.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_fi, width="stretch")

# HISTOGRAM of risk scores for the filtered customers (40 bars, chosen by hand), with a red dashed vertical line
# at the contact threshold 0.38. Everyone to the right of the line is in the retention campaign.
with col_g:
    fig_hist = px.histogram(
        df, x="churn_prob", nbins=40,
        color_discrete_sequence=[BLUE],
        title="Churn Risk Score Distribution",
        labels={"churn_prob": f"Risk score — {BEST}"},
    )
    fig_hist.add_vline(x=biz["optimal_threshold"], line_dash="dash", line_color=RED,
                       annotation_text=f"Contact threshold ({biz['optimal_threshold']:.2f})",
                       annotation_font_color=RED)
    fig_hist.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_hist, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL CUSTOMER LOOK-UP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 Individual Customer Risk Look-up</div>',
            unsafe_allow_html=True)

# Sort the filtered customers from highest to lowest score. reset_index(drop=True) renumbers rows 0, 1, 2 ...
# so row 0 = rank 1 = riskiest (with no filter: customer 15703793, score 0.997).
ranked   = df.sort_values("churn_prob", ascending=False).reset_index(drop=True)
# More than one customer: show a slider to pick a rank (1 = riskiest, default 1). Subtract 1 because Python
# counts rows from 0.
if len(ranked) > 1:
    cust_idx = st.slider("Customer rank (1 = highest risk in current filter)", 1, len(ranked), 1) - 1
# Exactly one customer: a slider from 1 to 1 would fail (min must be below max), so skip the slider and just
# show that one customer. (This is the "dashboard crash" fix from the last commit.)
else:
    cust_idx = 0
    st.caption("Only one customer matches the current filters.")
# Pick the chosen customer's row and their LightGBM score, Logistic Regression score and segment.
row      = ranked.iloc[cust_idx]
prob     = row["churn_prob"]
prob_lr  = row["churn_prob_lr"]
seg      = row["risk_segment"]

# Segment name -> a coloured HTML "pill" badge (the badge-* CSS classes defined at the top).
# .get(str(seg), seg): if the segment is not one of the three (should not happen), just show the plain value.
seg_badge = {
    "High Risk":   "<span class='badge-high'>High Risk</span>",
    "Medium Risk": "<span class='badge-medium'>Medium Risk</span>",
    "Low Risk":    "<span class='badge-low'>Low Risk</span>",
}.get(str(seg), seg)

# Left column wider (1.4) for the details card, right column (1) for the gauge.
c1, c2 = st.columns([1.4, 1])

# Customer details card (an HTML table built with an f-string): ID, segment badge, both model scores as %,
# age/gender/country, products, active?, balance, tenure, in campaign?, and the real outcome.
# The score text is red above 0.6, amber above 0.3, else green - the same cut-offs as the risk segments.
# Example (rank 1): Customer 15703793, High Risk, 99.70%, 89.70%, 58 / Male / Germany, 4 products, No, 133,745,
#   2 years, Yes, Churned.
with c1:
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:22px;">
      <h3 style="margin:0 0 16px;color:#F1F5F9;">Customer {int(row['customer_id'])}</h3>
      <table style="width:100%;font-size:14px;color:#CBD5E1;">
        <tr><td style="padding:6px 0;color:#64748B;">Risk Segment</td>
            <td>{seg_badge}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">{BEST} Risk Score</td>
            <td style="color:{RED if prob>0.6 else AMBER if prob>0.3 else GREEN};font-weight:700;">
              {prob:.2%}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">{BASE} Risk Score</td>
            <td>{prob_lr:.2%}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Age / Gender / Country</td>
            <td>{int(row['age'])} / {row['gender']} / {row['geography']}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Products Held</td>
            <td>{int(row['num_products'])}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Active Member</td>
            <td>{"Yes" if row['is_active_member'] else "No"}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Balance</td>
            <td>{row['balance']:,.0f}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Tenure (years)</td>
            <td>{int(row['tenure_years'])}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">In Retention Campaign</td>
            <td>{"Yes" if row['contact'] else "No"}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Actual Outcome</td>
            <td>{"❌ Churned" if row['actual_churn'] else "🟢 Retained"}</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

# Gauge (speedometer) for the chosen customer's score on a 0-100 scale (score x 100).
# The needle bar is red above 60, amber above 30, else green. The background bands 0-30 / 30-60 / 60-100 are dark
# green / brown / dark red, matching the Low / Medium / High segment cut-offs 0.30 and 0.60.
# Height 260px and margins are chosen by hand to fit beside the details card.
with c2:
    gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prob * 100,
        title = {"text": "Churn Risk Score", "font": {"color": "#E2E8F0", "size": 14}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#64748B"},
            "bar":   {"color": RED if prob > 0.6 else AMBER if prob > 0.3 else GREEN},
            "bgcolor": CARD_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#14532D"},
                {"range": [30, 60], "color": "#713F12"},
                {"range": [60, 100],"color": "#7F1D1D"},
            ],
        },
        number = {"suffix": "%", "font": {"color": "#E2E8F0", "size": 32}},
    ))
    gauge.update_layout(
        paper_bgcolor=CARD_BG, font_color="#CBD5E1",
        height=260, margin=dict(l=30, r=30, t=40, b=10)
    )
    st.plotly_chart(gauge, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 Scored Customer Data</div>',
            unsafe_allow_html=True)

# The 13 columns shown in the table (a subset of the CSV), riskiest customer first.
show_cols = ["customer_id", "churn_prob", "risk_segment", "contact", "actual_churn",
             "age", "geography", "gender", "num_products", "is_active_member",
             "balance", "tenure_years", "credit_score"]
display_df = ranked[show_cols].copy()
# Show the score as a percentage text: 0.99697 -> "99.70%". (.copy() above keeps `ranked` itself as numbers.)
display_df["churn_prob"] = display_df["churn_prob"].map("{:.2%}".format)

# Interactive table of the first 200 rows only (keeps the page fast), 320px tall, full width.
st.dataframe(display_df.head(200), width="stretch", height=320)

# Download button: the WHOLE filtered, ranked table (all columns, numbers not formatted) as a CSV file.
# .encode() turns the CSV text into bytes (UTF-8), which is what download_button sends to the browser.
csv_bytes = ranked.to_csv(index=False).encode()
st.download_button(
    "⬇️  Download Filtered Data (CSV)", data=csv_bytes,
    file_name="churn_scored_filtered.csv", mime="text/csv"
)
