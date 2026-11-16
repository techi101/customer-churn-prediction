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

# WHAT THIS FILE IS: the web page of the project. Trains nothing; only shows the 2 files churn_model.py wrote
# (data/scored_customers.csv, reports/metrics.json). Streamlit re-runs this whole file on every widget change.
# Example: rank 1 = customer 15703793 (Germany, Male, 58, 4 products, inactive): score 99.70%, really churned.
# Flow: page setup -> CSS -> load CSV + JSON (cached) -> sidebar filters -> KPI cards -> campaign economics
#   -> segment charts -> driver charts -> importance + histogram -> customer look-up + gauge -> table + download
import streamlit as st  # turns this script into a web page
import pandas as pd  # tables (DataFrame), used to filter customers
import numpy as np  # number arrays (imported, not used here)
import plotly.express as px  # quick interactive charts (bar, pie, scatter)
import plotly.graph_objects as go  # detailed builder, used for the gauge
import os  # check the input files exist
import json  # read reports/metrics.json

# ── BLOCK: Page config ───────────────────────────────────────────────────────
# Must come before any other st.* command.
st.set_page_config(
    page_title="Churn Analytics | Banking Intelligence",  # browser tab title
    page_icon="📊",  # browser tab icon
    layout="wide",  # use full page width
    initial_sidebar_state="expanded",  # sidebar open at start
)

# ── BLOCK: Global CSS ────────────────────────────────────────────────────────
# CSS = page styling: dark theme, KPI cards, .section-title, risk badges, Inter font.
# unsafe_allow_html=True = render raw HTML/CSS (safe here: our own fixed text). No comments inside the string.
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

# ── BLOCK: Colour constants ──────────────────────────────────────────────────
BG       = "#0F172A"  # dark page background
CARD_BG  = "#1E293B"  # card / chart background
BORDER   = "#334155"  # card border
RED      = "#E63946"  # high risk
GREEN    = "#2A9D8F"  # low risk
AMBER    = "#F4A261"  # medium risk
BLUE     = "#38BDF8"  # neutral charts
# Shared Plotly look, passed as **PLOTLY_LAYOUT (** = unpack dict as keyword arguments)
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,  # dark chart background
    font_color="#CBD5E1", font_family="Inter",  # light Inter text
    margin=dict(l=20, r=20, t=40, b=20),  # small margins, room for title
    title_font_size=14, title_font_color="#E2E8F0",  # title style
)

# ── BLOCK: Data loader ───────────────────────────────────────────────────────
# IN: nothing -> OUT: (2,000-row scored DataFrame, metrics dict), e.g. metrics["best_model"] == "LightGBM"
# cache_data = remember the result across reruns; clear cache after re-running churn_model.py.
@st.cache_data
def load_data():
    scored  = pd.read_csv("data/scored_customers.csv")  # 2,000 test customers with scores
    with open("reports/metrics.json") as f:  # auto-close file
        metrics = json.load(f)  # JSON text -> dict
    return scored, metrics  # both for the page

# ── BLOCK: Guard: run pipeline first ─────────────────────────────────────────
# Missing files -> clear red error instead of a crash.
if not os.path.exists("data/scored_customers.csv") or not os.path.exists("reports/metrics.json"):  # either missing?
    st.error("⚠️  Run `python churn_model.py` first to generate scored data.")  # red error box
    st.stop()  # end this run here

# ── BLOCK: load data and key values ──────────────────────────────────────────
scored, metrics = load_data()  # cached after first run
BEST     = metrics["best_model"]  # "LightGBM"
BASE     = metrics["baseline_model"]  # "Logistic Regression"
biz      = metrics["business"]  # business numbers block
best_auc = metrics["models"][BEST]["test_auc"]  # 0.8642
base_auc = metrics["models"][BASE]["test_auc"]  # 0.7847
SEG_ORDER = ["Low Risk", "Medium Risk", "High Risk"]  # same order in every chart
SEG_COLORS = {"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED}  # same colours in every chart

# ── BLOCK: Sidebar filters ───────────────────────────────────────────────────
st.sidebar.markdown("## 🏦 Churn Intelligence")  # sidebar title
st.sidebar.markdown("---")  # divider line

geo_options  = ["All"] + sorted(scored["geography"].unique().tolist())  # All, France, Germany, Spain
seg_options  = ["All"] + SEG_ORDER  # All + 3 segments

sel_geo  = st.sidebar.selectbox("Geography",     geo_options)  # dropdown, returns chosen text
sel_seg  = st.sidebar.selectbox("Risk Segment",  seg_options)  # dropdown
prob_min, prob_max = st.sidebar.slider(  # tuple default = range slider with 2 handles
    "Churn Risk Score Range", 0.0, 1.0, (0.0, 1.0), 0.01  # 0 to 1, step 0.01, e.g. (0.38, 1.0)
)

st.sidebar.markdown("---")  # divider line
st.sidebar.markdown(  # small grey note
    "<small style='color:#64748B'>Scores are for the 2,000-customer held-out test set, "  # held-out = never trained on
    "which the model never saw during training.</small>",
    unsafe_allow_html=True  # render the HTML
)

# ── BLOCK: Apply filters ─────────────────────────────────────────────────────
df = scored.copy()  # start from all 2,000 rows
if sel_geo != "All":  # a country was picked
    df = df[df["geography"] == sel_geo]  # keep that country only
if sel_seg != "All":  # a segment was picked
    df = df[df["risk_segment"] == sel_seg]  # keep that segment only
df = df[(df["churn_prob"] >= prob_min) & (df["churn_prob"] <= prob_max)]  # score range, always applied

if df.empty:  # e.g. Low Risk + range 0.9 to 1.0
    st.warning("No customers match these filters.")  # yellow warning
    st.stop()  # stop before charts crash on empty data

# ── BLOCK: Header banner ─────────────────────────────────────────────────────
# f-string HTML: "Real bank data (10,000 customers) · LightGBM · test AUC 0.864"
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
# KPI = headline number. Cards 1 to 3 follow the filters; card 4 is fixed.
k1, k2, k3, k4 = st.columns(4)  # 4 side-by-side boxes

k1.metric("👥 Customers (test set)", f"{len(df):,}")  # 2,000 with no filter
k2.metric("🔴 Actual Churn Rate",   f"{df['actual_churn'].mean()*100:.1f}%")  # ~20.4% with no filter
k3.metric("⚠️ High-Risk Customers", f"{(df['risk_segment'] == 'High Risk').sum():,}")  # 462 with no filter
k4.metric(f"🎯 Model AUC ({BEST})", f"{best_auc:.3f}",  # 0.864
          delta=f"{best_auc - base_auc:+.3f} vs {BASE}")  # "+0.080 vs Logistic Regression"

# ═══════════════════════════════════════════════════════════════════════════════
# RETENTION CAMPAIGN ECONOMICS
# ═══════════════════════════════════════════════════════════════════════════════
# From metrics.json, whole test set: does NOT change with filters.
st.markdown('<div class="section-title">Retention Campaign Economics</div>',  # styled heading
            unsafe_allow_html=True)  # render HTML
a = biz["assumptions"]  # cost, value, save rate
b1, b2, b3, b4 = st.columns(4)  # 4 boxes
b1.metric("Churners in top 10% of scores", f"{biz['top10_capture']:.0%}",  # "41%"
          delta=f"{biz['lift_top10']:.1f}x better than random")  # lift 4.1x
b2.metric("Contact threshold", f"{biz['optimal_threshold']:.2f}")  # 0.38, chosen on training folds
b3.metric("Customers to contact", f"{biz['customers_contacted']:,}",  # 751
          delta=f"reaches {biz['churners_reached']} churners", delta_color="off")  # 328; off = grey, not good/bad
b4.metric("Campaign net value", f"₹{biz['net_value_optimal_inr']/1e5:.2f} L",  # "₹6.08 L" (1 lakh = 100,000)
          delta=f"vs ₹{biz['net_value_contact_all_inr']/1e5:.2f} L contacting everyone",  # "₹2.21 L"
          delta_color="off")  # grey delta
# Fine print: mean score 0.35 vs real churn 0.20, so scores rank, not probabilities
st.caption("Risk scores rank customers; they are not calibrated probabilities (class weighting pushes them up). "
           "The contact threshold was chosen on training data and is reported here on the test set.")
st.caption(f"Assumptions (editable in churn_model.py): contacting a customer costs ₹{a['contact_cost_inr']:,}, "
           f"a retained churner is worth ₹{a['customer_value_inr']:,}, and the offer retains "  # ₹500, ₹10,000
           f"{a['save_rate']:.0%} of the churners it reaches.")  # 30%

st.markdown("<hr>", unsafe_allow_html=True)  # divider line

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Risk Segment + Geo breakdown
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Portfolio Risk Segmentation</div>',  # styled heading
            unsafe_allow_html=True)  # render HTML

col_a, col_b, col_c = st.columns([1.2, 1.2, 1])  # relative widths, pie a bit narrower

# ── Step 1: customers per segment (no filter: Low 1,079, Medium 459, High 462) ──
seg_counts = (df["risk_segment"].value_counts()  # count per segment
                .reindex(SEG_ORDER, fill_value=0).reset_index())  # fixed order, 0 if empty, 2-col table
seg_counts.columns = ["segment", "count"]  # name the 2 columns

# ── Step 2: bar chart of customers per segment ──
with col_a:  # draw inside the first column
    fig_seg = px.bar(
        seg_counts, x="segment", y="count", color="segment",  # one bar per segment
        color_discrete_map=SEG_COLORS,  # green / amber / red
        title="Customers by Risk Segment",  # chart title
        labels={"count": "Customers", "segment": ""},  # axis labels
    )
    fig_seg.update_layout(**PLOTLY_LAYOUT, showlegend=False)  # dark look, x labels already name segments
    fig_seg.update_traces(marker_line_width=0)  # no bar outlines
    st.plotly_chart(fig_seg, width="stretch")  # fill the column

# ── Step 3: real churn % per predicted segment (no filter: High 60.0%, Medium 16.1%, Low 5.2%) ──
# Checks the score holds up. observed=False does nothing here (column is plain text from CSV).
with col_b:  # second column
    seg_actual = (df.groupby("risk_segment", observed=False)["actual_churn"].mean()  # churn share per segment
                    .reindex(SEG_ORDER).mul(100).round(1).reset_index())  # Low->High order, as %
    fig_cal = px.bar(
        seg_actual, x="risk_segment", y="actual_churn", color="risk_segment",  # one bar per segment
        color_discrete_map=SEG_COLORS,  # green / amber / red
        title="Actual Churn Rate per Predicted Segment (%)",  # chart title
        labels={"actual_churn": "Actual churn (%)", "risk_segment": ""},  # axis labels
    )
    fig_cal.update_layout(**PLOTLY_LAYOUT, showlegend=False)  # dark look, no legend
    fig_cal.update_traces(marker_line_width=0)  # no bar outlines
    st.plotly_chart(fig_cal, width="stretch")  # fill the column

# ── Step 4: donut of the same segment counts ──
with col_c:  # third column
    fig_pie = px.pie(
        seg_counts, names="segment", values="count",  # slice per segment
        color="segment", color_discrete_map=SEG_COLORS,  # green / amber / red
        title="Risk Distribution",  # chart title
        hole=0.55,  # hole = 55% of radius (donut)
    )
    fig_pie.update_layout(**PLOTLY_LAYOUT)  # dark look
    fig_pie.update_traces(textfont_color="#E2E8F0")  # light slice labels
    st.plotly_chart(fig_pie, width="stretch")  # fill the column

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Drivers of churn
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Key Churn Drivers</div>',  # styled heading
            unsafe_allow_html=True)  # render HTML
col_d, col_e = st.columns(2)  # 2 equal columns

# ── Step 1: scatter, age vs risk score, max 1,000 random dots ──
with col_d:  # left column
    fig_scatter = px.scatter(
        df.sample(min(1000, len(df)), random_state=42),  # min() avoids error if fewer rows; same sample each rerun
        x="age", y="churn_prob",  # one dot per customer
        color="risk_segment",  # colour by segment
        color_discrete_map=SEG_COLORS,  # green / amber / red
        category_orders={"risk_segment": SEG_ORDER},  # legend Low, Medium, High
        opacity=0.65,  # see overlapping dots
        title="Age vs Churn Risk Score",  # chart title
        labels={"age": "Age", "churn_prob": f"Churn risk score ({BEST})"},  # axis labels
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT)  # dark look
    st.plotly_chart(fig_scatter, width="stretch")  # fill the column

# ── Step 2: avg risk score per (products, active?) pair ──
# 3 to 4 products tallest (EDA: churn 86%); Inactive taller than Active in every group.
with col_e:  # right column
    prod = (df.groupby(["num_products", "is_active_member"])["churn_prob"].mean()  # avg score per pair
              .mul(100).round(1).reset_index())  # x100, 1 decimal, flat table
    prod["member"] = prod["is_active_member"].map({0: "Inactive", 1: "Active"})  # 0/1 -> words for legend
    fig_prod = px.bar(
        prod, x="num_products", y="churn_prob", color="member", barmode="group",  # group = bars side by side
        color_discrete_map={"Inactive": RED, "Active": GREEN},  # red inactive, green active
        title="Avg Risk Score by Products Held and Activity",  # chart title
        labels={"num_products": "Number of products", "churn_prob": "Avg risk score (x100)", "member": ""},
    )
    fig_prod.update_layout(**PLOTLY_LAYOUT)  # dark look
    st.plotly_chart(fig_prod, width="stretch")  # fill the column

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Feature importance + risk-score histogram
# ═══════════════════════════════════════════════════════════════════════════════
col_f, col_g = st.columns(2)  # 2 equal columns

# ── Step 1: top 10 features from metrics.json (fixed, ignores filters) ──
with col_f:  # left column
    fi = pd.DataFrame(metrics["feature_importance"]).head(10).iloc[::-1]  # reversed so age (0.29) is on top
    fig_fi = px.bar(
        fi, x="importance", y="feature", orientation="h",  # horizontal bars
        color_discrete_sequence=[BLUE],  # all blue
        title=f"Top 10 Features ({BEST})",  # "Top 10 Features (LightGBM)"
        labels={"importance": "Relative importance", "feature": ""},  # axis labels
    )
    fig_fi.update_layout(**PLOTLY_LAYOUT)  # dark look
    st.plotly_chart(fig_fi, width="stretch")  # fill the column

# ── Step 2: histogram of risk scores + red line at threshold 0.38 ──
# Right of the line = in the retention campaign.
with col_g:  # right column
    fig_hist = px.histogram(
        df, x="churn_prob", nbins=40,  # 40 bars
        color_discrete_sequence=[BLUE],  # all blue
        title="Churn Risk Score Distribution",  # chart title
        labels={"churn_prob": f"Risk score — {BEST}"},  # x label
    )
    fig_hist.add_vline(x=biz["optimal_threshold"], line_dash="dash", line_color=RED,  # dashed red line at 0.38
                       annotation_text=f"Contact threshold ({biz['optimal_threshold']:.2f})",  # line label
                       annotation_font_color=RED)  # red label text
    fig_hist.update_layout(**PLOTLY_LAYOUT)  # dark look
    st.plotly_chart(fig_hist, width="stretch")  # fill the column

# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL CUSTOMER LOOK-UP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)  # divider line
st.markdown('<div class="section-title">🔍 Individual Customer Risk Look-up</div>',  # styled heading
            unsafe_allow_html=True)  # render HTML

# ── Step 1: rank customers, riskiest first (rank 1 = 15703793, score 0.997) ──
ranked   = df.sort_values("churn_prob", ascending=False).reset_index(drop=True)  # rows renumbered 0, 1, 2...
if len(ranked) > 1:  # several customers: show a slider
    # pick rank 1..N; -1 because rows start at 0
    cust_idx = st.slider("Customer rank (1 = highest risk in current filter)", 1, len(ranked), 1) - 1
else:  # one customer: slider 1 to 1 would crash
    cust_idx = 0  # just that customer
    st.caption("Only one customer matches the current filters.")  # small note
row      = ranked.iloc[cust_idx]  # chosen customer's row
prob     = row["churn_prob"]  # LightGBM score, e.g. 0.997
prob_lr  = row["churn_prob_lr"]  # Logistic Regression score, e.g. 0.897
seg      = row["risk_segment"]  # e.g. "High Risk"

# ── Step 2: segment -> coloured badge HTML (badge-* CSS classes) ──
seg_badge = {
    "High Risk":   "<span class='badge-high'>High Risk</span>",  # red pill
    "Medium Risk": "<span class='badge-medium'>Medium Risk</span>",  # amber pill
    "Low Risk":    "<span class='badge-low'>Low Risk</span>",  # green pill
}.get(str(seg), seg)  # unknown segment -> plain text

c1, c2 = st.columns([1.4, 1])  # wider details card, narrower gauge

# ── Step 3: details card (HTML table); score red > 0.6, amber > 0.3, else green ──
# e.g. rank 1: 15703793, High Risk, 99.70%, 89.70%, 58 / Male / Germany, 4 products, No, 133,745, Yes, Churned
with c1:  # left column
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

# ── Step 4: gauge (speedometer), score 0 to 100; bands match cut-offs 0.30 / 0.60 ──
with c2:  # right column
    gauge = go.Figure(go.Indicator(
        mode  = "gauge+number",  # dial + big number
        value = prob * 100,  # 0.997 -> 99.7
        title = {"text": "Churn Risk Score", "font": {"color": "#E2E8F0", "size": 14}},  # gauge title
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#64748B"},  # 0 to 100 scale
            "bar":   {"color": RED if prob > 0.6 else AMBER if prob > 0.3 else GREEN},  # needle bar colour
            "bgcolor": CARD_BG,  # dark background
            "borderwidth": 0,  # no border
            "steps": [  # background bands
                {"range": [0,  30], "color": "#14532D"},  # dark green = Low
                {"range": [30, 60], "color": "#713F12"},  # brown = Medium
                {"range": [60, 100],"color": "#7F1D1D"},  # dark red = High
            ],
        },
        number = {"suffix": "%", "font": {"color": "#E2E8F0", "size": 32}},  # "99.7%"
    ))
    gauge.update_layout(
        paper_bgcolor=CARD_BG, font_color="#CBD5E1",  # dark look
        height=260, margin=dict(l=30, r=30, t=40, b=10)  # fits beside the card
    )
    st.plotly_chart(gauge, width="stretch")  # fill the column

# ═══════════════════════════════════════════════════════════════════════════════
# DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)  # divider line
st.markdown('<div class="section-title">📋 Scored Customer Data</div>',  # styled heading
            unsafe_allow_html=True)  # render HTML

show_cols = ["customer_id", "churn_prob", "risk_segment", "contact", "actual_churn",  # 13 columns shown
             "age", "geography", "gender", "num_products", "is_active_member",
             "balance", "tenure_years", "credit_score"]
display_df = ranked[show_cols].copy()  # copy keeps `ranked` numeric
display_df["churn_prob"] = display_df["churn_prob"].map("{:.2%}".format)  # 0.99697 -> "99.70%"

st.dataframe(display_df.head(200), width="stretch", height=320)  # first 200 rows only, keeps page fast

# ── Step: download the whole filtered, ranked table ──
csv_bytes = ranked.to_csv(index=False).encode()  # CSV text -> UTF-8 bytes
st.download_button(
    "⬇️  Download Filtered Data (CSV)", data=csv_bytes,  # button text + file bytes
    file_name="churn_scored_filtered.csv", mime="text/csv"  # saved file name, file type
)
