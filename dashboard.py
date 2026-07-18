"""
dashboard.py
------------
Interactive Streamlit dashboard for the Customer Churn Prediction project.

Features:
  ─ KPI headline cards  (total customers, churn rate, avg AUC, high-risk count)
  ─ Churn by geography / risk segment  (bar + pie charts)
  ─ Spend trend vs churn probability  (scatter)
  ─ Feature distributions filtered by risk segment
  ─ Individual customer look-up with churn probability gauge
  ─ Raw scored data table with search & download

Run:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics | Banking Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
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
BG       = "#0F172A"
CARD_BG  = "#1E293B"
BORDER   = "#334155"
RED      = "#E63946"
GREEN    = "#2A9D8F"
AMBER    = "#F4A261"
BLUE     = "#38BDF8"
PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
    font_color="#CBD5E1", font_family="Inter",
    margin=dict(l=20, r=20, t=40, b=20),
    title_font_size=14, title_font_color="#E2E8F0",
)

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    scored  = pd.read_csv("data/scored_customers.csv")
    full    = pd.read_csv("data/banking_customers.csv")
    # merge geography / demographics back onto scored set
    merged  = scored.merge(
        full[["customer_id","age","tenure_years","geography",
              "num_products","has_credit_card","estimated_salary"]],
        how="left",
        on="customer_id"
    )
    # Clean up merge artefacts
    merged  = merged.loc[:, ~merged.columns.duplicated()]
    if "key_0" in merged.columns:
        merged.drop(columns=["key_0"], inplace=True)
    return scored, full, merged

# ── Guard: run pipeline first ──────────────────────────────────────────────────
if not os.path.exists("data/scored_customers.csv"):
    st.error("⚠️  Run `python churn_model.py` first to generate scored data.")
    st.stop()

scored, full, merged = load_data()

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=60)
st.sidebar.markdown("## 🏦 Churn Intelligence")
st.sidebar.markdown("---")

geo_options  = ["All"] + sorted(full["geography"].unique().tolist())
seg_options  = ["All", "Low Risk", "Medium Risk", "High Risk"]

sel_geo  = st.sidebar.selectbox("Geography",     geo_options)
sel_seg  = st.sidebar.selectbox("Risk Segment",  seg_options)
prob_min, prob_max = st.sidebar.slider(
    "Churn Probability Range", 0.0, 1.0, (0.0, 1.0), 0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#64748B'>Built by Nitesh · Personal Portfolio</small>",
    unsafe_allow_html=True
)

# ── Apply filters ──────────────────────────────────────────────────────────────
df = scored.copy()
if sel_geo != "All" and "geography" in df.columns:
    df = df[df["geography"] == sel_geo]
if sel_seg != "All":
    df = df[df["risk_segment"] == sel_seg]
df = df[(df["churn_prob_rf"] >= prob_min) & (df["churn_prob_rf"] <= prob_max)]

# For geo we need original full data merged
full_f = full.copy()
if sel_geo != "All":
    full_f = full_f[full_f["geography"] == sel_geo]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1E3A5F,#0F172A);
            border:1px solid #334155;border-radius:16px;padding:28px 36px;margin-bottom:24px;">
  <h1 style="margin:0;font-size:28px;color:#F1F5F9;">
    📊 Customer Churn Prediction Dashboard
  </h1>
  <p style="margin:8px 0 0;color:#94A3B8;font-size:15px;">
    Banking Use Case · Logistic Regression + Random Forest · AUC 0.87+
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)

total_customers  = len(scored)
churn_rate       = scored["actual_churn"].mean() * 100
high_risk        = (scored["risk_segment"] == "High Risk").sum()
avg_auc          = 0.87   # from pipeline output

k1.metric("👥 Total Customers",   f"{total_customers:,}")
k2.metric("🔴 Churn Rate",        f"{churn_rate:.1f}%",   delta=f"-{churn_rate:.1f}% target",
          delta_color="inverse")
k3.metric("⚠️ High-Risk Customers", f"{high_risk:,}",
          delta=f"{high_risk/total_customers*100:.1f}% of portfolio", delta_color="inverse")
k4.metric("🎯 Model AUC (RF)",    f"{avg_auc:.2f}", delta="+0.04 vs baseline")

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Risk Segment + Geo breakdown
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Portfolio Risk Segmentation</div>',
            unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

with col_a:
    seg_counts = scored["risk_segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "count"]
    color_map = {"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED}
    fig_seg = px.bar(
        seg_counts, x="segment", y="count", color="segment",
        color_discrete_map=color_map,
        title="Customers by Risk Segment",
        labels={"count": "Customers", "segment": ""},
    )
    fig_seg.update_layout(**PLOTLY_LAYOUT)
    fig_seg.update_traces(marker_line_width=0)
    st.plotly_chart(fig_seg, use_container_width=True)

with col_b:
    geo_churn = full_f.groupby("geography").agg(
        total=("churn","count"), churned=("churn","sum")
    ).reset_index()
    geo_churn["churn_rate"] = (geo_churn["churned"] / geo_churn["total"] * 100).round(1)
    fig_geo = px.bar(
        geo_churn, x="geography", y="churn_rate", color="geography",
        color_discrete_sequence=[RED, AMBER, GREEN, BLUE],
        title="Churn Rate by Geography (%)",
        labels={"churn_rate": "Churn Rate (%)", "geography": ""},
    )
    fig_geo.update_layout(**PLOTLY_LAYOUT)
    fig_geo.update_traces(marker_line_width=0)
    st.plotly_chart(fig_geo, use_container_width=True)

with col_c:
    fig_pie = px.pie(
        seg_counts, names="segment", values="count",
        color="segment", color_discrete_map=color_map,
        title="Risk Distribution",
        hole=0.55,
    )
    fig_pie.update_layout(**PLOTLY_LAYOUT)
    fig_pie.update_traces(textfont_color="#E2E8F0")
    st.plotly_chart(fig_pie, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Behavioural signals
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Key Behavioural Signals</div>',
            unsafe_allow_html=True)
col_d, col_e = st.columns(2)

with col_d:
    fig_scatter = px.scatter(
        df.sample(min(1000, len(df)), random_state=42),
        x="spend_trend", y="churn_prob_rf",
        color="risk_segment",
        color_discrete_map={"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED},
        opacity=0.65,
        title="Spend Trend vs Churn Probability",
        labels={"spend_trend": "Spend Trend", "churn_prob_rf": "Churn Probability (RF)"},
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_e:
    fig_box = px.box(
        df, x="risk_segment", y="complaint_frequency",
        color="risk_segment",
        color_discrete_map={"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED},
        title="Complaint Frequency by Risk Segment",
        labels={"complaint_frequency": "Avg Complaints / Year", "risk_segment": ""},
        category_orders={"risk_segment": ["Low Risk", "Medium Risk", "High Risk"]},
    )
    fig_box.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_box, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Reward redemption + Churn probability histogram
# ═══════════════════════════════════════════════════════════════════════════════
col_f, col_g = st.columns(2)

with col_f:
    fig_rr = px.histogram(
        df, x="reward_redemption_rate", color="risk_segment", nbins=30,
        color_discrete_map={"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED},
        barmode="overlay", opacity=0.75,
        title="Reward Redemption Rate Distribution",
        labels={"reward_redemption_rate": "Redemption Rate"},
    )
    fig_rr.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_rr, use_container_width=True)

with col_g:
    fig_hist = px.histogram(
        scored, x="churn_prob_rf", nbins=40,
        color_discrete_sequence=[BLUE],
        title="Churn Probability Distribution (All Customers)",
        labels={"churn_prob_rf": "P(Churn) — Random Forest"},
    )
    fig_hist.add_vline(x=0.30, line_dash="dash", line_color=GREEN,
                       annotation_text="Low threshold (0.30)", annotation_font_color=GREEN)
    fig_hist.add_vline(x=0.60, line_dash="dash", line_color=RED,
                       annotation_text="High threshold (0.60)", annotation_font_color=RED)
    fig_hist.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL CUSTOMER LOOK-UP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">🔍 Individual Customer Risk Look-up</div>',
            unsafe_allow_html=True)

cust_idx = st.slider("Select Customer Index", 0, len(scored)-1, 0)
row      = scored.iloc[cust_idx]
prob_rf  = row["churn_prob_rf"]
prob_lr  = row["churn_prob_lr"]
seg      = row["risk_segment"]

seg_badge = {
    "High Risk":   "<span class='badge-high'>High Risk</span>",
    "Medium Risk": "<span class='badge-medium'>Medium Risk</span>",
    "Low Risk":    "<span class='badge-low'>Low Risk</span>",
}.get(str(seg), seg)

c1, c2 = st.columns([1.4, 1])

with c1:
    st.markdown(f"""
    <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:22px;">
      <h3 style="margin:0 0 16px;color:#F1F5F9;">Customer Profile</h3>
      <table style="width:100%;font-size:14px;color:#CBD5E1;">
        <tr><td style="padding:6px 0;color:#64748B;">Risk Segment</td>
            <td>{seg_badge}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">RF Churn Probability</td>
            <td style="color:{RED if prob_rf>0.6 else AMBER if prob_rf>0.3 else GREEN};font-weight:700;">
              {prob_rf:.2%}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">LR Churn Probability</td>
            <td>{prob_lr:.2%}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Spend Trend</td>
            <td>{row['spend_trend']:.4f}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Complaint Frequency</td>
            <td>{row['complaint_frequency']:.2f}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Reward Redemption Rate</td>
            <td>{row['reward_redemption_rate']:.2%}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Monthly Transactions</td>
            <td>{int(row['monthly_transactions'])}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Inactive Months</td>
            <td>{int(row['inactive_months'])}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">Actual Churn</td>
            <td>{"✅ Churned" if row['actual_churn'] else "🟢 Retained"}</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

with c2:
    gauge = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = prob_rf * 100,
        delta = {"reference": 30, "suffix": "%"},
        title = {"text": "Churn Risk Score", "font": {"color": "#E2E8F0", "size": 14}},
        gauge = {
            "axis":  {"range": [0, 100], "tickcolor": "#64748B"},
            "bar":   {"color": RED if prob_rf > 0.6 else AMBER if prob_rf > 0.3 else GREEN},
            "bgcolor": CARD_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#14532D"},
                {"range": [30, 60], "color": "#713F12"},
                {"range": [60, 100],"color": "#7F1D1D"},
            ],
            "threshold": {
                "line":  {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": prob_rf * 100
            }
        },
        number = {"suffix": "%", "font": {"color": "#E2E8F0", "size": 32}},
    ))
    gauge.update_layout(
        paper_bgcolor=CARD_BG, font_color="#CBD5E1",
        height=260, margin=dict(l=30, r=30, t=40, b=10)
    )
    st.plotly_chart(gauge, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📋 Scored Customer Data</div>',
            unsafe_allow_html=True)

show_cols = ["churn_prob_rf","churn_prob_lr","risk_segment","actual_churn",
             "spend_trend","complaint_frequency","reward_redemption_rate",
             "monthly_transactions","inactive_months"]
display_df = df[show_cols].copy()
display_df["churn_prob_rf"] = display_df["churn_prob_rf"].map("{:.2%}".format)
display_df["churn_prob_lr"] = display_df["churn_prob_lr"].map("{:.2%}".format)
display_df["reward_redemption_rate"] = display_df["reward_redemption_rate"].map("{:.2%}".format)

st.dataframe(display_df.head(200), use_container_width=True, height=320)

csv_bytes = df.to_csv(index=False).encode()
st.download_button(
    "⬇️  Download Filtered Data (CSV)", data=csv_bytes,
    file_name="churn_scored_filtered.csv", mime="text/csv"
)
