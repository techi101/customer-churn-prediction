"""
dashboard.py
------------
Interactive Streamlit dashboard for the Customer Churn Prediction project.

Features:
  ─ KPI headline cards  (total customers, churn rate, avg AUC, high-risk count)
  ─ Retention campaign economics  (capture, lift, profit-optimal threshold)
  ─ Risk segments vs actual churn  (bar + pie charts)
  ─ Churn drivers: age, products held, activity, feature importance
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
import json

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
    with open("reports/metrics.json") as f:
        metrics = json.load(f)
    return scored, metrics

# ── Guard: run pipeline first ──────────────────────────────────────────────────
if not os.path.exists("data/scored_customers.csv") or not os.path.exists("reports/metrics.json"):
    st.error("⚠️  Run `python churn_model.py` first to generate scored data.")
    st.stop()

scored, metrics = load_data()
BEST     = metrics["best_model"]
BASE     = metrics["baseline_model"]
biz      = metrics["business"]
best_auc = metrics["models"][BEST]["test_auc"]
base_auc = metrics["models"][BASE]["test_auc"]
SEG_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
SEG_COLORS = {"Low Risk": GREEN, "Medium Risk": AMBER, "High Risk": RED}

# ── Sidebar filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏦 Churn Intelligence")
st.sidebar.markdown("---")

geo_options  = ["All"] + sorted(scored["geography"].unique().tolist())
seg_options  = ["All"] + SEG_ORDER

sel_geo  = st.sidebar.selectbox("Geography",     geo_options)
sel_seg  = st.sidebar.selectbox("Risk Segment",  seg_options)
prob_min, prob_max = st.sidebar.slider(
    "Churn Probability Range", 0.0, 1.0, (0.0, 1.0), 0.01
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#64748B'>Scores are for the 2,000-customer held-out test set, "
    "which the model never saw during training.</small>",
    unsafe_allow_html=True
)

# ── Apply filters ──────────────────────────────────────────────────────────────
df = scored.copy()
if sel_geo != "All":
    df = df[df["geography"] == sel_geo]
if sel_seg != "All":
    df = df[df["risk_segment"] == sel_seg]
df = df[(df["churn_prob"] >= prob_min) & (df["churn_prob"] <= prob_max)]

if df.empty:
    st.warning("No customers match these filters.")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
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
k1, k2, k3, k4 = st.columns(4)

k1.metric("👥 Customers (test set)", f"{len(df):,}")
k2.metric("🔴 Actual Churn Rate",   f"{df['actual_churn'].mean()*100:.1f}%")
k3.metric("⚠️ High-Risk Customers", f"{(df['risk_segment'] == 'High Risk').sum():,}")
k4.metric(f"🎯 Model AUC ({BEST})", f"{best_auc:.3f}",
          delta=f"{best_auc - base_auc:+.3f} vs {BASE}")

# ═══════════════════════════════════════════════════════════════════════════════
# RETENTION CAMPAIGN ECONOMICS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Retention Campaign Economics</div>',
            unsafe_allow_html=True)
a = biz["assumptions"]
b1, b2, b3, b4 = st.columns(4)
b1.metric("Churners in top 10% of scores", f"{biz['top10_capture']:.0%}",
          delta=f"{biz['lift_top10']:.1f}x better than random")
b2.metric("Profit-optimal threshold", f"{biz['optimal_threshold']:.2f}")
b3.metric("Customers to contact", f"{biz['customers_contacted']:,}",
          delta=f"reaches {biz['churners_reached']} churners", delta_color="off")
b4.metric("Campaign net value", f"₹{biz['net_value_optimal_inr']/1e5:.2f} L",
          delta=f"vs ₹{biz['net_value_contact_all_inr']/1e5:.2f} L contacting everyone",
          delta_color="off")
st.caption(f"Assumptions (editable in churn_model.py): contacting a customer costs ₹{a['contact_cost_inr']:,}, "
           f"a retained churner is worth ₹{a['customer_value_inr']:,}, and the offer retains "
           f"{a['save_rate']:.0%} of the churners it reaches.")

st.markdown("<hr>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Risk Segment + Geo breakdown
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Portfolio Risk Segmentation</div>',
            unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1.2, 1.2, 1])

seg_counts = (df["risk_segment"].value_counts()
                .reindex(SEG_ORDER, fill_value=0).reset_index())
seg_counts.columns = ["segment", "count"]

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

with col_d:
    fig_scatter = px.scatter(
        df.sample(min(1000, len(df)), random_state=42),
        x="age", y="churn_prob",
        color="risk_segment",
        color_discrete_map=SEG_COLORS,
        category_orders={"risk_segment": SEG_ORDER},
        opacity=0.65,
        title="Age vs Churn Probability",
        labels={"age": "Age", "churn_prob": f"Churn Probability ({BEST})"},
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_scatter, width="stretch")

with col_e:
    prod = (df.groupby(["num_products", "is_active_member"])["churn_prob"].mean()
              .mul(100).round(1).reset_index())
    prod["member"] = prod["is_active_member"].map({0: "Inactive", 1: "Active"})
    fig_prod = px.bar(
        prod, x="num_products", y="churn_prob", color="member", barmode="group",
        color_discrete_map={"Inactive": RED, "Active": GREEN},
        title="Avg Predicted Churn by Products Held and Activity (%)",
        labels={"num_products": "Number of products", "churn_prob": "Avg P(churn) %", "member": ""},
    )
    fig_prod.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_prod, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Feature importance + probability histogram
# ═══════════════════════════════════════════════════════════════════════════════
col_f, col_g = st.columns(2)

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

with col_g:
    fig_hist = px.histogram(
        df, x="churn_prob", nbins=40,
        color_discrete_sequence=[BLUE],
        title="Churn Probability Distribution",
        labels={"churn_prob": f"P(Churn) — {BEST}"},
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

ranked   = df.sort_values("churn_prob", ascending=False).reset_index(drop=True)
cust_idx = st.slider("Customer rank (1 = highest risk in current filter)", 1, len(ranked), 1) - 1
row      = ranked.iloc[cust_idx]
prob     = row["churn_prob"]
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
      <h3 style="margin:0 0 16px;color:#F1F5F9;">Customer {int(row['customer_id'])}</h3>
      <table style="width:100%;font-size:14px;color:#CBD5E1;">
        <tr><td style="padding:6px 0;color:#64748B;">Risk Segment</td>
            <td>{seg_badge}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">{BEST} Churn Probability</td>
            <td style="color:{RED if prob>0.6 else AMBER if prob>0.3 else GREEN};font-weight:700;">
              {prob:.2%}</td></tr>
        <tr><td style="padding:6px 0;color:#64748B;">{BASE} Churn Probability</td>
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

show_cols = ["customer_id", "churn_prob", "risk_segment", "contact", "actual_churn",
             "age", "geography", "gender", "num_products", "is_active_member",
             "balance", "tenure_years", "credit_score"]
display_df = ranked[show_cols].copy()
display_df["churn_prob"] = display_df["churn_prob"].map("{:.2%}".format)

st.dataframe(display_df.head(200), width="stretch", height=320)

csv_bytes = ranked.to_csv(index=False).encode()
st.download_button(
    "⬇️  Download Filtered Data (CSV)", data=csv_bytes,
    file_name="churn_scored_filtered.csv", mime="text/csv"
)
