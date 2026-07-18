"""
data_generator.py
-----------------
Generates 10,000+ synthetic banking customer records with realistic
behavioral patterns and a churn label.

Features engineered:
  - Demographics        : age, tenure, geography
  - Product usage       : credit_score, balance, num_products, has_credit_card
  - Behavioural signals : spend_trend, complaint_frequency, reward_redemption_rate,
                          monthly_transactions, inactive_months
  - Churn label         : binary (1 = churned, 0 = retained)

Run:
    python data_generator.py
Output:
    data/banking_customers.csv
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
N = 10_500          # total records

# ── Output path ────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
OUTPUT = "data/banking_customers.csv"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n: int = N) -> pd.DataFrame:
    """Build a synthetic dataset that mimics real banking churn dynamics."""

    # ── Demographics ──────────────────────────────────────────────────────────
    age = np.random.randint(18, 75, n)
    tenure = np.random.randint(0, 15, n)                    # years with bank
    geography = np.random.choice(
        ["India", "USA", "Germany", "France"], n,
        p=[0.40, 0.25, 0.20, 0.15]
    )

    # ── Financial profile ─────────────────────────────────────────────────────
    credit_score = np.clip(np.random.normal(650, 80, n), 300, 850).astype(int)
    balance = np.abs(np.random.normal(76_000, 62_000, n)).round(2)
    num_products = np.random.choice([1, 2, 3, 4], n, p=[0.50, 0.35, 0.10, 0.05])
    has_credit_card = np.random.choice([0, 1], n, p=[0.30, 0.70])
    salary = np.clip(np.random.normal(100_000, 48_000, n), 18_000, 350_000).round(2)

    # ── Behavioural signals ───────────────────────────────────────────────────
    # spend_trend: +ve = increasing spend, -ve = decreasing (churn signal)
    spend_trend = np.clip(np.random.normal(0.05, 0.25, n), -1, 1).round(4)

    # complaint_frequency: avg complaints per year (higher → more likely to churn)
    complaint_frequency = np.abs(np.random.exponential(0.8, n)).round(2)

    # reward_redemption_rate: % of earned rewards redeemed (low → disengaged)
    reward_redemption_rate = np.clip(np.random.beta(2, 3, n), 0, 1).round(4)

    # monthly_transactions: average monthly txns in last 6 months
    monthly_transactions = np.abs(np.random.poisson(12, n))

    # inactive_months: consecutive months with no primary product activity
    inactive_months = np.random.choice(
        range(0, 13), n,
        p=[0.35, 0.20, 0.12, 0.09, 0.07, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005]
    )

    is_active_member = np.random.choice([0, 1], n, p=[0.35, 0.65])

    # ── Churn label (logistic probability) ────────────────────────────────────
    # Weights derived from domain knowledge
    logit = (
        -2.5                                           # base intercept
        + 0.02  * (age - 40)                           # older → slightly higher
        - 0.003 * (credit_score - 650)                 # lower score → risk
        - 0.5   * (tenure / 10)                        # longer tenure → loyal
        - 0.8   * np.log1p(balance / 50_000)           # higher balance → loyal
        + 0.6   * (num_products == 1).astype(float)    # single product → risk
        - 0.4   * is_active_member                     # active → loyal
        - 2.0   * spend_trend                          # declining spend → churn
        + 1.2   * complaint_frequency                  # complaints → churn
        - 1.5   * reward_redemption_rate               # engaged → loyal
        - 0.05  * monthly_transactions                 # frequent txns → loyal
        + 0.15  * inactive_months                      # inactivity → churn
        + 0.5   * (geography == "Germany").astype(float)
    )

    churn_prob = sigmoid(logit)
    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    df = pd.DataFrame({
        "customer_id":            [f"CUST{str(i).zfill(6)}" for i in range(1, n + 1)],
        "age":                    age,
        "tenure_years":           tenure,
        "geography":              geography,
        "credit_score":           credit_score,
        "balance":                balance,
        "num_products":           num_products,
        "has_credit_card":        has_credit_card,
        "estimated_salary":       salary,
        "spend_trend":            spend_trend,
        "complaint_frequency":    complaint_frequency,
        "reward_redemption_rate": reward_redemption_rate,
        "monthly_transactions":   monthly_transactions,
        "inactive_months":        inactive_months,
        "is_active_member":       is_active_member,
        "churn":                  churn,
    })

    return df


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    print("[INFO] Generating synthetic banking dataset ...")
    df = generate_dataset()
    df.to_csv(OUTPUT, index=False)

    total = len(df)
    churned = df["churn"].sum()
    print(f"[DONE] Dataset saved -> {OUTPUT}")
    print(f"    Rows      : {total:,}")
    print(f"    Churned   : {churned:,}  ({churned/total*100:.1f}%)")
    print(f"    Retained  : {total - churned:,}  ({(total-churned)/total*100:.1f}%)")
    print("\nSample rows:")
    print(df.head(3).to_string(index=False))
