import json
import pickle
import pandas as pd
import numpy as np
from mcp.server.fastmcp import FastMCP
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Then change these lines:
df_all = pd.read_csv(os.path.join(BASE_DIR, "data", "transactions.csv"))

with open(os.path.join(BASE_DIR, "models", "classifier.pkl"), "rb") as f:
    clf_bundle  = pickle.load(f)
with open(os.path.join(BASE_DIR, "models", "anomaly_detector.pkl"), "rb") as f:
    anom_bundle = pickle.load(f)
with open(os.path.join(BASE_DIR, "models", "expense_predictor.pkl"), "rb") as f:
    pred_bundle = pickle.load(f)
    
mcp = FastMCP("WealthGain")

# Load models and data once
with open("models/classifier.pkl", "rb") as f:
    clf_bundle  = pickle.load(f)
with open("models/anomaly_detector.pkl", "rb") as f:
    anom_bundle = pickle.load(f)
with open("models/expense_predictor.pkl", "rb") as f:
    pred_bundle = pickle.load(f)

clf         = clf_bundle["model"]
le_merchant = clf_bundle["le_merchant"]
le_day      = clf_bundle["le_day"]
anom        = anom_bundle["model"]
regs        = pred_bundle["model"]

df_all = pd.read_csv("data/transactions.csv")

def safe_encode(le, val):
    return le.transform([val])[0] if val in le.classes_ else 0

@mcp.tool()
def get_spending_summary(user_id: str) -> str:
    """Get total spending by category for a user."""
    user_df  = df_all[(df_all["user_id"] == user_id) & (df_all["type"] == "debit")]
    by_cat   = user_df.groupby("category")["amount"].sum().sort_values(ascending=False)
    total    = user_df["amount"].sum()
    income   = df_all[(df_all["user_id"] == user_id) & (df_all["type"] == "credit")]["amount"].sum()

    result = {
        "user_id":        user_id,
        "total_spent":    round(total, 2),
        "total_income":   round(income, 2),
        "savings_rate":   round((income - total) / income * 100, 1) if income > 0 else 0,
        "by_category":    {cat: round(amt, 2) for cat, amt in by_cat.items()},
        "top_category":   by_cat.index[0] if len(by_cat) > 0 else "none",
    }
    return json.dumps(result)

@mcp.tool()
def get_savings_rate(user_id: str) -> str:
    """Calculate actual vs recommended savings rate for a user."""
    user_df  = df_all[df_all["user_id"] == user_id]
    income   = user_df[user_df["type"] == "credit"]["amount"].sum()
    spent    = user_df[user_df["type"] == "debit"]["amount"].sum()
    saved    = user_df[user_df["category"] == "savings"]["amount"].sum()

    actual_rate      = round(saved / income * 100, 1) if income > 0 else 0
    recommended_rate = 20.0
    gap              = round(recommended_rate - actual_rate, 1)

    result = {
        "user_id":          user_id,
        "total_income":     round(income, 2),
        "total_spent":      round(spent, 2),
        "total_saved":      round(saved, 2),
        "actual_rate_pct":  actual_rate,
        "recommended_pct":  recommended_rate,
        "gap_pct":          gap,
        "status":           "on track" if gap <= 0 else f"{gap}% below target",
    }
    return json.dumps(result)

@mcp.tool()
def flag_overspending(user_id: str) -> str:
    """Detect categories where user is overspending based on income %."""
    user_df = df_all[(df_all["user_id"] == user_id) & (df_all["type"] == "debit")]
    income  = df_all[(df_all["user_id"] == user_id) & (df_all["type"] == "credit")]["amount"].sum()
    by_cat  = user_df.groupby("category")["amount"].sum()

    thresholds = {
        "food":          15,
        "shopping":      10,
        "entertainment":  5,
        "transport":     10,
        "rent":          30,
        "utilities":      8,
    }

    flags = []
    for cat, threshold in thresholds.items():
        if cat in by_cat:
            pct = round(by_cat[cat] / income * 100, 1)
            if pct > threshold:
                flags.append({
                    "category":    cat,
                    "spent":       round(by_cat[cat], 2),
                    "pct_income":  pct,
                    "threshold":   threshold,
                    "overspend":   round(pct - threshold, 1),
                })

    # Detect anomalies
    user_debits = user_df.copy()
    user_debits["merchant_encoded"] = user_debits["merchant"].apply(
        lambda x: safe_encode(le_merchant, x))
    user_debits["day_encoded"] = user_debits["day_of_week"].apply(
        lambda x: safe_encode(le_day, x))

    X_anom = user_debits[["amount", "merchant_encoded", "day_encoded", "month"]]
    user_debits["is_anomaly"] = anom.predict(X_anom) == -1
    anomalies = user_debits[user_debits["is_anomaly"]][
        ["date", "merchant", "category", "amount"]
    ].to_dict("records")

    return json.dumps({
        "user_id":   user_id,
        "flags":     flags,
        "anomalies": anomalies[:5],
    })

@mcp.tool()
def project_goal_timeline(user_id: str, goal_amount: float, monthly_savings: float) -> str:
    """Project how many months to reach a savings goal."""
    if monthly_savings <= 0:
        return json.dumps({"error": "Monthly savings must be positive"})

    months_needed = goal_amount / monthly_savings
    import json as _json
    from datetime import datetime, timedelta

    target_date = datetime.now() + timedelta(days=months_needed * 30)

    result = {
        "user_id":        user_id,
        "goal_amount":    goal_amount,
        "monthly_savings": monthly_savings,
        "months_needed":  round(months_needed, 1),
        "target_date":    target_date.strftime("%B %Y"),
        "advice":         "Increase monthly savings to reach goal faster." if months_needed > 24 else "On track.",
    }
    return json.dumps(result)

if __name__ == "__main__":
    mcp.run()