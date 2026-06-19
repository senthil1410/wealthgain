import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import json

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
CATEGORIES = {
    "food":          ["Swiggy", "Zomato", "McDonald's", "Dominos", "KFC", "Subway"],
    "groceries":     ["BigBasket", "Blinkit", "DMart", "Reliance Fresh"],
    "transport":     ["Ola", "Uber", "Rapido", "Namma Metro", "Indian Oil"],
    "shopping":      ["Amazon", "Flipkart", "Myntra", "Ajio", "Nykaa"],
    "utilities":     ["BESCOM", "Airtel", "Jio", "BWSSB", "LPG Booking"],
    "healthcare":    ["Apollo Pharmacy", "Medplus", "Practo", "1mg"],
    "entertainment": ["Netflix", "Hotstar", "Spotify", "PVR Cinemas"],
    "education":     ["Udemy", "Coursera", "Unacademy"],
    "rent":          ["House Rent", "PG Rent", "Society Maintenance"],
    "salary":        ["Salary Credit", "Bonus Credit", "Freelance Payment"],
    "savings":       ["SIP Transfer", "RD Transfer", "FD Deposit"],
}

AMOUNT_RANGES = {
    "food":          (80,  900),
    "groceries":     (300, 4000),
    "transport":     (30,  600),
    "shopping":      (300, 10000),
    "utilities":     (100, 2500),
    "healthcare":    (50,  4000),
    "entertainment": (100, 1500),
    "education":     (500, 6000),
    "rent":          (8000, 25000),
    "salary":        (50000, 90000),
    "savings":       (2000, 15000),
}

PROFILES = [
    {
        "user_id":        "USR001",
        "name":           "Arjun Sharma",
        "monthly_income": 75000,
        "savings_goal":   200000,
        "goal_label":     "Emergency Fund",
        "goal_months":    12,
        "risk_profile":   "moderate",
    },
    {
        "user_id":        "USR002",
        "name":           "Priya Nair",
        "monthly_income": 95000,
        "savings_goal":   500000,
        "goal_label":     "Home Down Payment",
        "goal_months":    24,
        "risk_profile":   "aggressive",
    },
]

def generate_transactions(user_id: str, monthly_income: float, n_months: int = 6):
    transactions = []
    start_date = datetime.now() - timedelta(days=n_months * 30)

    for day_offset in range(n_months * 30):
        current_date = start_date + timedelta(days=day_offset)

        # Salary on 1st
        if current_date.day == 1:
            transactions.append({
                "user_id":     user_id,
                "date":        current_date.strftime("%Y-%m-%d"),
                "merchant":    random.choice(CATEGORIES["salary"]),
                "category":    "salary",
                "amount":      round(random.uniform(*AMOUNT_RANGES["salary"]), 2),
                "type":        "credit",
                "day_of_week": current_date.strftime("%A"),
                "month":       current_date.month,
            })

        # Rent on 5th
        if current_date.day == 5:
            transactions.append({
                "user_id":     user_id,
                "date":        current_date.strftime("%Y-%m-%d"),
                "merchant":    random.choice(CATEGORIES["rent"]),
                "category":    "rent",
                "amount":      round(random.uniform(*AMOUNT_RANGES["rent"]), 2),
                "type":        "debit",
                "day_of_week": current_date.strftime("%A"),
                "month":       current_date.month,
            })

        # SIP on 10th
        if current_date.day == 10:
            transactions.append({
                "user_id":     user_id,
                "date":        current_date.strftime("%Y-%m-%d"),
                "merchant":    random.choice(CATEGORIES["savings"]),
                "category":    "savings",
                "amount":      round(random.uniform(*AMOUNT_RANGES["savings"]), 2),
                "type":        "debit",
                "day_of_week": current_date.strftime("%A"),
                "month":       current_date.month,
            })

        # Daily transactions
        n_daily = random.randint(2, 5)
        daily_cats = random.choices(
            ["food", "transport", "groceries", "shopping",
             "utilities", "healthcare", "entertainment", "education"],
            weights=[30, 20, 15, 10, 8, 5, 7, 5],
            k=n_daily
        )

        for cat in daily_cats:
            amount = round(random.uniform(*AMOUNT_RANGES[cat]), 2)
            # 2% anomaly injection
            if random.random() < 0.02:
                amount = round(amount * random.uniform(4, 8), 2)

            transactions.append({
                "user_id":     user_id,
                "date":        current_date.strftime("%Y-%m-%d"),
                "merchant":    random.choice(CATEGORIES[cat]),
                "category":    cat,
                "amount":      amount,
                "type":        "debit",
                "day_of_week": current_date.strftime("%A"),
                "month":       current_date.month,
            })

    return transactions

def generate_finance_rules():
    rules = [
        "The 50/30/20 rule: allocate 50% of income to needs, 30% to wants, and 20% to savings.",
        "An emergency fund should cover 3 to 6 months of living expenses.",
        "Pay yourself first — automate savings before spending on discretionary items.",
        "High-interest debt should be paid off before investing.",
        "Diversify investments across equity, debt, and gold to manage risk.",
        "SIP in mutual funds is better than lump sum for volatile markets.",
        "Track every rupee spent — awareness is the first step to control.",
        "Food delivery apps are a major budget leak — cook at home at least 4 days a week.",
        "Subscription services add up — audit them every 3 months.",
        "Transport costs above 10% of income indicate overspending.",
        "Shopping impulse buys can be reduced by a 48-hour wait rule.",
        "Insurance is not an investment — keep term insurance and health insurance separate.",
        "Index funds outperform most actively managed funds over 10+ years.",
        "Rent should not exceed 30% of take-home income.",
        "Avoid lifestyle inflation — increase savings rate with every salary hike.",
        "Tax-saving investments (80C) should be planned in April, not March.",
        "Credit card bills must be paid in full every month to avoid interest.",
        "A financial goal without a deadline is just a wish.",
        "Entertainment spending above 5% of income needs review.",
        "Medical emergencies are the top reason people go into debt — health insurance is non-negotiable.",
        "Groceries bought online tend to be cheaper than impulse purchases at supermarkets.",
        "Utility bills can be reduced 10-20% by simple habit changes.",
        "Education spending is an investment — prioritise skill-building courses.",
        "Compound interest works best when started early — even small SIPs matter.",
        "Review your financial plan every 6 months and adjust for life changes.",
    ]
    return rules

if __name__ == "__main__":
    all_transactions = []
    for profile in PROFILES:
        txns = generate_transactions(profile["user_id"], profile["monthly_income"])
        all_transactions.extend(txns)

    df = pd.DataFrame(all_transactions)
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    df["transaction_id"] = [f"TXN{str(i).zfill(6)}" for i in range(len(df))]
    df.to_csv("data/transactions.csv", index=False)

    # Save profiles
    with open("data/profiles.json", "w") as f:
        json.dump(PROFILES, f, indent=2)

    # Save finance rules
    rules = generate_finance_rules()
    with open("data/finance_rules.txt", "w") as f:
        f.write("\n".join(rules))

    print(f"Generated {len(df)} transactions for {len(PROFILES)} users")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nPer user:")
    for uid, grp in df.groupby("user_id"):
        print(f"  {uid}: {len(grp)} transactions")
    print(f"\nSaved:")
    print(f"  data/transactions.csv")
    print(f"  data/profiles.json")
    print(f"  data/finance_rules.txt")