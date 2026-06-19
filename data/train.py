import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample

df = pd.read_csv("data/transactions.csv")
print(f"Loaded {len(df)} transactions\n")

# ─────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────
le_merchant = LabelEncoder()
le_day      = LabelEncoder()

df["merchant_encoded"] = le_merchant.fit_transform(df["merchant"])
df["day_encoded"]      = le_day.fit_transform(df["day_of_week"])

# ─────────────────────────────────────────
# MODEL 1 — Category Classifier
# Fix class imbalance with oversampling
# ─────────────────────────────────────────
print("=" * 50)
print("MODEL 1: Category Classifier (with oversampling)")
print("=" * 50)

features = ["merchant_encoded", "amount", "day_encoded", "month"]
X = df[features]
y = df["category"]

# Oversample minority classes to fix class imbalance
df_train = df.copy()
max_count = df_train["category"].value_counts().max()
oversampled = []
for cat in df_train["category"].unique():
    cat_df = df_train[df_train["category"] == cat]
    if len(cat_df) < max_count // 2:
        cat_df = resample(cat_df, replace=True,
                         n_samples=max_count // 2,
                         random_state=42)
    oversampled.append(cat_df)

df_balanced = pd.concat(oversampled).sample(frac=1, random_state=42)
X_bal = df_balanced[features]
y_bal = df_balanced["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred   = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────
# MODEL 2 — Anomaly Detector
# ─────────────────────────────────────────
print("=" * 50)
print("MODEL 2: Anomaly Detector")
print("=" * 50)

debits         = df[df["type"] == "debit"].copy()
X_anom         = debits[["amount", "merchant_encoded", "day_encoded", "month"]]
anom           = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
anom.fit(X_anom)
debits["is_anomaly"] = anom.predict(X_anom) == -1
print(f"Anomalies detected: {debits['is_anomaly'].sum()} / {len(debits)}")

# ─────────────────────────────────────────
# MODEL 3 — Expense Predictor (per user)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL 3: Monthly Expense Predictor")
print("=" * 50)

regs = {}
for uid in df["user_id"].unique():
    user_debits = df[(df["user_id"] == uid) & (df["type"] == "debit")]
    monthly     = user_debits.groupby("month")["amount"].sum().reset_index()
    monthly.columns = ["month", "total"]

    reg = LinearRegression()
    reg.fit(monthly[["month"]], monthly["total"])
    regs[uid] = reg

    next_month = int(monthly["month"].max()) + 1
    if next_month > 12:
        next_month = 1
    predicted = reg.predict([[next_month]])[0]
    print(f"  {uid}: predicted month {next_month} spending = ₹{predicted:,.0f}")

# ─────────────────────────────────────────
# Save models
# ─────────────────────────────────────────
print("\nSaving models...")
with open("models/classifier.pkl", "wb") as f:
    pickle.dump({"model": clf, "le_merchant": le_merchant, "le_day": le_day}, f)

with open("models/anomaly_detector.pkl", "wb") as f:
    pickle.dump({"model": anom, "le_merchant": le_merchant, "le_day": le_day}, f)

with open("models/expense_predictor.pkl", "wb") as f:
    pickle.dump({"model": regs}, f)

print("✅ Models saved to models/")