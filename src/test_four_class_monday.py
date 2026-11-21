
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

print("Loading four-class Random Forest...")

saved = joblib.load(
    "models/four_class_random_forest.pkl"
)

model = saved["model"]
features = saved["features"]
labels = saved["labels"]

# Find Monday dataset
files = list(
    Path("data").glob("Monday-WorkingHours*.csv")
)

if not files:
    raise FileNotFoundError(
        "Monday dataset not found."
    )

print("Loading Monday dataset...")

df = pd.read_csv(files[0], low_memory=False)

df.columns = df.columns.str.strip()

required = features + ["Label"]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# Clean numeric features
for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[features] = df[features].replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(subset=required).copy()

df["Label"] = (
    df["Label"].astype(str).str.strip()
)

if not df["Label"].eq("BENIGN").all():
    raise ValueError(
        "Unexpected labels in Monday dataset."
    )

print("Cleaned records:", len(df))

# Predict
print("\nRunning predictions...")

predictions = model.predict(df[features])

counts = pd.Series(predictions).value_counts()

print("\n===== FOUR-CLASS MONDAY EVALUATION =====")

for class_id in [0, 1, 2, 3]:
    print(
        f"Predicted {labels[class_id]}:",
        counts.get(class_id, 0)
    )

# Any attack prediction on Monday is a false alarm
false_alarms = (predictions != 0).sum()

false_positive_rate = (
    false_alarms / len(predictions) * 100
)

print("\nTotal records:", len(predictions))
print("False alarms:", false_alarms)

print(
    f"False Positive Rate: "
    f"{false_positive_rate:.4f}%"
)

print(
    f"Benign Classification Accuracy: "
    f"{100 - false_positive_rate:.4f}%"
)

print("\nEvaluation completed!")
