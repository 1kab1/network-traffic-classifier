
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import confusion_matrix

# Load trained Random Forest
print("Loading Random Forest model...")

saved = joblib.load("models/random_forest.pkl")

model = saved["model"]
features = saved["features"]

# Locate Monday dataset
files = list(
    Path("data").glob("Monday-WorkingHours*.csv")
)

if not files:
    raise FileNotFoundError(
        "Monday dataset not found in data folder."
    )

# Load Monday data
print("Loading Monday dataset...")

df = pd.read_csv(files[0], low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# Check required columns
required = features + ["Label"]

missing_columns = [
    col for col in required
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# Convert feature columns to numeric
for col in features:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Replace infinite values
df[features] = df[features].replace(
    [np.inf, -np.inf],
    np.nan
)

# Remove invalid rows
df = df.dropna(subset=required).copy()

# Clean labels
df["Label"] = df["Label"].astype(str).str.strip()

print("Cleaned shape:", df.shape)

print("\nActual labels:")
print(df["Label"].value_counts())

# Monday should contain only benign records
if not df["Label"].eq("BENIGN").all():
    raise ValueError(
        "Unexpected labels in Monday dataset."
    )

# Select features in training order
X = df[features]

# Predict traffic categories
print("\nRunning predictions...")

predictions = model.predict(X)

# Monday labels are all BENIGN (0)
actual = np.zeros(len(df), dtype=int)

tn, fp, fn, tp = confusion_matrix(
    actual,
    predictions,
    labels=[0, 1]
).ravel()

false_positive_rate = (
    fp / (tn + fp)
    if (tn + fp) > 0
    else 0
)

print("\n===== MONDAY EVALUATION =====")

print("Total evaluated records:", len(df))
print("Correctly classified BENIGN:", tn)
print("Incorrectly flagged as DDoS:", fp)

print(
    f"False Positive Rate: "
    f"{false_positive_rate * 100:.4f}%"
)

print(
    f"Benign Classification Accuracy: "
    f"{(1 - false_positive_rate) * 100:.4f}%"
)

print("\nMonday evaluation completed!")
