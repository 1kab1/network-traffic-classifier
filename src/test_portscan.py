
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# Load Random Forest and feature names
saved = joblib.load("models/random_forest.pkl")

model = saved["model"]
features = saved["features"]

# Find PortScan dataset
files = list(
    Path("data").glob(
        "Friday-WorkingHours-Afternoon-PortScan*.csv"
    )
)

if not files:
    raise FileNotFoundError(
        "PortScan CSV not found in data folder."
    )

print("Loading PortScan dataset...")

df = pd.read_csv(files[0], low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Check expected columns
required = features + ["Label"]

missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Convert features to numeric
for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid values
df[features] = df[features].replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(subset=required).copy()

# Normalize labels
df["Label"] = df["Label"].astype(str).str.strip()

print("\nActual labels:")
print(df["Label"].value_counts())

# Make predictions
print("\nRunning predictions...")

predictions = model.predict(df[features])

# Show model behavior for each actual class
df["Prediction"] = predictions

summary = pd.crosstab(
    df["Label"],
    df["Prediction"]
)

summary = summary.reindex(
    columns=[0, 1],
    fill_value=0
)

summary.columns = [
    "Predicted BENIGN",
    "Predicted DDoS"
]

print("\n===== PORTSCAN EVALUATION =====")
print(summary)

# Calculate how often PortScan is flagged
attack_rows = df["Label"].str.contains(
    "PortScan",
    case=False,
    na=False
)

if attack_rows.any():
    attack_predictions = df.loc[
        attack_rows, "Prediction"
    ]

    flagged_rate = (
        attack_predictions.eq(1).mean() * 100
    )

    print(
        f"\nPortScan flows flagged as DDoS: "
        f"{flagged_rate:.4f}%"
    )

print("\nPortScan evaluation completed!")
