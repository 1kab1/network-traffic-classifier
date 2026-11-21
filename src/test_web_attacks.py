
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

print("Loading multiclass model...")

saved = joblib.load(
    "models/multiclass_random_forest.pkl"
)

model = saved["model"]
features = saved["features"]

# Find Thursday Web Attacks dataset
files = list(
    Path("data").glob(
        "Thursday-WorkingHours-Morning-WebAttacks*.csv"
    )
)

if not files:
    raise FileNotFoundError(
        "Thursday Web Attacks CSV not found."
    )

print("Loading Thursday dataset...")

df = pd.read_csv(files[0], low_memory=False)

print("Original shape:", df.shape)

# Normalize column names
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

# Normalize labels
df["Label"] = (
    df["Label"].astype(str).str.strip()
)

print("Cleaned shape:", df.shape)

print("\nActual labels:")
print(df["Label"].value_counts())

# Predict using the trained model
print("\nRunning predictions...")

predictions = model.predict(df[features])

df["Prediction"] = predictions

# Show predictions for each actual label
summary = pd.crosstab(
    df["Label"],
    df["Prediction"]
)

summary = summary.reindex(
    columns=[0, 1, 2],
    fill_value=0
)

summary.columns = [
    "Predicted BENIGN",
    "Predicted DDoS",
    "Predicted PortScan"
]

print("\n===== WEB ATTACK EVALUATION =====")
print(summary.to_string())

# Benign traffic: measure false alarms
benign = df["Label"] == "BENIGN"

if benign.any():

    benign_predictions = df.loc[
        benign, "Prediction"
    ]

    false_alarms = (
        benign_predictions != 0
    ).sum()

    false_positive_rate = (
        false_alarms /
        len(benign_predictions) * 100
    )

    print("\nBenign records:", benign.sum())
    print("Benign false alarms:", false_alarms)

    print(
        f"Benign false-positive rate: "
        f"{false_positive_rate:.4f}%"
    )

# Web attacks: measure suspicious traffic flagged
attacks = df["Label"] != "BENIGN"

if attacks.any():

    attack_predictions = df.loc[
        attacks, "Prediction"
    ]

    flagged = (
        attack_predictions != 0
    ).sum()

    flagged_rate = (
        flagged /
        len(attack_predictions) * 100
    )

    print("\nWeb attack records:", attacks.sum())
    print("Web attacks flagged:", flagged)

    print(
        f"Web attack flagged rate: "
        f"{flagged_rate:.4f}%"
    )

print("\nEvaluation completed!")
