
import pandas as pd
import numpy as np

from pathlib import Path

print("Loading existing multiclass dataset...")

# Existing BENIGN, DDoS, PortScan data
existing = pd.read_csv(
    "data/multiclass_traffic.csv",
    low_memory=False
)

print("Existing shape:", existing.shape)

# Find Thursday Web Attacks CSV
files = list(
    Path("data").glob(
        "Thursday-WorkingHours-Morning-WebAttacks*.csv"
    )
)

if not files:
    raise FileNotFoundError(
        "Thursday Web Attacks CSV not found."
    )

print("\nLoading Thursday Web Attacks dataset...")

web = pd.read_csv(
    files[0],
    low_memory=False
)

web.columns = web.columns.str.strip()

print("Thursday original shape:", web.shape)

# Use exactly the same 78 feature columns
features = [
    col for col in existing.columns
    if col not in ["Label", "Source"]
]

missing = [
    col for col in features + ["Label"]
    if col not in web.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# Convert features to numeric
for col in features:
    web[col] = pd.to_numeric(
        web[col],
        errors="coerce"
    )

web[features] = web[features].replace(
    [np.inf, -np.inf],
    np.nan
)

web = web.dropna(
    subset=features + ["Label"]
).copy()

# Normalize original labels
web["Label"] = (
    web["Label"].astype(str).str.strip()
)

print("\nThursday label counts:")
print(web["Label"].value_counts())

# Keep only the actual Web Attack records.
# Thursday BENIGN records are reserved for
# a separate cross-day evaluation.
web_attacks = web[
    web["Label"].str.startswith(
        "Web Attack",
        na=False
    )
].copy()

# Map all Web Attack types to class 3
web_attacks["Label"] = 3
web_attacks["Source"] = "thursday_web"

# Match existing column order
web_attacks = web_attacks[
    existing.columns
]

# Combine with existing data
combined = pd.concat(
    [existing, web_attacks],
    ignore_index=True
)

# Remove exact duplicates within each source
combined = combined.drop_duplicates(
    subset=features + ["Label", "Source"]
)

# Ensure numeric labels
combined["Label"] = combined["Label"].astype(int)

output = "data/four_class_traffic.csv"

combined.to_csv(
    output,
    index=False
)

print("\n===== FOUR-CLASS DATASET =====")

print("Final shape:", combined.shape)

print("\nClass distribution:")
print(
    combined["Label"]
    .value_counts()
    .sort_index()
)

print("\nSource distribution:")
print(combined["Source"].value_counts())

print("\nDataset saved:", output)
