
import pandas as pd
import numpy as np

from pathlib import Path

data_folder = Path("data")

datasets = {
    "ddos": "Friday-WorkingHours-Afternoon-DDos*.csv",
    "portscan": "Friday-WorkingHours-Afternoon-PortScan*.csv"
}

frames = []

for source, pattern in datasets.items():

    files = list(data_folder.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"Dataset not found: {pattern}"
        )

    print(f"\nLoading {source} dataset...")

    df = pd.read_csv(
        files[0],
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    # Record the source capture
    df["Source"] = source

    frames.append(df)

# Combine both datasets
combined = pd.concat(
    frames,
    ignore_index=True
)

print("\nCombined shape:", combined.shape)

# Normalize labels
combined["Label"] = (
    combined["Label"]
    .astype(str)
    .str.strip()
)

allowed_labels = [
    "BENIGN",
    "DDoS",
    "PortScan"
]

# Keep only expected labels
combined = combined[
    combined["Label"].isin(allowed_labels)
].copy()

# Convert features to numeric
feature_columns = [
    col for col in combined.columns
    if col not in ["Label", "Source"]
]

for col in feature_columns:
    combined[col] = pd.to_numeric(
        combined[col],
        errors="coerce"
    )

# Replace infinite values
combined[feature_columns] = (
    combined[feature_columns]
    .replace([np.inf, -np.inf], np.nan)
)

# Remove invalid rows
combined = combined.dropna(
    subset=feature_columns
)

# Remove exact duplicates within each source
combined = combined.drop_duplicates(
    subset=feature_columns + ["Label", "Source"]
)

# Encode labels
label_mapping = {
    "BENIGN": 0,
    "DDoS": 1,
    "PortScan": 2
}

combined["Label"] = (
    combined["Label"]
    .map(label_mapping)
    .astype(int)
)

# Save combined dataset
output = data_folder / "multiclass_traffic.csv"

combined.to_csv(
    output,
    index=False
)

print("\n===== MULTICLASS DATASET =====")

print("Final shape:", combined.shape)

print("\nClass distribution:")
print(combined["Label"].value_counts().sort_index())

print("\nSource distribution:")
print(combined["Source"].value_counts())

print("\nDataset saved:", output)
