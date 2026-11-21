
import pandas as pd
import numpy as np
from pathlib import Path

# Step 1: Find our original dataset
data_folder = Path("data")

files = list(
    data_folder.glob("Friday-WorkingHours-Afternoon-DDos*.csv")
)

if not files:
    raise FileNotFoundError("DDoS dataset not found!")

file_path = files[0]

# Step 2: Load the dataset
print("Loading original dataset...")

df = pd.read_csv(file_path, low_memory=False)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# Step 3: Remove duplicate records
df = df.drop_duplicates().copy()

print("After removing duplicates:", df.shape)

# Step 4: Convert feature columns to numbers
feature_columns = df.columns.drop("Label")

for column in feature_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Step 5: Replace infinite values with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Step 6: Remove rows with missing values
df = df.dropna(subset=feature_columns.tolist() + ["Label"])

print("After removing invalid rows:", df.shape)

# Step 7: Convert labels into binary numbers
df["Label"] = df["Label"].str.strip().map({
    "BENIGN": 0,
    "DDoS": 1
})

# Check for unexpected labels
if df["Label"].isna().any():
    raise ValueError("Unexpected traffic labels found!")

df["Label"] = df["Label"].astype(int)

# Step 8: Save cleaned dataset
output_path = data_folder / "cleaned_ddos.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved!")

print("\nFinal dataset shape:")
print(df.shape)

print("\nLabel distribution:")
print(df["Label"].value_counts())

print("\nRemaining missing values:")
print(df.isnull().sum().sum())
