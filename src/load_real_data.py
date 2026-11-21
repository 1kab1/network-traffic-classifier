
import pandas as pd
from pathlib import Path

# Locate the dataset
data_folder = Path("data")

files = list(
    data_folder.glob("Friday-WorkingHours-Afternoon-DDos*.csv")
)

if not files:
    raise FileNotFoundError(
        "DDoS CSV file not found in the data folder."
    )

file_path = files[0]

print("Loading dataset...")
print("File:", file_path.name)

# Load the real network traffic dataset
df = pd.read_csv(file_path, low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Display basic information
print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head().to_string())

print("\nNumber of columns:")
print(len(df.columns))

print("\nColumn names:")
for column in df.columns:
    print(column)

# Inspect labels
if "Label" in df.columns:
    print("\nTraffic classification counts:")
    print(df["Label"].value_counts(dropna=False))
else:
    print("\nLabel column not found.")

# Check missing values
print("\nTotal missing values:")
print(df.isnull().sum().sum())

print("\nDataset loaded successfully!")

