
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Loading four-class training data...")

df = pd.read_csv("data/four_class_traffic.csv")

X = df.drop(columns=["Label", "Source"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training unweighted Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Save the model
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "labels": {
            0: "BENIGN",
            1: "DDoS",
            2: "PortScan",
            3: "Web Attack"
        }
    },
    "models/four_class_unweighted.pkl"
)

print("Unweighted model saved.")

# Find Monday dataset
files = list(
    Path("data").glob("Monday-WorkingHours*.csv")
)

if not files:
    raise FileNotFoundError(
        "Monday CSV not found."
    )

print("\nLoading Monday dataset...")

monday = pd.read_csv(files[0], low_memory=False)

monday.columns = monday.columns.str.strip()

features = X.columns.tolist()

missing = [
    col for col in features + ["Label"]
    if col not in monday.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in features:
    monday[col] = pd.to_numeric(
        monday[col],
        errors="coerce"
    )

monday[features] = monday[features].replace(
    [np.inf, -np.inf],
    np.nan
)

monday = monday.dropna(
    subset=features + ["Label"]
).copy()

monday["Label"] = (
    monday["Label"].astype(str).str.strip()
)

if not monday["Label"].eq("BENIGN").all():
    raise ValueError(
        "Unexpected labels in Monday dataset."
    )

print("Cleaned Monday records:", len(monday))

print("\nRunning predictions...")

predictions = model.predict(monday[features])

counts = pd.Series(predictions).value_counts()

labels = {
    0: "BENIGN",
    1: "DDoS",
    2: "PortScan",
    3: "Web Attack"
}

print("\n===== UNWEIGHTED MONDAY EVALUATION =====")

for class_id, name in labels.items():
    print(
        f"Predicted {name}:",
        counts.get(class_id, 0)
    )

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
