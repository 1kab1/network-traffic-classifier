
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading multiclass dataset...")

df = pd.read_csv("data/multiclass_traffic.csv")

# Exclude label and source metadata
X = df.drop(columns=["Label", "Source"])
y = df["Label"]

print("Dataset shape:", df.shape)
print("Number of features:", X.shape[1])

# Split while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining multiclass Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("\n===== MULTICLASS EVALUATION =====")

print(
    "Test Accuracy:",
    round(
        accuracy_score(y_test, predictions) * 100,
        4
    ),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "BENIGN",
            "DDoS",
            "PortScan"
        ],
        digits=4,
        zero_division=0
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )
)

# Save model and feature names
joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist(),
        "labels": {
            0: "BENIGN",
            1: "DDoS",
            2: "PortScan"
        }
    },
    "models/multiclass_random_forest.pkl"
)

print(
    "\nModel saved: "
    "models/multiclass_random_forest.pkl"
)
