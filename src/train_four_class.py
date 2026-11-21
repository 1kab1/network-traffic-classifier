
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Loading four-class dataset...")

df = pd.read_csv("data/four_class_traffic.csv")

# Select network features only
# Source is metadata, not a model feature
X = df.drop(columns=["Label", "Source"])
y = df["Label"]

print("Dataset shape:", df.shape)
print("Number of features:", X.shape[1])

# Stratified split preserves class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining four-class Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\n===== FOUR-CLASS EVALUATION =====")

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
        labels=[0, 1, 2, 3],
        target_names=[
            "BENIGN",
            "DDoS",
            "PortScan",
            "Web Attack"
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
        labels=[0, 1, 2, 3]
    )
)

# Save model and feature order
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
    "models/four_class_random_forest.pkl"
)

print(
    "\nModel saved: "
    "models/four_class_random_forest.pkl"
)
