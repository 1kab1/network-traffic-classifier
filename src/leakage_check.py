
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# Load cleaned data
df = pd.read_csv("data/cleaned_ddos.csv")

# Original feature set
X = df.drop(columns=["Label"])
y = df["Label"]

# Split the data once so both experiments
# use exactly the same training and test rows
train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def evaluate(features, experiment_name):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        features.loc[train_idx],
        y.loc[train_idx]
    )

    predictions = model.predict(
        features.loc[test_idx]
    )

    actual = y.loc[test_idx]

    print(f"\n===== {experiment_name} =====")

    print(
        "Accuracy:",
        round(accuracy_score(actual, predictions) * 100, 4),
        "%"
    )

    print(
        "Precision:",
        round(precision_score(actual, predictions) * 100, 4),
        "%"
    )

    print(
        "Recall:",
        round(recall_score(actual, predictions) * 100, 4),
        "%"
    )

    print("Confusion Matrix:")
    print(confusion_matrix(actual, predictions))


# Experiment 1: All features
evaluate(X, "ALL FEATURES")

# Experiment 2: Remove Destination Port
X_reduced = X.drop(columns=["Destination Port"])

evaluate(
    X_reduced,
    "WITHOUT DESTINATION PORT"
)
