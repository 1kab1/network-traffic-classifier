
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)

print("Loading four-class dataset...")

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

experiments = {
    "Balanced Subsample": "balanced_subsample",
    "No Class Weight": None
}

for name, weight in experiments.items():

    print(f"\nTraining: {name}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight=weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    web_precision = precision_score(
        y_test,
        predictions,
        labels=[3],
        average="macro",
        zero_division=0
    )

    web_recall = recall_score(
        y_test,
        predictions,
        labels=[3],
        average="macro",
        zero_division=0
    )

    print(f"\n===== {name.upper()} =====")

    print(
        "Overall Accuracy:",
        round(
            accuracy_score(y_test, predictions) * 100,
            4
        ),
        "%"
    )

    print(
        "Web Attack Precision:",
        round(web_precision * 100, 4),
        "%"
    )

    print(
        "Web Attack Recall:",
        round(web_recall * 100, 4),
        "%"
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1, 2, 3]
        )
    )

print("\nComparison completed!")
