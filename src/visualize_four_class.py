
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

print("Loading dataset and trained model...")

df = pd.read_csv("data/four_class_traffic.csv")

saved = joblib.load(
    "models/four_class_unweighted.pkl"
)

model = saved["model"]
features = saved["features"]

X = df[features]
y = df["Label"]

# Reproduce the original random test split
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Generating predictions...")

predictions = model.predict(X_test)

class_names = [
    "BENIGN",
    "DDoS",
    "PortScan",
    "Web Attack"
]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2, 3]
)

# Create confusion matrix plot
plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title(
    "Four-Class Network Intrusion Detection\n"
    "Random-Split Confusion Matrix"
)

plt.xlabel("Predicted Traffic Type")
plt.ylabel("Actual Traffic Type")

plt.tight_layout()

# Create results directory if needed
Path("results").mkdir(exist_ok=True)

output = "results/four_class_confusion_matrix.png"

plt.savefig(
    output,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        digits=4
    )
)

print("\nVisualization saved:", output)
