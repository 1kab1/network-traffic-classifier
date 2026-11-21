
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load cleaned dataset
print("Loading dataset...")

df = pd.read_csv("data/cleaned_ddos.csv")

X = df.drop(columns=["Label"])
y = df["Label"]

# Use the same split as our Decision Tree
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== RANDOM FOREST RESULTS =====")

print(f"Accuracy: {accuracy * 100:.4f}%")
print(f"Precision: {precision * 100:.4f}%")
print(f"Recall: {recall * 100:.4f}%")
print(f"F1 Score: {f1 * 100:.4f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["BENIGN", "DDoS"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Compare training and testing accuracy
print("\n===== OVERFITTING CHECK =====")

train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy * 100:.4f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.4f}%")

gap = (train_accuracy - test_accuracy) * 100
print(f"Accuracy Gap: {gap:.4f} percentage points")

# Save model
Path("models").mkdir(exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": X.columns.tolist()
    },
    "models/random_forest.pkl"
)

print("\nRandom Forest saved successfully!")
