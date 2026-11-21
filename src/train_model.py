
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Constants
DATA_PATH = "data/cleaned_ddos.csv"
MODEL_PATH = "models/decision_tree.pkl"

OVERFITTING_HEADER = "\n===== OVERFITTING CHECK ====="

# Step 1: Load cleaned dataset
print("Loading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Step 2: Separate features and labels
X = df.drop(columns=["Label"])
y = df["Label"]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Step 4: Create Decision Tree model
print("\nTraining Decision Tree model...")

model = DecisionTreeClassifier(
    max_depth=10,
    random_state=42,
)

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Step 7: Calculate accuracy
train_accuracy = accuracy_score(
    y_train,
    train_predictions,
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

# Step 8: Display evaluation results
print("\n===== MODEL EVALUATION =====")

print(
    f"Training Accuracy: {train_accuracy * 100:.4f}%"
)

print(
    f"Testing Accuracy: {test_accuracy * 100:.4f}%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["BENIGN", "DDoS"],
        digits=4,
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions,
    )
)

# Step 9: Check for overfitting
print(OVERFITTING_HEADER)

accuracy_gap = (
    train_accuracy - test_accuracy
) * 100

print(
    f"Training Accuracy: {train_accuracy * 100:.4f}%"
)

print(
    f"Testing Accuracy: {test_accuracy * 100:.4f}%"
)

print(
    f"Accuracy Gap: {accuracy_gap:.4f} percentage points"
)

# Step 10: Save the trained model
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")

print("\nTraining completed successfully!")
