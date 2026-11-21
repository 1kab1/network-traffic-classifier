
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load saved Random Forest
saved = joblib.load("models/random_forest.pkl")

model = saved["model"]
features = saved["features"]

# Extract feature importance scores
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

# Sort by importance
importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== TOP 15 IMPORTANT FEATURES =====")
print(importance.head(15).to_string(index=False))

# Save full importance ranking
Path("results").mkdir(exist_ok=True)

importance.to_csv(
    "results/feature_importance.csv",
    index=False
)

# Plot top 15 features
top_features = importance.head(15)

plt.figure(figsize=(11, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.title("Random Forest - Top 15 Features")
plt.xlabel("Feature Importance")
plt.ylabel("Network Traffic Feature")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature importance saved successfully!")
