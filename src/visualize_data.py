
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load our network traffic dataset
df = pd.read_csv("data/sample_traffic.csv")

# Create a folder for results
Path("results").mkdir(exist_ok=True)

# Set a chart theme
sns.set_theme(style="whitegrid")

# Create two charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Chart 1: Traffic category counts
sns.countplot(
    data=df,
    x="label",
    hue="label",
    palette="Set2",
    legend=False,
    ax=axes[0]
)

axes[0].set_title("Network Traffic Categories")
axes[0].set_xlabel("Traffic Type")
axes[0].set_ylabel("Number of Records")

# Chart 2: Bytes by traffic category
sns.scatterplot(
    data=df,
    x="duration",
    y="bytes",
    hue="label",
    s=120,
    ax=axes[1]
)

axes[1].set_title("Network Traffic Patterns")
axes[1].set_xlabel("Duration (seconds)")
axes[1].set_ylabel("Bytes Transferred")

plt.tight_layout()

# Save the charts
plt.savefig("results/traffic_visualization.png", dpi=300)

# Display the charts
plt.show()

print("Visualization saved successfully!")
