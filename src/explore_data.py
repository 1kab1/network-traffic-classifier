
import pandas as pd

# Create sample network traffic data
data = {
    "duration": [1.2, 0.5, 3.4, 0.2, 5.1],
    "bytes": [1500, 800, 5000, 200, 9000],
    "packets": [10, 5, 30, 2, 60],
    "label": [
        "Benign",
        "Benign",
        "Suspicious",
        "Benign",
        "Suspicious"
    ]
}

# Convert data into a table
df = pd.DataFrame(data)

# Save the table as a CSV file
df.to_csv("data/sample_traffic.csv", index=False)

# Display the dataset
print("Network Traffic Dataset:")
print(df)

print("\nDataset Shape:", df.shape)

print("\nTraffic Categories:")
print(df["label"].value_counts())

print("\nDataset saved successfully!")
