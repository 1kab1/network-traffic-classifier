
import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading cleaned dataset...")

df = pd.read_csv("data/cleaned_ddos.csv")

X = df.drop(columns=["Label"])
y = df["Label"]

# Recreate our existing train/test split
X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n===== DUPLICATE ANALYSIS =====")

# Check duplicates inside each partition
train_duplicates = X_train.duplicated().sum()
test_duplicates = X_test.duplicated().sum()

print("Training feature duplicates:", train_duplicates)
print("Testing feature duplicates:", test_duplicates)

# Identify feature records shared across partitions
train_hashes = pd.util.hash_pandas_object(
    X_train,
    index=False
)

test_hashes = pd.util.hash_pandas_object(
    X_test,
    index=False
)

shared_hashes = set(train_hashes)

shared_count = test_hashes.isin(shared_hashes).sum()

print("\nTest records matching training features:")
print(shared_count)

print(
    "Percentage:",
    round(shared_count / len(X_test) * 100, 4),
    "%"
)

print("\nDuplicate analysis completed!")
