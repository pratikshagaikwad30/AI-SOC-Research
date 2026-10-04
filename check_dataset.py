import pandas as pd

# Dataset file paths
train_path = "UNSW_NB15_training-set.csv"
test_path = "UNSW_NB15_testing-set.csv"

# Load datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Dataset size
print("TRAIN SHAPE:", train.shape)
print("TEST SHAPE:", test.shape)

# Column names
print("\nTRAIN COLUMNS:")
print(train.columns.tolist())

# Data types
print("\nDATA TYPES:")
print(train.dtypes)

# Missing values
print("\nMISSING VALUES:")
print(train.isnull().sum())

# Label distribution
if "label" in train.columns:
    print("\nLABEL DISTRIBUTION:")
    print(train["label"].value_counts())

# Attack category distribution
if "attack_cat" in train.columns:
    print("\nATTACK CATEGORY DISTRIBUTION:")
    print(train["attack_cat"].value_counts(dropna=False))

# First 5 rows
print("\nFIRST 5 ROWS:")
print(train.head())