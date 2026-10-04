import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load datasets
train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")

# Target variable
y_train = train["label"]
y_test = test["label"]

# Remove target and attack category
X_train = train.drop(columns=["label", "attack_cat"])
X_test = test.drop(columns=["label", "attack_cat"])

# Remove ID because it is only an identifier
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])

# Find categorical and numerical columns
categorical_columns = X_train.select_dtypes(include=["object", "str"]).columns.tolist()
numerical_columns = X_train.select_dtypes(exclude=["object", "str"]).columns.tolist()

print("Categorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numerical", "passthrough", numerical_columns)
    ]
)

# Transform training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nOriginal training shape:", X_train.shape)
print("Processed training shape:", X_train_processed.shape)

print("\nOriginal testing shape:", X_test.shape)
print("Processed testing shape:", X_test_processed.shape)

print("\nPreprocessing completed successfully!")