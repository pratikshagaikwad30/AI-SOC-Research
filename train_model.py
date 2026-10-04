import pandas as pd
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load datasets
train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")

# Separate target
y_train = train["label"]
y_test = test["label"]

# Remove target, attack category and ID
X_train = train.drop(columns=["label", "attack_cat", "id"])
X_test = test.drop(columns=["label", "attack_cat", "id"])

# Identify categorical columns
categorical_columns = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# Transform data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

# Train model
print("\nTraining Random Forest...")
start_time = time.time()

model.fit(X_train_processed, y_train)

training_time = time.time() - start_time

print("Training completed!")
print("Training time:", round(training_time, 2), "seconds")

# Prediction
print("\nMaking predictions...")
start_prediction = time.time()

y_pred = model.predict(X_test_processed)

prediction_time = time.time() - start_prediction

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== MODEL RESULTS =====")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1-Score :", round(f1 * 100, 2), "%")
print("Prediction time:", round(prediction_time, 2), "seconds")