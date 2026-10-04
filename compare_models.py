import pandas as pd
import time

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load datasets
train = pd.read_csv("UNSW_NB15_training-set.csv")
test = pd.read_csv("UNSW_NB15_testing-set.csv")

# Target
y_train = train["label"]
y_test = test["label"]

# Remove target, attack category and ID
X_train = train.drop(columns=["label", "attack_cat", "id"])
X_test = test.drop(columns=["label", "attack_cat", "id"])

# Categorical columns
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

# Transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# ==================================================
# LOGISTIC REGRESSION
# ==================================================

print("\nTraining Logistic Regression...")

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

start = time.time()

lr.fit(X_train_processed, y_train)

lr_train_time = time.time() - start

lr_pred = lr.predict(X_test_processed)

# Metrics
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)

lr_cm = confusion_matrix(y_test, lr_pred)

lr_tn, lr_fp, lr_fn, lr_tp = lr_cm.ravel()

lr_fpr = lr_fp / (lr_fp + lr_tn)


# ==================================================
# RANDOM FOREST
# ==================================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

start = time.time()

rf.fit(X_train_processed, y_train)

rf_train_time = time.time() - start

rf_pred = rf.predict(X_test_processed)

# Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

rf_cm = confusion_matrix(y_test, rf_pred)

rf_tn, rf_fp, rf_fn, rf_tp = rf_cm.ravel()

rf_fpr = rf_fp / (rf_fp + rf_tn)


# ==================================================
# RESULTS
# ==================================================

print("\n========================================")
print("       MODEL COMPARISON RESULTS")
print("========================================")

print("\nMetric                 Logistic Regression     Random Forest")
print("------------------------------------------------------------")

print(
    f"Accuracy              {lr_accuracy * 100:.2f}%"
    f"                 {rf_accuracy * 100:.2f}%"
)

print(
    f"Precision             {lr_precision * 100:.2f}%"
    f"                 {rf_precision * 100:.2f}%"
)

print(
    f"Recall                {lr_recall * 100:.2f}%"
    f"                 {rf_recall * 100:.2f}%"
)

print(
    f"F1-score              {lr_f1 * 100:.2f}%"
    f"                 {rf_f1 * 100:.2f}%"
)

print(
    f"False Positive Rate   {lr_fpr * 100:.2f}%"
    f"                 {rf_fpr * 100:.2f}%"
)

print("\nTraining Time:")
print("Logistic Regression:", round(lr_train_time, 2), "seconds")
print("Random Forest      :", round(rf_train_time, 2), "seconds")

print("\nLogistic Regression Confusion Matrix:")
print(lr_cm)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)