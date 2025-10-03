# Imports
import pandas as pd
import numpy as np
import joblib # For saving and loading the model
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

# --- Configuration ---
# Use the file name provided in the current directory (safer than hardcoded C:\ paths)
FILE_NAME = "../dataset/claimlevel_cleaned.csv"
MODEL_FILENAME = 'claimlevel_xgb_model.joblib'
RANDOM_STATE = 42

# --- 1. Load and Prepare Data ---

# Load dataset
try:
    # Use relative path for better portability
    data = pd.read_csv(FILE_NAME)
    print(f"Successfully loaded {FILE_NAME} with {len(data)} claims.")
except FileNotFoundError:
    print(f"Error: The file '{FILE_NAME}' was not found. Please ensure it's in the same directory.")
    exit()

# Features (X) and Target (y)
X = data.drop("PotentialFraud", axis=1)
y_raw = data["PotentialFraud"]

# Convert Target to Binary (0 and 1)
# Assuming 'PotentialFraud' is categorical (Yes/No or 1/0 strings)
y = y_raw.astype(int) 

# --- 2. Handle Data Imbalance ---

# Train-Test Split (Stratified to ensure balanced class distribution in subsets)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# Calculate scale_pos_weight for XGBoost
# ratio_weight = (Count of Majority class: 0) / (Count of Minority class: 1)
counts = y_train.value_counts()
count_neg = counts[0]
count_pos = counts[1]
ratio_weight = count_neg / count_pos

print(f"\n--- Imbalance Stats ---")
print(f"Training Set Size: {len(X_train)}")
print(f"Non-Fraud (0) Count: {count_neg}")
print(f"Fraud (1) Count: {count_pos}")
print(f"Calculated scale_pos_weight: {ratio_weight:.2f}")


# --- 3. XGBoost Model Training ---

xgb_model = XGBClassifier(
    n_estimators=200,          # number of trees
    learning_rate=0.1,         # step size shrinkage
    max_depth=6,               # tree depth
    scale_pos_weight=ratio_weight, # KEY: Imbalance handling
    random_state=RANDOM_STATE,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Train model
print("\n--- Starting Model Training ---")
xgb_model.fit(X_train, y_train)
print("--- Training Complete ---")

# --- 4. Model Saving (using Joblib) ---

joblib.dump(xgb_model, MODEL_FILENAME)
print(f"\n✅ Model successfully saved to: {os.path.abspath(MODEL_FILENAME)}")

# --- 5. Evaluation ---

# Predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1] # Probability for AUC

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {auc:.4f}")
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📈 Classification Report:\n", classification_report(y_test, y_pred))