import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmap visualization
import joblib  # Library for saving and loading models

# --- 1. Load Data and Preprocessing ---
try:
    df = pd.read_csv("../dataset/provider_data_final.csv")
except FileNotFoundError:
    print("Error: 'provider_data_final.csv' not found. Please check file path.")
    exit()

# ✅ Remove target from features
X = df.drop(columns=['Fraud_Label'])
y = df['Fraud_Label']

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Handle Class Imbalance ---
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

print(f"No Fraud count in training data: {neg_count}, Fraud count: {pos_count}")
print(f"Setting scale_pos_weight to: {scale_pos_weight:.2f} to balance classes.")

# --- 4. Train XGBoost Classifier ---
xgb_model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    eval_metric='logloss',
    scale_pos_weight=9.456,
    random_state=42,
    tree_method='hist',
    use_label_encoder=False
)

print("\nTraining XGBoost model...")
xgb_model.fit(X_train, y_train)
print("Training complete.")

# --- 5. Save the Trained Model ---
model_filename = 'xgb_fraud_detection_model.joblib'
joblib.dump(xgb_model, model_filename)
print(f"\n✅ Model successfully saved to: {model_filename}")

# --- 6. Prediction and Evaluation ---
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['No Fraud (0)', 'Fraud (1)'])

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

# ✅ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# ✅ Plot Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix plot saved as confusion_matrix.png")

# --- 7. Feature Importance ---
feature_importances = xgb_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='indianred')
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('xgb_feature_importance.png')
print("\nFeature importance plot saved as xgb_feature_importance.png")
