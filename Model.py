# Import libraries
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load preprocessed data
train_data = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Flight Delays\PP Train.csv")
test_data = pd.read_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Flight Delays\PP Test.csv")

# Separate features and target variable
X = train_data.drop('dep_delayed_15min', axis=1)
y = train_data['dep_delayed_15min']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Dictionary to store model results
model_results = {}

# 1. Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict_proba(X_val_scaled)[:, 1]
roc_auc_log_reg = roc_auc_score(y_val, y_pred_log_reg)
model_results['Logistic Regression'] = roc_auc_log_reg

# 2. Random Forest Classifier with Hyperparameter Tuning
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', None]
}

rf_clf = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(rf_clf, rf_param_grid, n_iter=10, scoring='roc_auc', cv=2, verbose=1, n_jobs=-1, random_state=42)
rf_search.fit(X_train_scaled, y_train)
best_rf = rf_search.best_estimator_

# Make predictions on the validation set
y_pred_rf = best_rf.predict_proba(X_val_scaled)[:, 1]
roc_auc_rf = roc_auc_score(y_val, y_pred_rf)
model_results['Random Forest'] = roc_auc_rf

# 3. XGBoost Classifier with Hyperparameter Tuning
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Set up parameter grid for Randomized Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.6, 0.8],
    'gamma': [0, 0.1],
    'scale_pos_weight': [1]  # Keep it simple for initial tests
}

# Perform Randomized Search
random_search = RandomizedSearchCV(
    xgb_clf, param_distributions=param_grid, n_iter=10, 
    scoring='roc_auc', cv=2, verbose=1, n_jobs=-1, random_state=42
)
random_search.fit(X_train_scaled, y_train)

# Get best model from Randomized Search
best_xgb = random_search.best_estimator_

# Make predictions on the validation set
y_pred_xgb = best_xgb.predict_proba(X_val_scaled)[:, 1]
roc_auc_xgb = roc_auc_score(y_val, y_pred_xgb)
model_results['XGBoost'] = roc_auc_xgb

# Display ROC-AUC scores for each model
print("ROC-AUC Scores:")
for model, score in model_results.items():
    print(f"{model}: {score:.4f}")

# Select the best model
best_model_name = max(model_results, key=model_results.get)
print(f"\nBest Model Based on ROC-AUC: {best_model_name} with score {model_results[best_model_name]:.4f}")

# Final training on the full dataset if XGBoost is the best model
if best_model_name == 'XGBoost':
    print("Training final model with best XGBoost parameters on full training data...")
    best_xgb.fit(scaler.fit_transform(X), y)  # Scale full data

    # Make predictions on the test dataset
    test_data_scaled = scaler.transform(test_data)  # Scale test data
    test_predictions = best_xgb.predict_proba(test_data_scaled)[:, 1]

    # Save predictions to a CSV file
    submission = pd.DataFrame({'Id': test_data.index, 'dep_delayed_15min': test_predictions})
    submission.to_csv(r"C:\Users\karan\OneDrive\Desktop\Kaggle\Community\Flight Delays\sample_submission.csv", index=False)
    print("Final predictions saved as 'submission.csv'.")
