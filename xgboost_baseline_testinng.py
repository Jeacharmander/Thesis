import sys
import matplotlib
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np
import matplotlib.pyplot as plt


# =========================================
# CONFIGURATION
# =========================================
matplotlib.use('Agg')  # Safe for non-GUI environments

TRAIN_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_train.csv"
TEST_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_test.csv"
TARGET_COL = 'risk_level'

# =========================================
# LOAD DATA
# =========================================
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]

X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# =========================================
# TRAIN MODEL (XGBoost)
# =========================================
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# =========================================
# PREDICTIONS
# =========================================
y_pred = model.predict(X_test)

# =========================================
# METRICS
# =========================================
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n=== MODEL PERFORMANCE ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))

# =========================================
# CONFUSION MATRIX (print in terminal)
# =========================================
cm = confusion_matrix(y_test, y_pred)

print("\n=== CONFUSION MATRIX ===")
print(pd.DataFrame(cm, 
                   index=['Actual_Low', 'Actual_Medium', 'Actual_High'], 
                   columns=['Pred_Low', 'Pred_Medium', 'Pred_High']))

# # =========================================
# # OPTIONAL: SHAP Feature Importance
# # =========================================
# explainer = shap.Explainer(model, X_train)
# shap_values = explainer(X_test)

# # Save SHAP summary plot (still useful for later)
# shap.summary_plot(shap_values, X_test, show=False)
# plt.savefig(r"C:\Cus\Jilliana Abogado\shap_summary_xgboost.png", dpi=300, bbox_inches='tight')
# plt.close()

# print("\nSHAP feature importance plot saved as 'shap_summary_xgboost.png'.")
