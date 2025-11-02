import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# === Load Data ===
TRAIN_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_train.csv"
TEST_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_test.csv"
TARGET_COL = 'risk_level'

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# === Tuned Parameters from Optuna ===
best_params = {
    'n_estimators': 207,
    'max_depth': 8,
    'learning_rate': 0.2970773146197139,
    'subsample': 0.7328657346243848,
    'colsample_bytree': 0.9260741278503519,
    'gamma': 0.006079175940272671,
    'reg_alpha': 0.9838677146904437,
    'reg_lambda': 1.5888472579213941
}

# === Train Tuned Model ===
model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

# === Evaluate ===
preds = model.predict(X_test)

print("\n=== TUNED XGBOOST MODEL PERFORMANCE ===")
print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"Precision: {precision_score(y_test, preds, average='macro'):.4f}")
print(f"Recall:    {recall_score(y_test, preds, average='macro'):.4f}")
print(f"F1-Score:  {f1_score(y_test, preds, average='macro'):.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, preds, target_names=['Low', 'Medium', 'High']))

print("\n=== CONFUSION MATRIX ===")
print("Rows = Actual | Columns = Predicted")
print(confusion_matrix(y_test, preds))
