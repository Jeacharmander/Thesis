import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import pandas as pd

# === Load CSVs ===
TRAIN_CSV = r"K:\Jilliana Abogado\021125\train.csv"
VAL_CSV = r"K:\Jilliana Abogado\021125\val.csv"
TEST_CSV = r"K:\Jilliana Abogado\021125\test.csv"

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

TARGET_COL = "risk_level"

# === Combine train + val for final training ===
full_train = pd.concat([train_df, val_df], ignore_index=True)
X_train = full_train.drop(columns=[TARGET_COL])
y_train = full_train[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# === Default XGBoost ===
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

# === Train ===
model.fit(X_train, y_train)

# === Predict ===
predictions = model.predict(X_test)

# === Metrics ===
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="macro")
recall = recall_score(y_test, predictions, average="macro")
f1 = f1_score(y_test, predictions, average="macro")
cm = confusion_matrix(y_test, predictions)

print("=== Baseline XGBoost Performance ===")
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1 Score: {f1:.6f}")
print("\nConfusion Matrix:\n", cm)

# === Plot Confusion Matrix ===
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y_test.unique()))
disp.plot(cmap="Blues", values_format="d")
plt.title("Baseline XGBoost Confusion Matrix")
plt.show()
