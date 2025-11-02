import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load CSVs ===
TRAIN_CSV = r"K:\Jilliana Abogado\021125\train.csv"
VAL_CSV = r"K:\Jilliana Abogado\021125\val.csv"
TEST_CSV = r"K:\Jilliana Abogado\021125\test.csv"

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

TARGET_COL = "risk_level"

# Combine train + val for final training
full_train = pd.concat([train_df, val_df], ignore_index=True)
X_train = full_train.drop(columns=[TARGET_COL])
y_train = full_train[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

# === Baseline SVM ===
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="macro")
recall = recall_score(y_test, preds, average="macro")
f1 = f1_score(y_test, preds, average="macro")
cm = confusion_matrix(y_test, preds)

print("Baseline SVM Performance:")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1 Score:  {f1:.6f}")
print("\nConfusion Matrix:\n", cm)

# === Plot Confusion Matrix ===
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM (Baseline)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
