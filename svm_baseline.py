# ===============================
# svm_baseline.py
# ===============================

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

if __name__ == "__main__":
    # === Paths ===
    TRAIN_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_train.csv"
    TEST_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_test.csv"
    TARGET_COL = 'risk_level'

    # === Load Data ===
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # === Baseline SVM Model (with scaling) ===
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', random_state=42)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # === Metrics ===
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # === Display ===
    print("=== BASELINE SVM MODEL PERFORMANCE ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))
    print("\n=== CONFUSION MATRIX ===")
    print("Rows = Actual | Columns = Predicted")
    print(confusion_matrix(y_test, y_pred))
