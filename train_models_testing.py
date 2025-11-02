import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from xgboost import XGBClassifier  # pip install xgboost

# 1️⃣ Load the balanced train and original test sets
train_df = pd.read_csv(r"C:\Users\jycab\Documents\Thesis\smote_balanced_train_data.csv")
test_df = pd.read_csv(r"C:\Users\jycab\Documents\Thesis\original_test_data.csv")

# 🔧 Drop extra unnamed columns if present
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# 2️⃣ Separate features and target
X_train = train_df.drop(columns=["risk_level"])
y_train = train_df["risk_level"]

X_test = test_df.drop(columns=["risk_level"])
y_test = test_df["risk_level"]

# 3️⃣ Define models to test
models = {
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss", random_state=42
    ),
}

# 4️⃣ Train and evaluate each model
for name, model in models.items():
    print(f"\n🔹 Training {name}...")

    if name == "XGBoost":
        y_train_fixed = y_train - 1
        y_test_fixed = y_test - 1
        model.fit(X_train, y_train_fixed)
        y_pred = model.predict(X_test) + 1
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Print nicely formatted results
    print(f"\n📊 {name} Results:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("-" * 50)

    # ✅ Save the trained XGBoost model for future predictions
    if name == "XGBoost":
        joblib.dump(model, "xgboost_cervical_risk_model.pkl")
        print("✅ Model saved as xgboost_cervical_risk_model.pkl")
