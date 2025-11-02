import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

# === Load all saved Optuna hyperparameters ===
with open(r"K:\Jilliana Abogado\021125\optuna_best_hyperparameters_by_trials.json", "r") as f:
    all_params = json.load(f)

# === Prepare plotting area ===
num_models = len(all_params)
fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5))
if num_models == 1:
    axes = [axes]  # make iterable if only one

# === Metrics storage ===
metrics_summary = []

# === Evaluate each Optuna variant ===
for i, (trial_name, params) in enumerate(all_params.items()):
    print(f"\n🔍 Evaluating {trial_name} ...")

    # Build model
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        **params,
    )

    # Train on combined train+val
    model.fit(X_train, y_train, verbose=False)

    # Predict
    preds = model.predict(X_test)

    # === Compute metrics ===
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    metrics_summary.append({"Trial": trial_name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-score": f1})

    # === Confusion Matrix Plot ===
    cm = confusion_matrix(y_test, preds)
    ax = axes[i]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{trial_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# === Final Layout ===
plt.tight_layout()
plt.show()

# === Print summary table ===
metrics_df = pd.DataFrame(metrics_summary)
print("\n📊 Performance Summary Across Trials:")
print(metrics_df.to_string(index=False))
