import xgboost as xgb

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# Load the balanced train and original test sets (created by `SMOTE.py`)
TRAIN_CSV = r"K:\Jilliana Abogado\021125\train.csv"
VAL_CSV = r"K:\Jilliana Abogado\021125\val.csv"
TEST_CSV = r"K:\Jilliana Abogado\021125\test.csv"


train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# Separate features and labels
X_train = train_df.drop(columns=["risk_level"])
y_train = train_df["risk_level"]

X_val = val_df.drop(columns=["risk_level"])
y_val = val_df["risk_level"]

X_test = test_df.drop(columns=["risk_level"])
y_test = test_df["risk_level"]

# Create model
model = xgb.XGBClassifier(
    objective="multi:softprob",  # multiclass classification
    num_class=3,  # 3 classes: 0, 1, 2
    learning_rate=0.05,  # small LR for smoother learning
    n_estimators=1000,  # many trees; we’ll stop early
    max_depth=6,  # typical starting depth
    subsample=0.8,  # random row sampling for regularization
    colsample_bytree=0.8,  # random feature sampling
    eval_metric="mlogloss",  # monitor log loss
    random_state=42,
    early_stopping_rounds=0,  # stop if val doesn't improve for 20 rounds
)

# Evaluation sets for training monitoring
eval_set = [(X_train, y_train), (X_val, y_val)]

# Train with early stopping
model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=True,
)

# Extract results
results = model.evals_result()
train_loss = results["validation_0"]["mlogloss"]
val_loss = results["validation_1"]["mlogloss"]
rounds = range(1, len(train_loss) + 1)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(rounds, train_loss, label="Train Loss")
plt.plot(rounds, val_loss, label="Validation Loss")
plt.axvline(model.get_booster().best_iteration, color="r", linestyle="--", label="Best Iteration")
plt.xlabel("Boosting Round (n_estimators)")
plt.ylabel("Multiclass Log Loss")
plt.title("Training vs Validation Loss (XGBoost)")
plt.legend()
plt.grid(True)
plt.show()
