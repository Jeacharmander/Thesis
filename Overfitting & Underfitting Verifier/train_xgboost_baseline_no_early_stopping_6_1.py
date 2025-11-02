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

model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    learning_rate=0.05,
    n_estimators=3000,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42,
)

eval_set = [(X_train, y_train), (X_val, y_val)]

model.fit(
    X_train,
    y_train,
    eval_set=eval_set,  # just to record the training/validation loss
    verbose=True,
)

results = model.evals_result()
train_loss = results["validation_0"]["mlogloss"]
val_loss = results["validation_1"]["mlogloss"]
rounds = range(1, len(train_loss) + 1)

plt.figure(figsize=(10, 5))
plt.plot(rounds, train_loss, label="Train Loss")
plt.plot(rounds, val_loss, label="Validation Loss")
plt.xlabel("Boosting Round (n_estimators)")
plt.ylabel("Multiclass Log Loss")
plt.title("Training vs Validation Loss (No Early Stopping)")
plt.legend()
plt.grid(True)
plt.show()
