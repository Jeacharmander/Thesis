"""
Optuna tuner for XGBoost — compatible with all XGBoost versions.
"""
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


# -------------------------------
# Objective Function for Optuna
# -------------------------------
def objective(trial, X, y):
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 10.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # --- Compatibility fix for all versions ---
    try:
        model = xgb.XGBClassifier(**params, random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
    except TypeError:
        # Fallback for older xgboost versions (using DMatrix)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        num_boost_round = params.pop('n_estimators')
        bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dval, 'validation')], verbose_eval=False)
        preds = np.argmax(bst.predict(dval), axis=1)

    f1 = f1_score(y_val, preds, average='macro')
    return f1


# -------------------------------
# Run the Optuna Study
# -------------------------------
def tune_xgboost(X, y, n_trials=50):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("\n=== OPTUNA TUNING RESULTS ===")
    print(f"Best Trial #: {study.best_trial.number}")
    print(f"Best F1 Score: {study.best_value:.4f}\n")
    print("Best Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    return study.best_trial.params, study


# -------------------------------
# Train and Evaluate Final Model
# -------------------------------
def train_best_model(X_train, y_train, X_test, y_test, best_params):
    model = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n=== TUNED XGBOOST MODEL PERFORMANCE ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))
    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))
    return model


# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    TRAIN_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_train.csv"
    TEST_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_test.csv"
    TARGET_COL = 'risk_level'

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    best_params, study = tune_xgboost(X_train, y_train, n_trials=1000)
    train_best_model(X_train, y_train, X_test, y_test, best_params)
