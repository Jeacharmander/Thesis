"""Optuna tuning utilities for XGBoost."""
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def tune_xgb(X, y, trials=10, metric='accuracy'):
    """Run an Optuna study to tune XGBoost for X,y. Returns best_params and study object."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'use_label_encoder': False,
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y)),
            'seed': 42
        }

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        try:
            model = xgb.XGBClassifier(**params, eval_metric='mlogloss')
            model.fit(X_tr, y_tr, early_stopping_rounds=20, eval_set=[(X_val, y_val)], verbose=False)
            preds = model.predict(X_val)
        except TypeError:
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)
            train_params = params.copy()
            num_boost_round = train_params.pop('n_estimators', 100)
            train_params.pop('use_label_encoder', None)
            bst = xgb.train(train_params, dtrain, num_boost_round=num_boost_round, evals=[(dval, 'validation')],
                            early_stopping_rounds=20, verbose_eval=False)
            preds = np.argmax(bst.predict(dval), axis=1)

        if metric == 'accuracy':
            score = accuracy_score(y_val, preds)
        else:
            score = f1_score(y_val, preds, average='macro')

        print(f"Trial {trial.number} completed with {metric}={score:.4f}")
        return score

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    return study.best_trial.params, study



if __name__ == "__main__":
    TRAIN_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_train.csv"
    TEST_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_test.csv"
    TARGET_COL = 'risk_level'

    # Load datasets
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Split into features and labels
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # Run Optuna tuning
    best_params, study = tune_xgb(X_train, y_train, trials=50, metric='f1')

    # Get the best trial info
    best_trial = study.best_trial

    print("\n=== OPTUNA TUNING RESULTS ===")
    print(f"Best Trial #: {best_trial.number}")
    print(f"Best F1 Score: {best_trial.value:.4f}")
    print("\nBest Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")