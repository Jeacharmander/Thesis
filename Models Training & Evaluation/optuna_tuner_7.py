import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def tune_xgb(X, y, trials=10, metric="accuracy"):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_class": 3,
            "random_state": 42,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "objective": "multi:softprob",
        }

        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = xgb.XGBClassifier(
            **params,
            eval_metric="mlogloss",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)

        if metric == "accuracy":
            return accuracy_score(y_val, preds)
        else:
            return f1_score(y_val, preds, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study.best_trial.params, study


if __name__ == "__main__":
    TRAIN_CSV = r"K:\Jilliana Abogado\021125\train.csv"
    VAL_CSV = r"K:\Jilliana Abogado\021125\val.csv"
    TEST_CSV = r"K:\Jilliana Abogado\021125\test.csv"

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    TARGET_COL = "risk_level"

    full_train_df = pd.concat([train_df, val_df])
    X_full_train = full_train_df.drop(columns=[TARGET_COL])
    y_full_train = full_train_df[TARGET_COL]

    best_params, study = tune_xgb(X_full_train, y_full_train, trials=1000, metric="accuracy")

    print("Best parameters found:")
    print(best_params)
