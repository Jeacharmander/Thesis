"""
Improved Optuna tuner for XGBoost with cross-validation and better parameter handling
"""
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Baseline Performance Check
# -------------------------------
def check_baseline(X_train, y_train, X_test, y_test):
    """Check what a simple default model achieves"""
    print("=== CHECKING BASELINE PERFORMANCE ===")
    
    # Default model
    default_model = xgb.XGBClassifier(
        objective='multi:softprob',
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    
    # Cross-validation score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(default_model, X_train, y_train, cv=cv, scoring='f1_macro')
    print(f"Default model CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Test score
    default_model.fit(X_train, y_train)
    y_pred = default_model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Default model test F1: {test_f1:.4f}")
    
    return test_f1

# -------------------------------
# Improved Objective Function for Optuna
# -------------------------------
def objective(trial, X, y):
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'random_state': 42
    }

    # Use cross-validation for more reliable evaluation
    model = xgb.XGBClassifier(**params)
    
    # Use fewer folds for faster tuning, but more reliable than single split
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
    
    return scores.mean()

# -------------------------------
# Run the Improved Optuna Study
# -------------------------------
def tune_xgboost(X, y, n_trials=100):
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    print(f"Starting Optuna optimization with {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("\n" + "="*50)
    print("OPTUNA TUNING RESULTS")
    print("="*50)
    print(f"Best Trial #: {study.best_trial.number}")
    print(f"Best F1 Score: {study.best_value:.4f}")
    print(f"Number of completed trials: {len(study.trials)}")
    print("\nBest Parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    return study.best_trial.params, study

# -------------------------------
# Train and Evaluate Final Model
# -------------------------------
def train_best_model(X_train, y_train, X_test, y_test, best_params):
    # Ensure all required parameters are included
    final_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    final_params.update(best_params)
    
    # Remove parameters that might cause issues in XGBClassifier
    if 'num_class' in final_params:
        final_params.pop('num_class')
    
    # For XGBClassifier, early_stopping_rounds goes in the constructor, not fit()
    final_params['early_stopping_rounds'] = 50
    
    model = xgb.XGBClassifier(**final_params)
    
    # Create validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Train with early stopping - note: eval_set is the only parameter in fit()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "="*50)
    print("TUNED XGBOOST MODEL PERFORMANCE")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nCLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    print("\nCONFUSION MATRIX")
    print("-" * 50)
    print(confusion_matrix(y_test, y_pred))
    
    return model, f1

# -------------------------------
# Alternative Simple Training (if you still face issues)
# -------------------------------
def train_best_model_simple(X_train, y_train, X_test, y_test, best_params):
    """Simplified version without early stopping"""
    # Ensure all required parameters are included
    final_params = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42
    }
    final_params.update(best_params)
    
    # Remove parameters that might cause issues
    if 'num_class' in final_params:
        final_params.pop('num_class')
    
    model = xgb.XGBClassifier(**final_params)
    
    # Simple training without validation split
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("\n" + "="*50)
    print("TUNED XGBOOST MODEL PERFORMANCE (SIMPLE TRAINING)")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nCLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    print("\nCONFUSION MATRIX")
    print("-" * 50)
    print(confusion_matrix(y_test, y_pred))
    
    return model, f1

# -------------------------------
# Visualization Functions
# -------------------------------
def plot_study_results(study):
    """Plot optimization history and parameter importance"""
    try:
        # Optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.show()
        
        # Parameter importance
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.show()
        
    except ImportError:
        print("Plotly not available for plotting")
    except Exception as e:
        print(f"Could not generate plots: {e}")

# -------------------------------
# MAIN EXECUTION
# -------------------------------
if __name__ == "__main__":
    # Configuration
    TRAIN_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_train.csv"
    TEST_CSV = r"C:\Cus\Jilliana Abogado\Datasets\balanced_test.csv"
    TARGET_COL = 'risk_level'
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)}")
    print(f"Class distribution in training: {np.bincount(y_train)}")
    print(f"Class distribution in test: {np.bincount(y_test)}")
    
    # Step 1: Check baseline performance
    baseline_f1 = check_baseline(X_train, y_train, X_test, y_test)
    
    # Step 2: Run Optuna tuning
    best_params, study = tune_xgboost(X_train, y_train, n_trials=50)  # Reduced for testing
    
    # Step 3: Train and evaluate final model (try the corrected version first)
    try:
        final_model, tuned_f1 = train_best_model(X_train, y_train, X_test, y_test, best_params)
    except Exception as e:
        print(f"Error with advanced training: {e}")
        print("Falling back to simple training...")
        final_model, tuned_f1 = train_best_model_simple(X_train, y_train, X_test, y_test, best_params)
    
    # Step 4: Compare results
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    print(f"Baseline F1-score: {baseline_f1:.4f}")
    print(f"Tuned F1-score:    {tuned_f1:.4f}")
    print(f"Improvement:       {tuned_f1 - baseline_f1:.4f}")
    
    if tuned_f1 > baseline_f1:
        print("✅ Tuning improved performance!")
    else:
        print("❌ Tuning did not improve performance. Consider:")
        print("   - Checking data quality and feature engineering")
        print("   - Trying different ML algorithms")
        print("   - Increasing n_trials")
        print("   - Adjusting parameter search spaces")
    
    # Step 5: Plot results (optional)
    try:
        plot_study_results(study)
    except:
        print("Skipping plots due to dependency issues")
    
    print("\nTuning completed!")