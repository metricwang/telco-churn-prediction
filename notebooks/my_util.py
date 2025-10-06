import mlflow
import mlflow.sklearn

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

PRIMARY_METRIC = 'roc_auc'  # What to optimize during GridSearch

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Comprehensive model evaluation with visualizations"""
    
    # Get probabilities (what the model actually outputs)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Get predictions using default threshold=0.5
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\n{'='*70}")
    print(f"{model_name} Results")
    print(f"{'='*70}")
    print(f"Train AUC: {train_auc:.4f}")
    print(f"Test AUC:  {test_auc:.4f}")
    print(f"Overfit:   {train_auc - test_auc:.4f}")
    
    print(f"\nTest Set Performance (threshold=0.5):")
    print(classification_report(y_test, y_test_pred))
    
    print(f"\nConfusion Matrix (threshold=0.5):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'AUC = {test_auc:.3f}')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate (Recall)')
    axes[0, 0].set_title(f'{model_name} - ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
    axes[0, 1].plot(recall, precision, linewidth=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title(f'{model_name} - Precision-Recall Curve')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix (threshold=0.5)')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xlabel('Predicted')
    
    # 4. Probability Distribution
    axes[1, 1].hist(y_test_proba[y_test == 0], bins=50, alpha=0.5, label='No Churn', density=True)
    axes[1, 1].hist(y_test_proba[y_test == 1], bins=50, alpha=0.5, label='Churn', density=True)
    axes[1, 1].axvline(0.5, color='red', linestyle='--', label='Default Threshold')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Probability Distribution by True Class')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'y_test_proba': y_test_proba,
        'figure': fig
    }

# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================

def train_logistic(pipeline, X_train, y_train, X_test, y_test, quick_mode=True):
    """
    Train Logistic Regression with hyperparameter tuning
    
    Parameters:
    -----------
    pipeline : Pipeline with LogisticRegression as final step
    quick_mode : bool
        If True, use smaller grid for faster iteration
        If False, use comprehensive grid (takes hours!)
    """
    
    print("\n" + "="*70)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*70)
    
    if quick_mode:
        print("Running in QUICK mode (faster, smaller grid)")
        param_grid = {
            'regressor__C': [0.1, 1.0],
            'regressor__penalty': ['l2'],
            'regressor__solver': ['liblinear'],  # Both support l1
            'regressor__class_weight': [None]
        }
    else:
        print("Running in COMPREHENSIVE mode (slower, full grid)")
        param_grid = {
            'regressor__C': [0.01, 0.1, 1.0, 10.0],
            'regressor__penalty': ['l1', 'l2'],
            'regressor__solver': ['liblinear', 'saga'],  # Both support l1
            'regressor__class_weight': [None, 'balanced']
        }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring=PRIMARY_METRIC,
        n_jobs=-1
    )
    
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV {PRIMARY_METRIC}: {grid_search.best_score_:.4f}")
    
    # Evaluate
    best_model = grid_search.best_estimator_
    eval_results = evaluate_model(best_model, X_train, y_train, X_test, y_test, "Logistic Regression")
    
    # Log to MLflow
    cv_results = {
        'mean_test_auc': grid_search.best_score_,
        'std_test_auc': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    }
    
    log_model_to_mlflow(
        best_model,
        "Logistic_Regression",
        grid_search.best_params_,
        cv_results,
        eval_results,
        X_train
    )
    
    return best_model, grid_search, eval_results



# ============================================================================
# RANDOM FOREST
# ============================================================================

def train_random_forest(pipeline, X_train, y_train, X_test, y_test, quick_mode=True):
    """
    Train Random Forest with hyperparameter tuning
    
    Parameters:
    -----------
    pipeline : Pipeline with RandomForestClassifier as final step
    quick_mode : bool
        If True, use smaller grid for faster iteration
        If False, use comprehensive grid (takes hours!)
    """
    
    print("\n" + "="*70)
    print("TRAINING RANDOM FOREST")
    print("="*70)
    
    if quick_mode:
        print("Running in QUICK mode (faster, smaller grid)")
        param_grid = {
            'randomforestclassifier__n_estimators': [100, 200],
            'randomforestclassifier__max_depth': [10, 20, None],
            'randomforestclassifier__min_samples_split': [2, 5],
            'randomforestclassifier__min_samples_leaf': [1, 2],
            'randomforestclassifier__class_weight': ['balanced', None]
        }
    else:
        print("Running in COMPREHENSIVE mode (slower, full grid)")
        param_grid = {
            'randomforestclassifier__n_estimators': [100, 200, 300],
            'randomforestclassifier__max_depth': [10, 20, 30, None],
            'randomforestclassifier__min_samples_split': [2, 5, 10],
            'randomforestclassifier__min_samples_leaf': [1, 2, 4],
            'randomforestclassifier__max_features': ['sqrt', 'log2'],
            'randomforestclassifier__class_weight': ['balanced', None]
        }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring=PRIMARY_METRIC,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV {PRIMARY_METRIC}: {grid_search.best_score_:.4f}")
    
    # Evaluate
    best_model = grid_search.best_estimator_
    eval_results = evaluate_model(best_model, X_train, y_train, X_test, y_test, "Random Forest")
    
    # Log to MLflow
    cv_results = {
        'mean_test_auc': grid_search.best_score_,
        'std_test_auc': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    }
    
    log_model_to_mlflow(
        best_model,
        "Random_Forest",
        grid_search.best_params_,
        cv_results,
        eval_results,
        X_train
    )
    
    return best_model, grid_search, eval_results


# ============================================================================
# XGBOOST
# ============================================================================

def train_xgboost(pipeline, X_train, y_train, X_test, y_test, quick_mode=True):
    """
    Train XGBoost with hyperparameter tuning
    
    Parameters:
    -----------
    pipeline : Pipeline with XGBClassifier as final step
    quick_mode : bool
        If True, use smaller grid for faster iteration
    """
    
    print("\n" + "="*70)
    print("TRAINING XGBOOST")
    print("="*70)
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio: {scale_pos_weight:.2f}")
    
    if quick_mode:
        print("Running in QUICK mode (faster, smaller grid)")
        param_grid = {
            'xgbclassifier__n_estimators': [100, 200],
            'xgbclassifier__max_depth': [3, 5, 7],
            'xgbclassifier__learning_rate': [0.05, 0.1],
            'xgbclassifier__subsample': [0.8, 1.0],
            'xgbclassifier__colsample_bytree': [0.8, 1.0],
            'xgbclassifier__scale_pos_weight': [1, scale_pos_weight]
        }
    else:
        print("Running in COMPREHENSIVE mode (slower, full grid)")
        param_grid = {
            'xgbclassifier__n_estimators': [100, 200, 300],
            'xgbclassifier__max_depth': [3, 5, 7, 10],
            'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
            'xgbclassifier__subsample': [0.8, 0.9, 1.0],
            'xgbclassifier__colsample_bytree': [0.8, 0.9, 1.0],
            'xgbclassifier__min_child_weight': [1, 3, 5],
            'xgbclassifier__gamma': [0, 0.1, 0.2],
            'xgbclassifier__scale_pos_weight': [1, scale_pos_weight]
        }
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring=PRIMARY_METRIC,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV {PRIMARY_METRIC}: {grid_search.best_score_:.4f}")
    
    # Evaluate
    best_model = grid_search.best_estimator_
    eval_results = evaluate_model(best_model, X_train, y_train, X_test, y_test, "XGBoost")
    
    # Log to MLflow
    cv_results = {
        'mean_test_auc': grid_search.best_score_,
        'std_test_auc': grid_search.cv_results_['std_test_score'][grid_search.best_index_]
    }
    
    log_model_to_mlflow(
        best_model,
        "XGBoost",
        grid_search.best_params_,
        cv_results,
        eval_results,
        X_train
    )
    
    return best_model, grid_search, eval_results

# ============================================================================
# Find optimal threshold
# ============================================================================
def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Find optimal threshold for a given metric
    
    Parameters:
    -----------
    y_true : array
        True labels
    y_proba : array
        Predicted probabilities
    metric : str
        'f1', 'precision', 'recall', or 'business' (custom)
    
    Returns:
    --------
    optimal_threshold : float
    metrics_at_threshold : dict
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'business':
            # Example: Maximize (TP * value) - (FP * cost)
            # Customize based on your business case
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            score = tp * 100 - fp * 10  # $100 per saved customer, $10 per false alarm
        
        scores.append(score) # type: ignore
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'precision': precision_score(y_true, y_pred_optimal),
        'recall': recall_score(y_true, y_pred_optimal),
        'f1': f1_score(y_true, y_pred_optimal),
        'confusion_matrix': confusion_matrix(y_true, y_pred_optimal)
    }
    
    # Plot threshold tuning
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, scores, linewidth=2)
    ax.axvline(optimal_threshold, color='red', linestyle='--', 
               label=f'Optimal = {optimal_threshold:.3f}')
    ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default = 0.5')
    ax.set_xlabel('Threshold')
    ax.set_ylabel(f'{metric.capitalize()} Score')
    ax.set_title(f'Threshold Tuning - Optimizing {metric.capitalize()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"\n{'='*70}")
    print(f"OPTIMAL THRESHOLD ANALYSIS (optimizing {metric})")
    print(f"{'='*70}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    plt.show()
    
    return optimal_threshold, metrics, fig

# ============================================================================
# Log model to mlflow
# ============================================================================
def log_model_to_mlflow(model, model_name, params, cv_results, eval_results, X_train):
    """Log model, parameters, and metrics to MLflow"""
    
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)
        
        # Log CV results
        mlflow.log_metric("cv_mean_auc", cv_results['mean_test_auc'])
        mlflow.log_metric("cv_std_auc", cv_results['std_test_auc'])
        
        # Log train/test metrics
        mlflow.log_metric("train_auc", eval_results['train_auc'])
        mlflow.log_metric("test_auc", eval_results['test_auc'])
        mlflow.log_metric("overfit", eval_results['train_auc'] - eval_results['test_auc'])
        
        # Create input example (first row of training data)
        input_example = X_train.iloc[:1] if hasattr(X_train, 'iloc') else X_train[:1]
        
        # Log model with signature and input example
        mlflow.sklearn.log_model(  # pyright: ignore[reportPrivateImportUsage]
            sk_model=model, 
            artifact_path="model",  # Changed from positional to keyword arg
            input_example=input_example
        )

        # Log figure
        mlflow.log_figure(eval_results['figure'], "evaluation_plots.png")
        
        print(f"Logged {model_name} to MLflow")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_logistic_regression_coefficients(model, feature_names, model_name, top_n=20):
    """Plot coefficients for logistic regression (shows direction)"""
    
    if hasattr(model, 'named_steps'):
        lr = model.named_steps.get('regressor') or model.named_steps.get('logisticregression')
    else:
        lr = model
    
    # Get coefficients (keep sign for interpretation)
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr.coef_[0],
        'abs_coefficient': np.abs(lr.coef_[0])
    }).sort_values('abs_coefficient', ascending=False).head(top_n)
    
    # Plot with colors based on sign
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'green' for x in coef_df['coefficient']]
    plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
    plt.xlabel('Coefficient Value')
    plt.title(f'{model_name} - Top {top_n} Features')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()
    
    return coef_df


def plot_tree_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models"""
    
    if hasattr(model, 'named_steps'):
        if 'randomforestclassifier' in model.named_steps:
            clf = model.named_steps['randomforestclassifier']
        elif 'xgbclassifier' in model.named_steps:
            clf = model.named_steps['xgbclassifier']
        else:
            print("Not a tree-based model")
            return None
    else:
        clf = model
    
    # Get feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'{model_name} - Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return importance_df