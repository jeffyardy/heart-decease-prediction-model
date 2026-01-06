"""
Model training script with MLflow integration
Trains multiple models and tracks experiments
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data.preprocess import HeartDiseasePreprocessor


def train_logistic_regression(X_train, y_train, X_test, y_test, experiment_name="heart_disease"):
    """Train Logistic Regression model"""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="LogisticRegression"):
        # Model parameters
        params = {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs"
        }
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "cv_roc_auc_mean": cv_mean,
            "cv_roc_auc_std": cv_std
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['No Disease', 'Disease'], 
                            columns=['No Disease', 'Disease'])
        cm_path = "confusion_matrix_lr.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        print(f"Logistic Regression - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return model, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }


def train_random_forest(X_train, y_train, X_test, y_test, experiment_name="heart_disease"):
    """Train Random Forest model"""
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="RandomForest"):
        # Model parameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1
        }
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "cv_roc_auc_mean": cv_mean,
            "cv_roc_auc_std": cv_std
        })
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(model.feature_importances_))],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance_path = "feature_importance_rf.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        os.remove(importance_path)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['No Disease', 'Disease'], 
                            columns=['No Disease', 'Disease'])
        cm_path = "confusion_matrix_rf.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        print(f"Random Forest - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        return model, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }


def main():
    """Main training function"""
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_path = "data/raw/heart_disease.csv"
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        print("Please run the download script first: python src/data/download_data.py")
        return
    
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.data.preprocess import load_and_preprocess_data
    df, preprocessor = load_and_preprocess_data(data_path)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Save preprocessor
    os.makedirs("models", exist_ok=True)
    preprocessor.save("models/preprocessor.pkl")
    print("Preprocessor saved to models/preprocessor.pkl")
    
    # Train models
    print("\nTraining Logistic Regression...")
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print("\nTraining Random Forest...")
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    
    # Compare models
    print("\n" + "="*50)
    print("Model Comparison:")
    print("="*50)
    print(f"Logistic Regression - ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"Random Forest - ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    
    # Select best model based on ROC-AUC
    if lr_metrics['roc_auc'] > rf_metrics['roc_auc']:
        best_model = lr_model
        best_name = "LogisticRegression"
        print(f"\nBest model: Logistic Regression (ROC-AUC: {lr_metrics['roc_auc']:.4f})")
    else:
        best_model = rf_model
        best_name = "RandomForest"
        print(f"\nBest model: Random Forest (ROC-AUC: {rf_metrics['roc_auc']:.4f})")
    
    # Save best model
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\nBest model saved to models/best_model.pkl")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

