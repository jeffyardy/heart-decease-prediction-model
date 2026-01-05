"""
Unit tests for model training and evaluation
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from src.data.preprocess import HeartDiseasePreprocessor


@pytest.fixture
def prepared_data():
    """Create prepared dataset for model testing"""
    # Generate synthetic data that mimics heart disease dataset
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(100, 600, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 220, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
    })
    
    # Create target with some relationship to features
    y = ((X['age'] > 60) | (X['chol'] > 250) | (X['trestbps'] > 140)).astype(int)
    y = pd.Series(y)
    
    # Preprocess
    preprocessor = HeartDiseasePreprocessor()
    X_processed = preprocessor.fit_transform(X)
    X_processed['target'] = y
    
    return X_processed.drop('target', axis=1), y


def test_logistic_regression_training(prepared_data):
    """Test Logistic Regression model training"""
    X, y = prepared_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Test prediction
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy > 0.5  # Should achieve reasonable accuracy
    assert len(y_pred) == len(y_test)


def test_random_forest_training(prepared_data):
    """Test Random Forest model training"""
    X, y = prepared_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Test prediction
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    assert accuracy > 0.5
    assert 0 <= roc_auc <= 1
    assert len(y_pred) == len(y_test)


def test_model_prediction_shape(prepared_data):
    """Test that model predictions have correct shape"""
    X, y = prepared_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape[0] == len(y_test)
    assert y_pred_proba.shape[1] == 2  # Binary classification


def test_model_feature_importance(prepared_data):
    """Test that Random Forest has feature importance"""
    X, y = prepared_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    assert hasattr(model, 'feature_importances_')
    assert len(model.feature_importances_) == X_train.shape[1]
    assert all(imp >= 0 for imp in model.feature_importances_)

