"""
Unit tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app, load_model
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.preprocess import HeartDiseasePreprocessor
from sklearn.linear_model import LogisticRegression
from prometheus_client import CONTENT_TYPE_LATEST


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_model_files(tmp_path):
    """Create mock model and preprocessor files"""
    # Create models directory
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Create a simple preprocessor
    preprocessor = HeartDiseasePreprocessor()
    sample_data = pd.DataFrame({
        'age': [63, 37, 41],
        'sex': [1, 1, 0],
        'cp': [3, 2, 1],
        'trestbps': [145, 130, 130],
        'chol': [233, 250, 204],
        'fbs': [1, 0, 0],
        'restecg': [0, 1, 0],
        'thalach': [150, 187, 172],
        'exang': [0, 0, 0],
        'oldpeak': [2.3, 3.5, 1.4],
        'slope': [0, 0, 2],
        'ca': [0, 0, 0],
        'thal': [1, 2, 2],
        'target': [1, 0, 0]
    })
    preprocessor.fit(sample_data)
    preprocessor.save(str(model_dir / "preprocessor.pkl"))
    
    # Create a simple model
    X_train = sample_data.drop('target', axis=1)
    y_train = sample_data['target']
    X_train_processed = preprocessor.transform(X_train)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_processed, y_train)
    joblib.dump(model, str(model_dir / "best_model.pkl"))
    
    # Set environment variables
    os.environ["MODEL_PATH"] = str(model_dir / "best_model.pkl")
    os.environ["PREPROCESSOR_PATH"] = str(model_dir / "preprocessor.pkl")
    
    # Load model in app
    load_model()
    
    return model_dir


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check(client, mock_model_files):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True


def test_predict_endpoint(client, mock_model_files):
    """Test prediction endpoint"""
    input_data = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "confidence" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1


def test_predict_endpoint_invalid_input(client, mock_model_files):
    """Test prediction endpoint with invalid input"""
    # Missing required field
    input_data = {
        "age": 63,
        "sex": 1
        # Missing other fields
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_invalid_range(client, mock_model_files):
    """Test prediction endpoint with out-of-range values"""
    input_data = {
        "age": 200,  # Invalid age
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422  # Validation error


def test_metrics_endpoint(client, mock_model_files):
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST


def test_predict_updates_metrics(client, mock_model_files):
    """Test that predictions update metrics"""
    input_data = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }
    
    # Make a prediction
    client.post("/predict", json=input_data)
    
    # Check metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics_text = response.text
    assert "heart_disease_predictions_total" in metrics_text

