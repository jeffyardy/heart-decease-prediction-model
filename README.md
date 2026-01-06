# MLOps Assignment: Heart Disease Prediction

End-to-end machine learning solution for predicting heart disease risk using MLOps best practices.

## Project Overview

This project implements a complete MLOps pipeline for heart disease prediction, including:
- Data acquisition and EDA
- Feature engineering and model development
- Experiment tracking with MLflow
- Model packaging and reproducibility
- CI/CD pipelines with GitHub Actions
- Docker containerization
- Kubernetes deployment
- Monitoring and logging

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── download_data.py      # Data acquisition script
│   │   └── preprocess.py          # Data preprocessing pipeline
│   ├── models/
│   │   └── train.py               # Model training with MLflow
│   └── api/
│       └── app.py                 # FastAPI application
├── tests/
│   ├── test_preprocessing.py      # Unit tests for preprocessing
│   ├── test_model.py              # Unit tests for models
│   └── test_api.py                # Unit tests for API
├── notebooks/
│   └── 01_EDA.ipynb               # Exploratory Data Analysis
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment
│   ├── service.yaml               # Kubernetes service
│   └── ingress.yaml               # Kubernetes ingress
├── scripts/
│   ├── setup.sh                   # Setup script
│   ├── run_api.sh                 # Run API locally
│   └── test_api.sh                # Test API endpoints
├── .github/
│   └── workflows/
│       └── ci_cd.yml              # CI/CD pipeline
├── Dockerfile                     # Docker container definition
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker (for containerization)
- Kubernetes cluster (optional, for deployment)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Assignment1
   ```

2. **Run setup script**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

   Or manually:
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Download dataset
   python src/data/download_data.py
   ```

3. **Train models**
   ```bash
   python src/models/train.py
   ```

## Usage

### Running the API Locally

```bash
# Using the script
chmod +x scripts/run_api.sh
./scripts/run_api.sh

# Or directly
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics

### Making Predictions

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## Docker

### Build Docker Image

```bash
docker build -t heart-disease-api:latest .
```

### Run Docker Container

```bash
docker run -d -p 8000:8000 --name heart-disease-api heart-disease-api:latest
```

### Test Docker Container

```bash
docker exec -it heart-disease-api curl http://localhost:8000/health
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (Minikube, GKE, EKS, AKS, or Docker Desktop)
- kubectl configured

### Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy application
kubectl apply -f k8s/deployment.yaml

# Check status
kubectl get pods -n mlops-assignment
kubectl get services -n mlops-assignment
```

### Access the API

For local Kubernetes (Minikube/Docker Desktop):
```bash
# Get service URL
minikube service heart-disease-api-service -n mlops-assignment
```

For cloud Kubernetes:
- Use the LoadBalancer external IP
- Or configure Ingress as per your cloud provider

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci_cd.yml`) includes:

1. **Linting**: Code quality checks with flake8 and black
2. **Testing**: Unit tests with pytest and coverage
3. **Training**: Model training with MLflow tracking
4. **Docker Build**: Container build and test

The pipeline runs automatically on push to `main` or `develop` branches.

## Monitoring

The API includes Prometheus metrics at `/metrics` endpoint:

- `heart_disease_predictions_total`: Total predictions made
- `heart_disease_prediction_latency_seconds`: Prediction latency

### Grafana Dashboard (Optional)

You can configure Grafana to scrape metrics from the `/metrics` endpoint for visualization.

## Model Information

### Models Trained

1. **Logistic Regression**: Baseline model with good interpretability
2. **Random Forest**: Ensemble model with feature importance

### Metrics Tracked

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Cross-validation scores

All metrics are logged to MLflow for experiment tracking.

## EDA and Modeling Choices

### Data Preprocessing

- Missing value imputation using median strategy
- Feature scaling using StandardScaler
- Target binarization (0 = no disease, 1 = disease)

### Model Selection

Models are evaluated using cross-validation and ROC-AUC score. The best model is selected based on performance.

## Experiment Tracking

MLflow is used for experiment tracking. To view experiments:

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

## Troubleshooting

### Model files not found
```bash
# Train the model first
python src/models/train.py
```

### Port already in use
```bash
# Use a different port
uvicorn src.api.app:app --host 0.0.0.0 --port 8001
```

### Docker build fails
```bash
# Check Docker is running
docker ps

# Rebuild without cache
docker build --no-cache -t heart-disease-api:latest .
```

## License

This project is for educational purposes as part of MLOps coursework.

## Author

MLOps Assignment - group 118

