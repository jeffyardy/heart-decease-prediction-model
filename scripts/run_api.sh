#!/bin/bash
# Script to run the API locally

set -e

# Check if model files exist
if [ ! -f "models/best_model.pkl" ]; then
    echo "Model file not found. Please train the model first:"
    echo "python src/models/train.py"
    exit 1
fi

if [ ! -f "models/preprocessor.pkl" ]; then
    echo "Preprocessor file not found. Please train the model first:"
    echo "python src/models/train.py"
    exit 1
fi

# Run the API
echo "Starting Heart Disease Prediction API..."
echo "API will be available at http://localhost:8000"
echo "API documentation at http://localhost:8000/docs"
echo ""

uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

