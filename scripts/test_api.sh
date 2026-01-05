#!/bin/bash
# Script to test the API endpoints

API_URL="http://localhost:8000"

echo "Testing Heart Disease Prediction API..."
echo ""

# Test root endpoint
echo "1. Testing root endpoint..."
curl -s $API_URL/ | jq .
echo ""

# Test health endpoint
echo "2. Testing health endpoint..."
curl -s $API_URL/health | jq .
echo ""

# Test prediction endpoint
echo "3. Testing prediction endpoint..."
curl -s -X POST "$API_URL/predict" \
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
  }' | jq .
echo ""

# Test metrics endpoint
echo "4. Testing metrics endpoint..."
curl -s $API_URL/metrics | head -20
echo ""

echo "API testing completed!"

