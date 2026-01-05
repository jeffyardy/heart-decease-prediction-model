"""
FastAPI application for heart disease prediction model serving
"""
import os
import sys
import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data.preprocess import HeartDiseasePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'heart_disease_predictions_total',
    'Total number of predictions made',
    ['prediction']
)

PREDICTION_LATENCY = Histogram(
    'heart_disease_prediction_latency_seconds',
    'Prediction latency in seconds'
)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps API for predicting heart disease risk",
    version="1.0.0"
)

# Global variables for model and preprocessor
model = None
preprocessor = None


class HeartDiseaseInput(BaseModel):
    """Input schema for prediction request"""
    age: float = Field(..., description="Age in years", ge=0, le=120)
    sex: float = Field(..., description="Sex (1=male, 0=female)")
    cp: float = Field(..., description="Chest pain type (0-3)")
    trestbps: float = Field(..., description="Resting blood pressure", ge=0)
    chol: float = Field(..., description="Serum cholesterol", ge=0)
    fbs: float = Field(..., description="Fasting blood sugar > 120 (1=true, 0=false)")
    restecg: float = Field(..., description="Resting electrocardiographic results (0-2)")
    thalach: float = Field(..., description="Maximum heart rate achieved", ge=0)
    exang: float = Field(..., description="Exercise induced angina (1=yes, 0=no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise", ge=0)
    slope: float = Field(..., description="Slope of peak exercise ST segment (0-2)")
    ca: float = Field(..., description="Number of major vessels colored by flourosopy (0-3)", ge=0)
    thal: float = Field(..., description="Thalassemia (1-3)")


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    prediction: int = Field(..., description="Predicted class (0=no disease, 1=disease)")
    probability: float = Field(..., description="Probability of heart disease", ge=0, le=1)
    confidence: str = Field(..., description="Confidence level")


def load_model():
    """Load model and preprocessor"""
    global model, preprocessor
    
    model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
    
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        
        model = joblib.load(model_path)
        preprocessor = HeartDiseasePreprocessor.load(preprocessor_path)
        logger.info(f"Model and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Heart Disease Prediction API...")
    load_model()
    logger.info("API ready to serve predictions")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: HeartDiseaseInput):
    """
    Predict heart disease risk
    
    Args:
        input_data: Patient health data
        
    Returns:
        Prediction with confidence
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with PREDICTION_LATENCY.time():
        try:
            # Convert input to dataframe
            input_dict = input_data.dict()
            df = pd.DataFrame([input_dict])
            
            # Preprocess
            df_processed = preprocessor.transform(df)
            
            # Make prediction
            prediction = model.predict(df_processed)[0]
            probability = model.predict_proba(df_processed)[0][1]
            
            # Determine confidence
            if probability < 0.3:
                confidence = "Low"
            elif probability < 0.7:
                confidence = "Medium"
            else:
                confidence = "High"
            
            # Update metrics
            PREDICTION_COUNTER.labels(prediction=int(prediction)).inc()
            
            # Log prediction
            logger.info(f"Prediction made: {prediction} (probability: {probability:.4f})")
            
            return PredictionResponse(
                prediction=int(prediction),
                probability=float(probability),
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

