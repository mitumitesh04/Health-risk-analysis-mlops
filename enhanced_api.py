import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import joblib
import numpy as np
import json
import logging
from datetime import datetime
from typing import List, Optional
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and metadata
def load_model_artifacts():
    """Load model and associated artifacts"""
    try:
        model = joblib.load('models/health_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        logger.info("✅ Model artifacts loaded successfully")
        return model, scaler, metadata
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None, None, None

model, scaler, metadata = load_model_artifacts()

# FastAPI app
app = FastAPI(
    title="Health Risk Prediction API",
    description="Production-ready API for health risk assessment",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class HealthInput(BaseModel):
    heart_rate: float
    steps_daily: float
    sleep_hours: float
    age: int
    
    @validator('heart_rate')
    def validate_heart_rate(cls, v):
        if not 40 <= v <= 150:
            raise ValueError('Heart rate must be between 40-150 bpm')
        return v
    
    @validator('steps_daily')
    def validate_steps(cls, v):
        if not 0 <= v <= 50000:
            raise ValueError('Steps must be between 0-50000')
        return v
    
    @validator('sleep_hours')
    def validate_sleep(cls, v):
        if not 3 <= v <= 12:
            raise ValueError('Sleep hours must be between 3-12')
        return v
    
    @validator('age')
    def validate_age(cls, v):
        if not 18 <= v <= 100:
            raise ValueError('Age must be between 18-100')
        return v

class HealthOutput(BaseModel):
    risk_level: str
    probability: float
    confidence: str
    recommendations: List[str]
    timestamp: str

class BatchHealthInput(BaseModel):
    patients: List[HealthInput]

class ModelInfo(BaseModel):
    model_type: str
    version: str
    features: List[str]
    metrics: dict
    last_trained: str

# Prediction counter for monitoring
prediction_counter = 0

def log_prediction(input_data: dict, output_data: dict):
    """Log prediction for monitoring"""
    global prediction_counter
    prediction_counter += 1
    
    # In production, this would go to a proper logging system
    log_entry = {
        'prediction_id': prediction_counter,
        'timestamp': datetime.now().isoformat(),
        'input': input_data,
        'output': output_data
    }
    
    # Log to file (in production, use proper logging infrastructure)
    with open('logs/predictions.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def generate_recommendations(input_data: dict, risk_level: str) -> List[str]:
    """Generate personalized health recommendations"""
    recommendations = []
    
    if input_data['heart_rate'] > 100:
        recommendations.append("Consider stress management and consult a cardiologist")
    elif input_data['heart_rate'] < 60:
        recommendations.append("Monitor for bradycardia, consider medical evaluation")
    
    if input_data['steps_daily'] < 5000:
        recommendations.append("Increase daily physical activity - aim for 8000+ steps")
    
    if input_data['sleep_hours'] < 6:
        recommendations.append("Improve sleep hygiene - aim for 7-8 hours nightly")
    elif input_data['sleep_hours'] > 9:
        recommendations.append("Evaluate sleep quality - excessive sleep may indicate health issues")
    
    if input_data['age'] > 65 and risk_level == "High Risk":
        recommendations.append("Schedule regular health checkups with your physician")
    
    if not recommendations:
        recommendations.append("Maintain current healthy lifestyle habits")
    
    return recommendations

# API endpoints
@app.get("/")
def root():
    return {
        "message": "Health Risk Prediction API v2.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "predictions_served": prediction_counter,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfo)
def get_model_info():
    if metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")
    
    return ModelInfo(
        model_type=metadata['model_type'],
        version="2.0.0",
        features=metadata['features'],
        metrics=metadata['metrics'],
        last_trained=metadata['timestamp']
    )

@app.post("/predict", response_model=HealthOutput)
def predict_risk(data: HealthInput, background_tasks: BackgroundTasks):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        input_dict = data.dict()
        features = np.array([[
            input_dict['heart_rate'],
            input_dict['steps_daily'], 
            input_dict['sleep_hours'],
            input_dict['age']
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Determine confidence
        confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
        
        # Generate response
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        recommendations = generate_recommendations(input_dict, risk_level)
        
        result = HealthOutput(
            risk_level=risk_level,
            probability=probability,
            confidence=confidence,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction asynchronously
        background_tasks.add_task(log_prediction, input_dict, result.dict())
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
def predict_batch(data: BatchHealthInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for patient_data in data.patients:
        try:
            # This could be optimized for true batch processing
            prediction = predict_risk(patient_data, BackgroundTasks())
            results.append(prediction)
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results}

@app.get("/metrics")
def get_api_metrics():
    """Get API usage metrics"""
    return {
        "total_predictions": prediction_counter,
        "model_info": {
            "loaded": model is not None,
            "type": metadata.get('model_type') if metadata else None,
            "version": "2.0.0"
        },
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    print("🚀 Starting Enhanced Health Risk API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")