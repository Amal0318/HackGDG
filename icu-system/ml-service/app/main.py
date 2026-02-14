"""
VitalX ML Service - FastAPI Inference Engine
=============================================

Real-time deterioration prediction using trained LSTM model.

Endpoints:
- POST /predict - Single sequence prediction
- POST /predict/batch - Batch prediction
- GET /health - Health check
- GET /model/info - Model information
"""

import os
import sys
import logging
import json
import pickle
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMAttentionModel, LogisticRegressionFallback
from app.kafka_consumer import start_ml_consumer, stop_ml_consumer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml-service")

# Initialize FastAPI app
app = FastAPI(
    title="VitalX ML Service",
    description="LSTM-based ICU Deterioration Prediction Engine",
    version="1.0.0"
)

# ==============================
# GLOBAL STATE
# ==============================


class ModelState:
    """Global model state."""
    lstm_model = None
    fallback_model = None
    scaler = None
    config = None
    # Prefer CUDA GPU, fallback to CPU for inference only
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_loaded = False
    fallback_loaded = False
    
    def __init__(self):
        if torch.cuda.is_available():
            logger.info(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("⚠️  CUDA GPU not available, using CPU (slower inference)")

state = ModelState()


# ==============================
# PYDANTIC MODELS
# ==============================


class VitalSequence(BaseModel):
    """Single vital sign sequence (60 timesteps × 14 features)."""
    sequence: List[List[float]] = Field(
        ..., description="60x14 array of vitals"
    )
    patient_id: Optional[str] = Field(
        None, description="Patient identifier"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "sequence": [
                    [85.0, 120.0, 80.0, 98.0, 16.0, 37.0, 0.71]
                    + [0.0] * 7
                ] * 60,
                "patient_id": "P001"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    patient_id: Optional[str]
    risk_score: float = Field(
        ..., description="Deterioration probability [0, 1]"
    )
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    timestamp: str
    model_used: str = Field(..., description="lstm or fallback")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    sequences: List[VitalSequence]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    total: int


class ModelInfo(BaseModel):
    """Model information."""
    lstm_loaded: bool
    fallback_loaded: bool
    device: str
    lstm_parameters: Optional[int]
    config: Optional[dict]


# ==============================
# MODEL LOADING
# ==============================
def load_lstm_model():
    """Load trained LSTM model."""
    try:
        model_path = "model.pth"
        config_path = "feature_config.json"
        
        if not os.path.exists(model_path):
            logger.warning(f"LSTM model not found at {model_path}")
            return False
        
        # Load config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                state.config = json.load(f)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=state.device, weights_only=False)
        
        # Create model
        model_config = checkpoint.get('config', {})
        state.lstm_model = LSTMAttentionModel(
            input_size=model_config.get('input_size', 14),
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.3)
        )
        
        # Load weights
        state.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        state.lstm_model.to(state.device)
        state.lstm_model.eval()
        
        state.model_loaded = True
        logger.info(f"✅ LSTM model loaded successfully from {model_path}")
        logger.info(f"   Device: {state.device}")
        logger.info(f"   Best ROC-AUC: {checkpoint.get('best_roc_auc', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load LSTM model: {e}")
        return False


def load_fallback_model():
    """Load logistic regression fallback."""
    try:
        fallback_path = "../training/saved_models/fallback_logistic.pkl"
        
        if not os.path.exists(fallback_path):
            logger.warning(f"Fallback model not found at {fallback_path}")
            return False
        
        with open(fallback_path, 'rb') as f:
            data = pickle.load(f)
        
        state.fallback_model = LogisticRegressionFallback()
        state.fallback_model.model = data['model']
        state.fallback_model.scaler = data['scaler']
        state.fallback_model.is_fitted = data['is_fitted']
        
        state.fallback_loaded = True
        logger.info(f"✅ Fallback model loaded successfully from {fallback_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load fallback model: {e}")
        return False


def load_scaler():
    """Load feature scaler."""
    try:
        scaler_path = "scaler.pkl"
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                state.scaler = pickle.load(f)
            logger.info("✅ Feature scaler loaded")
            return True
        else:
            logger.warning("⚠️  No scaler found, using raw features")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to load scaler: {e}")
        return False


# ==============================
# STARTUP
# ==============================
@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("=" * 70)
    logger.info("VitalX ML Service Starting...")
    logger.info("=" * 70)
    
    # Load models
    load_scaler()
    load_lstm_model()
    load_fallback_model()
    
    if not state.model_loaded and not state.fallback_loaded:
        logger.error("❌ No models loaded! Service may not function correctly.")
    else:
        # Start Kafka consumer for real-time predictions
        logger.info("Starting Kafka consumer for real-time predictions...")
        try:
            start_ml_consumer(predict_fn=lambda seq: predict_lstm(seq) if state.model_loaded else 0.0)
            logger.info("✅ Kafka consumer started")
        except Exception as e:
            logger.error(f"❌ Failed to start Kafka consumer: {e}")
    
    logger.info("=" * 70)
    logger.info("ML Service Ready")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down ML Service...")
    stop_ml_consumer()
    logger.info("✅ ML Service stopped")


# ==============================
# HELPER FUNCTIONS
# ==============================
def validate_sequence(sequence: List[List[float]]) -> bool:
    """Validate sequence shape."""
    if len(sequence) != 60:
        return False
    if not all(len(timestep) == 14 for timestep in sequence):
        return False
    return True


def classify_risk(risk_score: float) -> str:
    """Classify risk level."""
    if risk_score < 0.3:
        return "LOW"
    elif risk_score < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_lstm(sequence: np.ndarray) -> float:
    """Predict using LSTM model."""
    # Convert to tensor
    X = torch.FloatTensor(sequence).unsqueeze(0).to(state.device)  # (1, 60, 14)
    
    # Predict
    with torch.no_grad():
        output, _ = state.lstm_model(X)
    
    risk_score = output.item()
    return risk_score


def predict_fallback(sequence: np.ndarray) -> float:
    """Predict using fallback model."""
    # Reshape for fallback
    X = sequence.reshape(1, -1)  # (1, 60*14)
    
    # Predict
    risk_score = state.fallback_model.predict_proba(X)[0]
    
    return risk_score


# ==============================
# ENDPOINTS
# ==============================
@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        status_code=200,
        content={
            "service": "VitalX ML Service",
            "version": "1.0.0",
            "status": "operational",
            "lstm_loaded": state.model_loaded,
            "fallback_loaded": state.fallback_loaded
        }
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = "healthy" if (state.model_loaded or state.fallback_loaded) else "degraded"
    
    return JSONResponse(
        status_code=200,
        content={
            "status": status,
            "service": "ml-service",
            "lstm_available": state.model_loaded,
            "fallback_available": state.fallback_loaded,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    lstm_params = None
    if state.lstm_model:
        lstm_params = sum(p.numel() for p in state.lstm_model.parameters())
    
    return ModelInfo(
        lstm_loaded=state.model_loaded,
        fallback_loaded=state.fallback_loaded,
        device=str(state.device),
        lstm_parameters=lstm_params,
        config=state.config
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(vital_sequence: VitalSequence):
    """
    Predict deterioration risk for a single sequence.
    
    Input: 60-second vital sign sequence (60 × 14)
    Output: Risk score [0, 1]
    """
    # Validate
    if not validate_sequence(vital_sequence.sequence):
        raise HTTPException(
            status_code=400,
            detail="Invalid sequence shape. Expected (60, 14)"
        )
    
    # Convert to numpy
    sequence = np.array(vital_sequence.sequence, dtype=np.float32)
    
    # Predict
    try:
        if state.model_loaded:
            risk_score = predict_lstm(sequence)
            model_used = "lstm"
        elif state.fallback_loaded:
            risk_score = predict_fallback(sequence)
            model_used = "fallback"
        else:
            raise HTTPException(status_code=503, detail="No model available")
        
        # Ensure risk_score is in [0, 1]
        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        
        return PredictionResponse(
            patient_id=vital_sequence.patient_id,
            risk_score=risk_score,
            risk_level=classify_risk(risk_score),
            timestamp=datetime.now().isoformat(),
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple sequences.
    """
    predictions = []
    
    for vital_seq in request.sequences:
        try:
            pred = await predict(vital_seq)
            predictions.append(pred)
        except HTTPException as e:
            # Skip invalid sequences
            logger.warning(f"Skipping invalid sequence: {e.detail}")
            continue
    
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions)
    )


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
