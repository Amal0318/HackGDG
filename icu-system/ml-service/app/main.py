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
    
    # Metrics tracking
    total_predictions = 0
    lstm_predictions = 0
    fallback_predictions = 0
    risk_scores = []
    high_risk_count = 0
    startup_time = None
    
    def __init__(self):
        if torch.cuda.is_available():
            logger.info(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            logger.warning("⚠️  CUDA GPU not available, using CPU (slower inference)")
        self.startup_time = datetime.now()

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


class ExplainedPredictionResponse(BaseModel):
    """Prediction with explainability."""
    patient_id: Optional[str]
    risk_score: float
    risk_level: str
    timestamp: str
    model_used: str
    attention_weights: Optional[List[float]] = Field(
        None, description="Attention weights for each timestep"
    )
    feature_importance: Optional[dict] = Field(
        None, description="Feature importance scores"
    )
    critical_timesteps: Optional[List[dict]] = Field(
        None, description="Top-k most critical timesteps"
    )


class ValidationRequest(BaseModel):
    """Validation request."""
    sequence: List[List[float]]


class ValidationResponse(BaseModel):
    """Validation response."""
    is_valid: bool
    warnings: List[str]
    timestamp: str


class MetricsResponse(BaseModel):
    """Service metrics."""
    total_predictions: int
    lstm_predictions: int
    fallback_predictions: int
    avg_risk_score: float
    high_risk_count: int
    uptime_seconds: float
    timestamp: str


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
        model_path = "../training/saved_models/best_model.pth"
        config_path = "../training/saved_models/feature_config.json"
        
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
        scaler_path = "../training/saved_models/scaler.pkl"
        
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
    
    logger.info("=" * 70)
    logger.info("ML Service Ready")
    logger.info("=" * 70)


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


def predict_lstm(sequence: np.ndarray, return_attention=False):
    """Predict using LSTM model."""
    # Convert to tensor
    X = torch.FloatTensor(sequence).unsqueeze(0).to(state.device)  # (1, 60, 14)
    
    # Predict
    with torch.no_grad():
        output, attention_weights = state.lstm_model(X)
    
    risk_score = output.item()
    
    if return_attention:
        attn = attention_weights.squeeze().cpu().numpy()
        return risk_score, attn
    
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
        
        # Update metrics
        state.total_predictions += 1
        if model_used == "lstm":
            state.lstm_predictions += 1
        else:
            state.fallback_predictions += 1
        state.risk_scores.append(risk_score)
        if risk_score >= 0.7:
            state.high_risk_count += 1
        
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


@app.post("/predict/explain", response_model=ExplainedPredictionResponse)
async def predict_with_explanation(vital_sequence: VitalSequence):
    """
    Predict deterioration risk with explainability (attention weights + feature importance).
    
    Returns:
    - risk_score: Deterioration probability [0, 1]
    - attention_weights: Attention scores for each timestep (60,)
    - feature_importance: Importance score for each feature
    - critical_timesteps: Top-5 most critical timesteps
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
            risk_score, attention_weights = predict_lstm(sequence, return_attention=True)
            model_used = "lstm"
            
            # Calculate feature importance
            feature_names = [
                "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
                "glucose", "ph", "lactate", "creatinine", "wbc", "hemoglobin", "platelets"
            ]
            
            # Weight each timestep by attention
            weighted_sequence = sequence * attention_weights[:, np.newaxis]  # (60, 14)
            
            # Calculate importance as weighted standard deviation
            importance = {}
            for i, feature in enumerate(feature_names):
                feature_values = weighted_sequence[:, i]
                importance[feature] = float(np.std(feature_values))
            
            # Normalize
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            # Get critical timesteps
            top_indices = np.argsort(attention_weights)[-5:][::-1]
            critical_timesteps = [
                {
                    "timestep": int(idx),
                    "seconds_ago": int(60 - idx),
                    "attention_score": float(attention_weights[idx])
                }
                for idx in top_indices
            ]
            
            # Update metrics
            state.total_predictions += 1
            state.lstm_predictions += 1
            state.risk_scores.append(risk_score)
            if risk_score >= 0.7:
                state.high_risk_count += 1
            
        elif state.fallback_loaded:
            risk_score = predict_fallback(sequence)
            model_used = "fallback"
            attention_weights = None
            importance = None
            critical_timesteps = None
            
            # Update metrics
            state.total_predictions += 1
            state.fallback_predictions += 1
            state.risk_scores.append(risk_score)
            if risk_score >= 0.7:
                state.high_risk_count += 1
        else:
            raise HTTPException(status_code=503, detail="No model available")
        
        # Ensure risk_score is in [0, 1]
        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        
        return ExplainedPredictionResponse(
            patient_id=vital_sequence.patient_id,
            risk_score=risk_score,
            risk_level=classify_risk(risk_score),
            timestamp=datetime.now().isoformat(),
            model_used=model_used,
            attention_weights=attention_weights.tolist() if attention_weights is not None else None,
            feature_importance=importance,
            critical_timesteps=critical_timesteps
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate", response_model=ValidationResponse)
async def validate_vital_signs(request: ValidationRequest):
    """
    Validate vital sign sequences without making predictions.
    
    Checks:
    - Correct shape (60, 14)
    - No NaN/Inf values
    - Values within normal physiological ranges
    """
    sequence = np.array(request.sequence, dtype=np.float32)
    warnings_list = []
    
    # Check shape
    if sequence.shape != (60, 14):
        warnings_list.append(f"Invalid shape: {sequence.shape}, expected (60, 14)")
        return ValidationResponse(
            is_valid=False,
            warnings=warnings_list,
            timestamp=datetime.now().isoformat()
        )
    
    # Check for NaN/Inf
    if np.isnan(sequence).any():
        warnings_list.append("Contains NaN values")
    if np.isinf(sequence).any():
        warnings_list.append("Contains infinite values")
    
    # Normal ranges
    normal_ranges = {
        "heart_rate": (40, 180),
        "sbp": (70, 200),
        "dbp": (40, 120),
        "map": (60, 140),
        "resp_rate": (8, 40),
        "spo2": (70, 100),
        "temperature": (35.0, 41.0),
        "glucose": (50, 500),
        "ph": (6.8, 7.8),
        "lactate": (0.5, 10.0),
        "creatinine": (0.3, 10.0),
        "wbc": (1.0, 50.0),
        "hemoglobin": (5.0, 20.0),
        "platelets": (10, 1000)
    }
    
    feature_names = [
        "heart_rate", "sbp", "dbp", "map", "resp_rate", "spo2", "temperature",
        "glucose", "ph", "lactate", "creatinine", "wbc", "hemoglobin", "platelets"
    ]
    
    # Check ranges
    for i, feature in enumerate(feature_names):
        if feature in normal_ranges:
            min_val, max_val = normal_ranges[feature]
            feature_vals = sequence[:, i]
            
            if (feature_vals < min_val).any() or (feature_vals > max_val).any():
                warnings_list.append(
                    f"{feature}: out of range [{min_val}, {max_val}] "
                    f"(min={feature_vals.min():.2f}, max={feature_vals.max():.2f})"
                )
    
    is_valid = len(warnings_list) == 0 or not np.isnan(sequence).any()
    
    return ValidationResponse(
        is_valid=is_valid,
        warnings=warnings_list,
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get service metrics for monitoring.
    
    Returns:
    - total_predictions: Total predictions made
    - lstm_predictions: Predictions using LSTM
    - fallback_predictions: Predictions using fallback
    - avg_risk_score: Average risk score
    - high_risk_count: Number of high-risk predictions
    - uptime_seconds: Service uptime
    """
    avg_risk = np.mean(state.risk_scores) if state.risk_scores else 0.0
    uptime = (datetime.now() - state.startup_time).total_seconds() if state.startup_time else 0.0
    
    return MetricsResponse(
        total_predictions=state.total_predictions,
        lstm_predictions=state.lstm_predictions,
        fallback_predictions=state.fallback_predictions,
        avg_risk_score=float(avg_risk),
        high_risk_count=state.high_risk_count,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
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
