"""
ML Service - Machine Learning Inference Engine
Phase 2.2: Integrated LSTM + Correlation Risk Analysis
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.correlation_risk import CorrelationRiskCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ml-service")


# ============================================================================
# LSTM Model Architecture
# ============================================================================

class AttentionLayer(nn.Module):
    """Attention mechanism over time dimension"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        attention_scores = self.attention(lstm_output)
        attention_scores = attention_scores.squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)
        return context, attention_weights


class LSTMAttentionModel(nn.Module):
    """LSTM with Attention for ICU deterioration prediction"""
    
    def __init__(self, input_size=5, hidden_size=256, num_layers=1):
        super(LSTMAttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attention_weights


# ============================================================================
# Request/Response Models 
# ============================================================================

class PredictRequest(BaseModel):
    """Risk prediction request"""
    patient_id: str = Field(..., description="Patient identifier")
    vitals_sequence: List[List[float]] = Field(
        ..., 
        description="20 timesteps of vitals [HR, SpO2, SBP, RR, Temp]",
        min_items=20,
        max_items=20
    )
    baseline: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Patient baseline from calibration"
    )
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT001",
                "vitals_sequence": [[75, 98, 120, 16, 37.0]] * 20,
                "baseline": {
                    "HR": {"mean": 75, "std": 3, "green_zone_min": 70, "green_zone_max": 80}
                },
                "timestamp": "2026-02-14T10:00:00"
            }
        }


class PredictResponse(BaseModel):
    """Comprehensive risk assessment response"""
    patient_id: str
    timestamp: str
    lstm_risk: float = Field(..., description="LSTM model deterioration risk (0-1)")
    correlation_risk: float = Field(..., description="Multivariate correlation risk (0-1)")
    combined_risk: float = Field(..., description="Combined risk score (0-1)")
    risk_level: str = Field(..., description="LOW, MODERATE, HIGH, CRITICAL")
    deviations: Dict = Field(..., description="Vital deviations from baseline")
    detected_patterns: List[Dict] = Field(..., description="Correlation patterns")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    confidence: float = Field(..., description="Prediction confidence (0-1)")


# ============================================================================
# Global State
# ============================================================================

class MLServiceState:
    """Global ML service state"""
    def __init__(self):
        self.model: Optional[LSTMAttentionModel] = None
        self.device: torch.device = torch.device('cpu')
        self.correlation_calculator: Optional[CorrelationRiskCalculator] = None
        self.model_loaded: bool = False


state = MLServiceState()


# ============================================================================
# Initialize FastAPI
# ============================================================================
app = FastAPI(
    title="VitalX ML Service",
    description="LSTM + Multivariate Correlation Risk Analysis for ICU Monitoring",
    version="2.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Load LSTM model and initialize correlation calculator"""
    
    kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')
    model_path = os.getenv('MODEL_PATH', '../best_lstm_model.pth')
    
    logger.info("=" * 70)
    logger.info("VitalX ML Service Starting (Phase 2.2)")
    logger.info("=" * 70)
    
    # Initialize correlation calculator
    state.correlation_calculator = CorrelationRiskCalculator(sensitivity=1.0)
    logger.info("✓ Correlation Risk Calculator initialized")
    
    # Load LSTM model
    try:
        # Find model path
        model_file = Path(model_path)
        if not model_file.exists():
            # Try alternative paths
            alt_paths = [
                Path(__file__).parent.parent.parent / "best_lstm_model.pth",
                Path("./best_lstm_model.pth"),
                Path("../best_lstm_model.pth")
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_file = alt_path
                    break
        
        if model_file.exists():
            state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state.model = LSTMAttentionModel(
                input_size=5,
                hidden_size=256,
                num_layers=1
            )
            state.model.load_state_dict(torch.load(model_file, map_location=state.device))
            state.model.to(state.device)
            state.model.eval()
            state.model_loaded = True
            
            logger.info(f"✓ LSTM model loaded from: {model_file}")
            logger.info(f"✓ Using device: {state.device}")
            
            # Count parameters
            total_params = sum(p.numel() for p in state.model.parameters())
            logger.info(f"✓ Model parameters: {total_params:,}")
        else:
            logger.warning("⚠ LSTM model not found - using correlation-only mode")
            logger.warning(f"  Searched: {model_path}")
            state.model_loaded = False
    
    except Exception as e:
        logger.error(f"❌ Failed to load LSTM model: {e}")
        logger.warning("⚠ Falling back to correlation-only risk assessment")
        state.model_loaded = False
    
    logger.info(f"Connected to Kafka: {kafka_servers}")
    logger.info("=" * 70)
    logger.info("ML Service ready for inference requests")
    logger.info("=" * 70)

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "vitalx-ml-service",
            "version": "2.0.0",
            "model_loaded": state.model_loaded,
            "correlation_enabled": state.correlation_calculator is not None,
            "message": "VitalX ML Service is running"
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "service": "VitalX ML Service",
            "status": "operational",
            "version": "2.0.0",
            "features": [
                "LSTM deterioration prediction",
                "Multivariate correlation analysis",
                "Baseline deviation detection",
                "Cross-vital dependency tracking"
            ]
        }
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_risk(request: PredictRequest):
    """
    Predict deterioration risk using LSTM + Correlation Analysis
    
    Args:
        request: Vitals sequence (20 timesteps) + baseline metrics
    
    Returns:
        Comprehensive risk assessment with LSTM and correlation risk
    """
    
    timestamp = request.timestamp or datetime.now().isoformat()
    
    # Step 1: LSTM prediction (if model loaded)
    lstm_risk = 0.0
    if state.model_loaded and state.model is not None:
        try:
            # Convert to tensor
            vitals_array = np.array(request.vitals_sequence, dtype=np.float32)
            vitals_tensor = torch.FloatTensor(vitals_array).unsqueeze(0).to(state.device)
            
            # LSTM inference
            with torch.no_grad():
                output, attention_weights = state.model(vitals_tensor)
                lstm_risk = torch.sigmoid(output).item()
            
            logger.info(f"Patient {request.patient_id}: LSTM risk = {lstm_risk:.3f}")
        
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            lstm_risk = 0.0
    
    # Step 2: Correlation risk analysis (if baseline provided)
    correlation_risk = 0.0
    deviations = {}
    detected_patterns = []
    risk_factors = []
    confidence = 0.5
    
    if request.baseline and state.correlation_calculator:
        try:
            # Get current vitals (last timestep)
            current_vitals_list = request.vitals_sequence[-1]
            current_vitals = {
                "HR": current_vitals_list[0],
                "SpO2": current_vitals_list[1],
                "SBP": current_vitals_list[2],
                "RR": current_vitals_list[3],
                "Temp": current_vitals_list[4]
            }
            
            # Compute correlation risk
            assessment = state.correlation_calculator.compute_risk(
                vitals=current_vitals,
                baseline=request.baseline,
                timestamp=timestamp
            )
            
            correlation_risk = assessment.overall_risk
            deviations = {name: {
                "value": dev.value,
                "baseline_mean": dev.baseline_mean,
                "deviation_sigma": round(dev.deviation_sigma, 2),
                "direction": dev.direction,
                "in_green_zone": dev.in_green_zone
            } for name, dev in assessment.deviations.items()}
            
            detected_patterns = [
                {
                    "vitals": f"{p.vital1} ↔ {p.vital2}",
                    "type": p.expected_correlation,
                    "is_anomalous": p.is_anomalous,
                    "risk_contribution": round(p.risk_contribution, 3),
                    "explanation": p.explanation
                }
                for p in assessment.detected_patterns
            ]
            
            risk_factors = assessment.risk_factors
            confidence = assessment.confidence
            
            logger.info(f"Patient {request.patient_id}: Correlation risk = {correlation_risk:.3f}")
        
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            correlation_risk = 0.0
    
    # Step 3: Combine risks (weighted average)
    if state.model_loaded and request.baseline:
        # Both LSTM and correlation available
        combined_risk = 0.6 * lstm_risk + 0.4 * correlation_risk
    elif state.model_loaded:
        # LSTM only
        combined_risk = lstm_risk
    elif request.baseline:
        # Correlation only
        combined_risk = correlation_risk
    else:
        # Fallback
        combined_risk = 0.0
    
    # Step 4: Determine risk level
    if combined_risk < 0.3:
        risk_level = "LOW"
    elif combined_risk < 0.5:
        risk_level = "MODERATE"
    elif combined_risk < 0.7:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"
    
    logger.info(f"Patient {request.patient_id}: Combined risk = {combined_risk:.3f} ({risk_level})")
    
    return PredictResponse(
        patient_id=request.patient_id,
        timestamp=timestamp,
        lstm_risk=round(lstm_risk, 3),
        correlation_risk=round(correlation_risk, 3),
        combined_risk=round(combined_risk, 3),
        risk_level=risk_level,
        deviations=deviations,
        detected_patterns=detected_patterns,
        risk_factors=risk_factors if risk_factors else ["No baseline provided for correlation analysis"],
        confidence=round(confidence, 2)
    )