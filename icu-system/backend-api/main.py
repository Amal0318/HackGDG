"""
ICU Deterioration Prediction Microservice
Real-time risk assessment using LSTM + Attention model
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import uvicorn
from contextlib import asynccontextmanager

# =========================================================
# MODEL ARCHITECTURE
# =========================================================

class AttentionLayer(nn.Module):
    """Attention mechanism over time dimension"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        attention_scores = self.attention(lstm_output).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_output
        ).squeeze(1)
        return context, attention_weights


class LSTMAttentionModel(nn.Module):
    """LSTM with Attention for ICU deterioration prediction"""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, attention_weights = self.attention(lstm_out)
        output = self.fc(context)
        return output, attention_weights


# =========================================================
# GLOBAL STATE
# =========================================================

class ModelState:
    """Container for model and scaler"""
    model: LSTMAttentionModel = None
    scaler = None
    device: torch.device = None


state = ModelState()

# =========================================================
# CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 20
NUM_FEATURES = 5
HIDDEN_SIZE = 256
MODEL_PATH = "best_lstm_model.pth"
SCALER_PATH = "feature_scaler.pkl"

# =========================================================
# LIFESPAN MANAGEMENT
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler at startup"""
    
    print("\n" + "="*60)
    print("INITIALIZING ICU PREDICTION SERVICE")
    print("="*60)
    
    # Detect device
    state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {state.device}")
    
    # Load model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        state.model = LSTMAttentionModel(
            input_size=NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=1
        )
        state.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=state.device)
        )
        state.model = state.model.to(state.device)
        state.model.eval()
        
        total_params = sum(p.numel() for p in state.model.parameters())
        print(f"[OK] Model loaded ({total_params:,} parameters)")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise
    
    # Load scaler (optional - use identity if not available)
    try:
        print(f"Loading feature scaler from {SCALER_PATH}...")
        with open(SCALER_PATH, 'rb') as f:
            state.scaler = pickle.load(f)
        print(f"[OK] Scaler loaded")
    except FileNotFoundError:
        print("[WARNING] Scaler not found - using identity scaling")
        state.scaler = None
    except Exception as e:
        print(f"[WARNING] Failed to load scaler: {e} - using identity scaling")
        state.scaler = None
    
    print("="*60)
    print("SERVICE READY")
    print("="*60 + "\n")
    
    yield
    
    # Cleanup (if needed)
    print("\nShutting down service...")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="ICU Deterioration Prediction API",
    description="Real-time risk assessment using LSTM + Attention",
    version="1.0.0",
    lifespan=lifespan
)

# =========================================================
# REQUEST/RESPONSE SCHEMAS
# =========================================================

class PredictionRequest(BaseModel):
    """
    Request schema for prediction endpoint
    
    sequence: List of timesteps, each containing vital signs
    Format: [[hr, spo2, sbp, rr, temp], ...]
    """
    sequence: List[List[float]] = Field(
        ...,
        description="Sequence of vital sign measurements",
        example=[
            [85.2, 97.5, 125.0, 16.5, 37.2],
            [86.1, 97.3, 124.5, 16.8, 37.3],
            # ... 20 timesteps total
        ]
    )
    
    @validator('sequence')
    def validate_sequence(cls, v):
        """Validate sequence shape"""
        if len(v) != SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence must contain exactly {SEQUENCE_LENGTH} timesteps, "
                f"got {len(v)}"
            )
        
        for i, timestep in enumerate(v):
            if len(timestep) != NUM_FEATURES:
                raise ValueError(
                    f"Each timestep must contain exactly {NUM_FEATURES} features, "
                    f"timestep {i} has {len(timestep)}"
                )
        
        return v


class PredictionResponse(BaseModel):
    """Response schema with risk probability"""
    risk: float = Field(
        ...,
        description="Deterioration risk probability (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.8245
    )


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(default="ok")
    model_loaded: bool = Field(default=True)
    device: str = Field(default="cpu")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def preprocess_input(sequence: List[List[float]]) -> torch.Tensor:
    """
    Convert input sequence to normalized tensor
    
    Args:
        sequence: Raw vital sign measurements
    
    Returns:
        Preprocessed tensor ready for model
    """
    # Convert to numpy array
    arr = np.array(sequence, dtype=np.float32)
    
    # Apply scaling if scaler is available
    if state.scaler is not None:
        # Reshape for scaling: (timesteps * features) -> scale -> reshape back
        n_timesteps, n_features = arr.shape
        arr_flat = arr.reshape(-1, n_features)
        arr_scaled = state.scaler.transform(arr_flat)
        arr = arr_scaled.reshape(n_timesteps, n_features)
    
    # Convert to torch tensor
    tensor = torch.FloatTensor(arr)
    
    # Add batch dimension: (seq_len, features) -> (1, seq_len, features)
    tensor = tensor.unsqueeze(0)
    
    # Move to device
    tensor = tensor.to(state.device)
    
    return tensor


def run_inference(input_tensor: torch.Tensor) -> float:
    """
    Run model inference and return risk probability
    
    Args:
        input_tensor: Preprocessed input tensor
    
    Returns:
        Risk probability between 0 and 1
    """
    with torch.no_grad():
        # Forward pass
        output, attention_weights = state.model(input_tensor)
        
        # Apply sigmoid to get probability
        risk = torch.sigmoid(output).item()
        
        # Round to 4 decimal places
        risk = round(risk, 4)
    
    return risk


# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns service status and model information
    """
    return HealthResponse(
        status="ok",
        model_loaded=state.model is not None,
        device=str(state.device)
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict ICU deterioration risk
    
    Input: 20 timesteps of vital signs
    - heart_rate (bpm)
    - spo2 (%)
    - systolic_bp (mmHg)
    - respiratory_rate (breaths/min)
    - temperature (°C)
    
    Output: Risk probability (0.0 = stable, 1.0 = high risk)
    """
    try:
        # Preprocess input
        input_tensor = preprocess_input(request.sequence)
        
        # Run inference
        risk = run_inference(input_tensor)
        
        return PredictionResponse(risk=risk)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "ICU Deterioration Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        },
        "model": {
            "type": "LSTM + Attention",
            "input_shape": f"({SEQUENCE_LENGTH}, {NUM_FEATURES})",
            "features": [
                "heart_rate",
                "spo2",
                "systolic_bp",
                "respiratory_rate",
                "temperature"
            ]
        }
    }


# =========================================================
# MAIN ENTRY POINT
# =========================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING ICU DETERIORATION PREDICTION SERVICE")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
