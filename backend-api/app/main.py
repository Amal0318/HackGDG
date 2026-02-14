"""
Backend API Service - REST + WebSocket API Gateway
VitalX Phase 1.3: Rolling Baseline Updates & Stability Detection
Phase 6: Alert Acknowledgment & Analytics Integration
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from enum import Enum
import json

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx

from app.baseline_calibrator import BaselineCalibrator, BaselineMetrics
from app.intervention_tracker import InterventionTracker, InterventionType
from app.alert_manager import AlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend-api")


# ============================================================================
# Pydantic Models
# ============================================================================

class CalibrationStatus(str, Enum):
    """Patient baseline calibration status"""
    COLD_START = "cold_start"       # Initial calibration in progress
    STABLE = "stable"                # Calibration complete, baseline locked
    RECALIBRATING = "recalibrating"  # Updating baseline during stable period


class BaselineVitals(BaseModel):
    """Patient baseline vital sign statistics"""
    mean: float = Field(..., description="Mean value during baseline period")
    std: float = Field(..., description="Standard deviation")
    green_zone: tuple[float, float] = Field(..., description="Safe range (mean ± 1.5×std)")
    
    class Config:
        schema_extra = {
            "example": {
                "mean": 75.0,
                "std": 5.0,
                "green_zone": [67.5, 82.5]
            }
        }


class BaselineMetricsResponse(BaseModel):
    """Complete baseline metrics for a patient"""
    patient_id: str
    vitals: Dict[str, BaselineVitals]
    timestamp: datetime
    stability_confidence: float = Field(..., ge=0.0, le=1.0, description="0.0=unstable, 1.0=very stable")
    sample_count: int
    calibration_status: CalibrationStatus


class PatientState(BaseModel):
    """Patient state in VitalX system"""
    patient_id: str = Field(..., description="Unique patient identifier")
    baseline_vitals: Optional[Dict[str, Dict]] = Field(None, description="Calibrated baseline vital ranges")
    calibration_status: CalibrationStatus = Field(
        CalibrationStatus.COLD_START, 
        description="Current calibration state"
    )
    admission_time: datetime = Field(default_factory=datetime.now, description="When patient was admitted")
    last_update: datetime = Field(default_factory=datetime.now, description="Last state update timestamp")
    vitals_buffer: List[List[float]] = Field(default_factory=list, description="Cold-start vitals buffer")
    
    # Stability tracking for rolling baseline updates
    recent_risk_scores: List[float] = Field(default_factory=list, description="Recent risk scores for stability detection")
    last_baseline_update: Optional[datetime] = Field(None, description="When baseline was last updated")
    stable_period_start: Optional[datetime] = Field(None, description="When stable period started (risk < 0.3)")
    stable_vitals_buffer: List[List[float]] = Field(default_factory=list, description="Vitals during stable period for rolling update")
    
    # Monitoring history for LSTM prediction (need 20 timesteps)
    vitals_history: List[List[float]] = Field(default_factory=list, description="Last 20 vitals for LSTM prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT001",
                "baseline_vitals": None,
                "calibration_status": "cold_start",
                "admission_time": "2026-02-14T08:00:00",
                "last_update": "2026-02-14T08:00:00",
                "vitals_buffer": [],
                "recent_risk_scores": [],
                "last_baseline_update": None,
                "stable_period_start": None,
                "stable_vitals_buffer": []
            }
        }


class AdmitPatientRequest(BaseModel):
    """Request to admit a new patient"""
    patient_id: str = Field(..., description="Unique patient identifier")
    initial_vitals: Optional[List[float]] = Field(
        None, 
        description="Optional first vital reading [HR, SpO2, SBP, RR, Temp]"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "PT001",
                "initial_vitals": [75, 98, 120, 16, 37.0]
            }
        }


class VitalsReading(BaseModel):
    """Single vital signs reading"""
    vitals: List[float] = Field(..., min_items=5, max_items=5, description="[HR, SpO2, SBP, RR, Temp]")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "vitals": [75, 98, 120, 16, 37.0],
                "timestamp": "2026-02-14T08:00:00"
            }
        }


class InterventionRequest(BaseModel):
    """Request to log a clinical intervention"""
    type: str = Field(..., description="Intervention type (e.g., vasopressors, nebulizer)")
    dosage: Optional[str] = Field(None, description="Dosage information")
    administered_by: Optional[str] = Field(None, description="Clinician identifier")
    notes: Optional[str] = Field(None, description="Additional notes")
    timestamp: Optional[datetime] = Field(None, description="When intervention was given (default: now)")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "vasopressors",
                "dosage": "5mcg/min norepinephrine",
                "administered_by": "DR_SMITH",
                "notes": "Started for septic shock",
                "timestamp": "2026-02-14T10:30:00"
            }
        }


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}  # patient_id -> [websockets]
        self.global_connections: List[WebSocket] = []  # Dashboard connections monitoring all patients
        
    async def connect(self, websocket: WebSocket, patient_id: Optional[str] = None):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        if patient_id:
            if patient_id not in self.active_connections:
                self.active_connections[patient_id] = []
            self.active_connections[patient_id].append(websocket)
            logger.info(f"WebSocket connected for patient {patient_id}")
        else:
            self.global_connections.append(websocket)
            logger.info("Global dashboard WebSocket connected")
    
    def disconnect(self, websocket: WebSocket, patient_id: Optional[str] = None):
        """Remove a WebSocket connection"""
        if patient_id and patient_id in self.active_connections:
            if websocket in self.active_connections[patient_id]:
                self.active_connections[patient_id].remove(websocket)
                logger.info(f"WebSocket disconnected for patient {patient_id}")
                if not self.active_connections[patient_id]:
                    del self.active_connections[patient_id]
        elif websocket in self.global_connections:
            self.global_connections.remove(websocket)
            logger.info("Global dashboard WebSocket disconnected")
    
    async def send_to_patient(self, patient_id: str, message: dict):
        """Send message to all connections monitoring a specific patient"""
        if patient_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[patient_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to patient {patient_id}: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn, patient_id)
    
    async def broadcast_to_dashboard(self, message: dict):
        """Broadcast message to all global dashboard connections"""
        disconnected = []
        for connection in self.global_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to dashboard: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_vitals_update(self, patient_id: str, vitals_data: dict):
        """Send vitals update to patient monitors and global dashboard"""
        message = {
            "type": "vitals_update",
            "patient_id": patient_id,
            "data": vitals_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_to_patient(patient_id, message)
        await self.broadcast_to_dashboard(message)
    
    async def send_alert(self, patient_id: str, alert_data: dict):
        """Send alert to patient monitors and global dashboard"""
        message = {
            "type": "alert",
            "patient_id": patient_id,
            "data": alert_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.send_to_patient(patient_id, message)
        await self.broadcast_to_dashboard(message)


# ============================================================================
# Global State (In-Memory Storage)
# ============================================================================

class APIState:
    """Global API state container"""
    def __init__(self):
        self.active_patients: Dict[str, PatientState] = {}
        self.calibrator: Optional[BaselineCalibrator] = None
        self.intervention_tracker: Optional[InterventionTracker] = None
        self.alert_manager: Optional[AlertManager] = None
        self.connection_manager: Optional[ConnectionManager] = None
        self.kafka_servers: str = ""
        self.ml_service_url: str = ""
    
state = APIState()

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="VitalX Backend API", 
    description="REST + WebSocket API Gateway for VitalX Digital Twin System",
    version="2.0.0"
)

# Add CORS middleware for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for HTML dashboards
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Static files mounted from: {static_dir}")


@app.on_event("startup")
async def startup_event():
    """Service startup event"""
    state.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')
    state.ml_service_url = os.getenv('ML_SERVICE_URL', 'http://localhost:8001')
    state.calibrator = BaselineCalibrator(min_samples=10, max_samples=30)
    state.intervention_tracker = InterventionTracker()
    state.alert_manager = AlertManager()
    state.connection_manager = ConnectionManager()
    logger.info("VitalX Backend API Service started successfully")
    logger.info(f"Connected to Kafka servers: {state.kafka_servers}")
    logger.info(f"ML Service URL: {state.ml_service_url}")
    logger.info("BaselineCalibrator initialized for dynamic patient baselines")
    logger.info("WebSocket ConnectionManager initialized")
    logger.info("InterventionTracker initialized for Phase 3 intervention masking")
    logger.info("AlertManager initialized for intelligent alert suppression")
    logger.info("Patient state management initialized")
    logger.info("Service is ready for API requests")


# ============================================================================
# ML Service Helper
# ============================================================================

async def predict_risk_ml_service(
    patient_id: str,
    vitals_sequence: List[List[float]],
    baseline: Optional[Dict]
) -> Dict:
    """
    Call ML service for risk prediction
    Args:
        patient_id: Patient identifier
        vitals_sequence: 20 timesteps of vitals
        baseline: Patient baseline metrics (from BaselineMetrics.to_dict())
    
    Returns:
        Prediction response from ML service or fallback
    """
    try:
        # Transform baseline format from backend to ML service format
        ml_baseline = None
        if baseline and 'vitals' in baseline and 'green_zones' in baseline:
            ml_baseline = {}
            vital_names = ['HR', 'SpO2', 'SBP', 'RR', 'Temp']
            for vital in vital_names:
                if vital in baseline['vitals']:
                    ml_baseline[vital] = {
                        'mean': baseline['vitals'][vital]['mean'],
                        'std': baseline['vitals'][vital]['std'],
                        'green_zone_min': baseline['green_zones'][vital][0],
                        'green_zone_max': baseline['green_zones'][vital][1]
                    }
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{state.ml_service_url}/predict",
                json={
                    "patient_id": patient_id,
                    "vitals_sequence": vitals_sequence,
                    "baseline": ml_baseline,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"ML service returned {response.status_code}: {response.text}")
                return None
    
    except Exception as e:
        logger.error(f"ML service call failed: {e}")
        return None



# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "vitalx-backend-api",
            "active_patients": len(state.active_patients),
            "message": "VitalX Backend API is running"
        }
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200, 
        content={
            "service": "VitalX Backend API",
            "status": "operational",
            "version": "2.0.0",
            "features": [
                "patient_state_management",
                "baseline_calibration",
                "real_time_monitoring",
                "websocket_streaming",
                "intelligent_alerting"
            ]
        }
    )


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/patient/{patient_id}")
async def websocket_patient_endpoint(websocket: WebSocket, patient_id: str):
    """
    WebSocket endpoint for real-time patient monitoring
    
    Streams:
    - Vitals updates
    - Risk scores
    - Alerts
    - Intervention updates
    """
    await state.connection_manager.connect(websocket, patient_id)
    try:
        # Send initial patient state
        if patient_id in state.active_patients:
            patient = state.active_patients[patient_id]
            initial_data = {
                "type": "initial_state",
                "patient_id": patient_id,
                "data": {
                    "status": patient.status.value,
                    "vitals_history_count": len(patient.vitals_history),
                    "has_baseline": patient.baseline is not None,
                },
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_json(initial_data)
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back for heartbeat/acknowledgment
            await websocket.send_json({"type": "ack", "message": "received"})
    
    except WebSocketDisconnect:
        state.connection_manager.disconnect(websocket, patient_id)
    except Exception as e:
        logger.error(f"WebSocket error for patient {patient_id}: {e}")
        state.connection_manager.disconnect(websocket, patient_id)


@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for multi-patient dashboard
    
    Streams updates for all active patients:
    - All vitals updates
    - All alerts
    - Patient admissions/discharges
    """
    await state.connection_manager.connect(websocket)
    try:
        # Send initial dashboard state
        initial_data = {
            "type": "dashboard_state",
            "data": {
                "active_patients": list(state.active_patients.keys()),
                "patient_count": len(state.active_patients),
                "patients": {}
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add summary for each patient
        for patient_id, patient in state.active_patients.items():
            initial_data["data"]["patients"][patient_id] = {
                "status": patient.status.value,
                "vitals_count": len(patient.vitals_history),
                "has_baseline": patient.baseline is not None,
                "last_update": patient.vitals_history[-1]["timestamp"] if patient.vitals_history else None
            }
        
        await websocket.send_json(initial_data)
        
        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"type": "ack", "message": "received"})
    
    except WebSocketDisconnect:
        state.connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket dashboard error: {e}")
        state.connection_manager.disconnect(websocket)



# ============================================================================
# Patient Management Endpoints
# ============================================================================

@app.post("/patients/{patient_id}/admit", status_code=status.HTTP_201_CREATED)
async def admit_patient(patient_id: str, initial_vitals: Optional[List[float]] = None):
    """
    Admit a new patient and initiate baseline calibration (cold-start phase).
    
    During cold-start, the system collects 10-30 seconds of vitals to establish
    patient-specific baseline ranges (the "fingerprint").
    
    Args:
        patient_id: Unique patient identifier
        initial_vitals: Optional first reading [HR, SpO2, SBP, RR, Temp]
    
    Returns:
        Patient state with calibration status
    
    Example:
        POST /patients/PT001/admit
        Body: [75, 98, 120, 16, 37.0]
    """
    # Check if patient already admitted
    if patient_id in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Patient {patient_id} is already admitted. Use /patients/{patient_id}/discharge first."
        )
    
    # Create new patient state
    patient_state = PatientState(
        patient_id=patient_id,
        calibration_status=CalibrationStatus.COLD_START,
        admission_time=datetime.now(),
        last_update=datetime.now(),
        vitals_buffer=[]
    )
    
    # If initial vitals provided, add to buffer
    if initial_vitals:
        if len(initial_vitals) != 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Initial vitals must contain exactly 5 values: [HR, SpO2, SBP, RR, Temp]"
            )
        patient_state.vitals_buffer.append(initial_vitals)
        logger.info(f"Patient {patient_id} admitted with initial vitals")
    else:
        logger.info(f"Patient {patient_id} admitted - awaiting vitals for calibration")
    
    # Store patient state
    state.active_patients[patient_id] = patient_state
    
    # Phase 5: Broadcast patient admission to dashboard
    admission_message = {
        "type": "patient_admitted",
        "patient_id": patient_id,
        "calibration_status": patient_state.calibration_status.value,
        "admission_time": patient_state.admission_time
    }
    await state.connection_manager.broadcast_to_dashboard(admission_message)
    
    return {
        "message": "Patient admitted successfully",
        "patient_id": patient_id,
        "calibration_status": patient_state.calibration_status,
        "admission_time": patient_state.admission_time,
        "vitals_collected": len(patient_state.vitals_buffer),
        "vitals_needed": "10-30 timesteps for baseline calibration"
    }


@app.get("/patients")
async def list_patients():
    """
    List all active patients in the system.
    
    Returns:
        List of patient IDs with their calibration status
    """
    patients_info = []
    for patient_id, patient_state in state.active_patients.items():
        patients_info.append({
            "patient_id": patient_id,
            "calibration_status": patient_state.calibration_status,
            "admission_time": patient_state.admission_time,
            "baseline_calibrated": patient_state.baseline_vitals is not None
        })
    
    return {
        "total_patients": len(patients_info),
        "patients": patients_info
    }


@app.get("/patients/{patient_id}")
async def get_patient_state(patient_id: str):
    """
    Get current state of a patient.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Complete patient state including calibration status and baseline if available
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found. Use /patients/{patient_id}/admit first."
        )
    
    patient_state = state.active_patients[patient_id]
    
    return {
        "patient_id": patient_state.patient_id,
        "calibration_status": patient_state.calibration_status,
        "baseline_calibrated": patient_state.baseline_vitals is not None,
        "admission_time": patient_state.admission_time,
        "last_update": patient_state.last_update,
        "vitals_buffer_size": len(patient_state.vitals_buffer)
    }


@app.get("/patients/{patient_id}/baseline")
async def get_patient_baseline(patient_id: str):
    """
    Get calibrated baseline vital ranges for a patient.
    
    Returns the patient's "Green Zone" - their normal vital sign ranges.
    Only available after cold-start calibration completes (10-30 timesteps).
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Baseline metrics with vital ranges
    
    Example Response:
        {
          "patient_id": "PT001",
          "vitals": {
            "HR": {"mean": 75.0, "std": 5.0, "green_zone": [67.5, 82.5]},
            ...
          },
          "stability_confidence": 0.85,
          "timestamp": "2026-02-14T08:00:00"
        }
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    patient_state = state.active_patients[patient_id]
    
    # Check if baseline is calibrated
    if patient_state.baseline_vitals is None:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=f"Baseline not yet calibrated for patient {patient_id}. "
                   f"Status: {patient_state.calibration_status}. "
                   f"Vitals collected: {len(patient_state.vitals_buffer)}/10-30 needed."
        )
    
    return patient_state.baseline_vitals


@app.post("/patients/{patient_id}/vitals")
async def ingest_vitals(patient_id: str, reading: VitalsReading):
    """
    Ingest a vital signs reading for a patient.
    
    During cold-start phase, this builds the baseline calibration.
    After calibration, this is used for real-time monitoring.
    
    Args:
        patient_id: Patient identifier
        reading: Vital signs reading [HR, SpO2, SBP, RR, Temp]
    
    Returns:
        Calibration progress or acknowledgment
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found. Admit patient first."
        )
    
    patient_state = state.active_patients[patient_id]
    
    # Validate vitals format
    if len(reading.vitals) != 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vitals must contain exactly 5 values: [HR, SpO2, SBP, RR, Temp]"
        )
    
    # Update last update time
    patient_state.last_update = datetime.now()
    
    # If in cold-start, collect for baseline
    if patient_state.calibration_status == CalibrationStatus.COLD_START:
        patient_state.vitals_buffer.append(reading.vitals)
        vitals_count = len(patient_state.vitals_buffer)
        
        # Attempt baseline calibration using BaselineCalibrator
        baseline = state.calibrator.ingest_cold_start(patient_id, reading.vitals)
        
        if baseline is not None:
            # Baseline successfully computed
            patient_state.baseline_vitals = baseline.to_dict()
            patient_state.calibration_status = CalibrationStatus.STABLE
            patient_state.last_baseline_update = datetime.now()
            logger.info(f"Patient {patient_id}: Baseline calibrated with {baseline.sample_count} samples")
            
            return {
                "status": "calibrated",
                "message": f"Baseline calibration complete",
                "vitals_collected": baseline.sample_count,
                "baseline_vitals": patient_state.baseline_vitals,
                "calibration_status": "stable",
                "stability_confidence": baseline.stability_confidence
            }
        else:
            # Still collecting samples
            logger.info(f"Patient {patient_id}: Collected {vitals_count} vitals - baseline pending")
            return {
                "status": "collecting",
                "message": f"Collecting vitals for baseline calibration",
                "vitals_collected": vitals_count,
                "vitals_needed": "10-30",
                "can_compute_baseline": vitals_count >= 10
            }
    
    else:
        # Normal monitoring mode - Phase 2-4: ML + Intervention-Aware Monitoring
        
        # Maintain vitals history for LSTM prediction (need 20 timesteps)
        patient_state.vitals_history.append(reading.vitals)
        if len(patient_state.vitals_history) > 20:
            patient_state.vitals_history.pop(0)
        
        # Phase 4: Get active intervention masks
        current_time = datetime.now()
        active_masks = state.intervention_tracker.get_active_masks(patient_id, current_time)
        has_active_interventions = len(active_masks) > 0
        
        # Get risk prediction from ML service
        risk_score = 0.2  # Default fallback
        ml_response = None
        
        if len(patient_state.vitals_history) >= 20 and patient_state.baseline_vitals:
            # Have enough history and baseline - call ML service
            ml_response = await predict_risk_ml_service(
                patient_id=patient_id,
                vitals_sequence=patient_state.vitals_history,
                baseline=patient_state.baseline_vitals
            )
            
            if ml_response:
                risk_score = ml_response.get("combined_risk", 0.2)
                logger.info(f"Patient {patient_id}: ML risk = {risk_score:.3f} "
                          f"(LSTM: {ml_response.get('lstm_risk', 0):.3f}, "
                          f"Correlation: {ml_response.get('correlation_risk', 0):.3f})")
            else:
                logger.warning(f"Patient {patient_id}: ML service unavailable, using fallback risk")
        elif len(patient_state.vitals_history) < 20:
            logger.debug(f"Patient {patient_id}: Building history ({len(patient_state.vitals_history)}/20 timesteps)")
        
        # Phase 4: Record risk score for temporal smoothing
        state.alert_manager.record_risk_score(patient_id, risk_score)
        smoothed_risk = state.alert_manager.get_smoothed_risk(patient_id)
        
        # Use smoothed risk if available, otherwise use raw risk
        if smoothed_risk is None:
            smoothed_risk = risk_score
        
        # Track risk scores for stability detection (keep last 30)
        patient_state.recent_risk_scores.append(risk_score)
        if len(patient_state.recent_risk_scores) > 30:
            patient_state.recent_risk_scores.pop(0)
        
        # Collect vitals during stable period
        patient_state.stable_vitals_buffer.append(reading.vitals)
        if len(patient_state.stable_vitals_buffer) > 100:  # Keep last 100 vitals
            patient_state.stable_vitals_buffer.pop(0)
        
        # Stability detection: risk < 0.3 for 30+ consecutive minutes
        is_currently_stable = all(r < 0.3 for r in patient_state.recent_risk_scores) and len(patient_state.recent_risk_scores) >= 30
        
        if is_currently_stable:
            if patient_state.stable_period_start is None:
                # Just became stable
                patient_state.stable_period_start = datetime.now()
                logger.info(f"Patient {patient_id}: Stable period started (risk < 0.3)")
            
            # Check if 4 hours passed since last baseline update
            time_since_update = None
            if patient_state.last_baseline_update is not None:
                time_since_update = datetime.now() - patient_state.last_baseline_update
            
            should_update = (
                patient_state.last_baseline_update is None or 
                time_since_update.total_seconds() >= 4 * 3600  # 4 hours
            ) and len(patient_state.stable_vitals_buffer) >= 20
            
            if should_update:
                # Perform rolling baseline update using EMA
                new_baseline = state.calibrator.update_baseline(
                    patient_id,
                    patient_state.stable_vitals_buffer[-20:]  # Use last 20 stable vitals
                )
                
                if new_baseline is not None:
                    patient_state.baseline_vitals = new_baseline.to_dict()
                    patient_state.last_baseline_update = datetime.now()
                    patient_state.stable_vitals_buffer = []  # Reset buffer
                    logger.info(f"Patient {patient_id}: Baseline updated via EMA (4-hour interval)")
                    
                    return {
                        "status": "baseline_updated",
                        "message": "Rolling baseline update performed",
                        "timestamp": reading.timestamp,
                        "baseline_vitals": patient_state.baseline_vitals,
                        "stability_confidence": new_baseline.stability_confidence
                    }
        
        else:
            # Not stable - reset stable period start
            if patient_state.stable_period_start is not None:
                logger.info(f"Patient {patient_id}: Stable period ended (risk >= 0.3)")
            patient_state.stable_period_start = None
            patient_state.stable_vitals_buffer = []  # Clear buffer when unstable
        
        # Phase 4: Check intervention effectiveness and generate alerts
        treatment_failure_alerts = []
        generated_alerts = []
        
        if has_active_interventions and patient_state.baseline_vitals:
            # Check if interventions are working as expected
            vital_names = ["HR", "SpO2", "SBP", "RR", "Temp"]
            current_vitals = {vital_names[i]: reading.vitals[i] for i in range(5)}
            
            # Get baseline means for comparison
            baseline_vitals = {}
            for vital_name in vital_names:
                if vital_name in patient_state.baseline_vitals:
                    baseline_vitals[vital_name] = patient_state.baseline_vitals[vital_name].get("mean", 0)
            
            # Check intervention effectiveness
            failure_alerts = state.intervention_tracker.check_intervention_effectiveness(
                patient_id=patient_id,
                current_vitals=current_vitals,
                baseline_vitals=baseline_vitals,
                current_time=current_time
            )
            
            if failure_alerts:
                treatment_failure_alerts = failure_alerts
                logger.warning(f"Patient {patient_id}: {len(failure_alerts)} treatment failure(s) detected")
        
        # Phase 4: Generate alerts with intelligent suppression
        if smoothed_risk >= 0.5:  # MODERATE risk or higher
            # Determine alert type based on risk level
            if smoothed_risk >= 0.85:
                alert_type = "critical_deterioration"
                alert_message = "CRITICAL: Severe patient deterioration detected"
            elif smoothed_risk >= 0.7:
                alert_type = "high_risk_deterioration"
                alert_message = "HIGH RISK: Significant deterioration detected"
            else:
                alert_type = "moderate_risk"
                alert_message = "MODERATE RISK: Patient showing signs of deterioration"
            
            # Create alert with suppression logic
            alert_details = {
                "risk_score": risk_score,
                "smoothed_risk": smoothed_risk,
                "ml_prediction": ml_response if ml_response else {}
            }
            
            alert = state.alert_manager.create_alert(
                patient_id=patient_id,
                alert_type=alert_type,
                risk_score=smoothed_risk,
                message=alert_message,
                ml_details=alert_details,
                active_intervention_masks=active_masks,
                current_time=current_time
            )
            
            if alert:
                generated_alerts.append(alert)
                if not alert.suppressed:
                    logger.warning(f"Patient {patient_id}: ALERT - {alert_message} (risk={smoothed_risk:.3f})")
                else:
                    logger.info(f"Patient {patient_id}: Alert suppressed - {alert.suppression_reason or 'unknown'}")
        
        # Build response
        response = {
            "status": "monitoring",
            "message": "Vitals received for real-time monitoring",
            "timestamp": reading.timestamp,
            "risk_score": risk_score,
            "smoothed_risk": smoothed_risk,
            "is_stable": is_currently_stable,
            "stable_duration_minutes": (
                (datetime.now() - patient_state.stable_period_start).total_seconds() / 60
                if patient_state.stable_period_start else 0
            ),
            "vitals_history_count": len(patient_state.vitals_history),
            # Phase 4: Intervention-aware monitoring details
            "has_active_interventions": has_active_interventions,
            "active_intervention_count": len(active_masks),
            "masked_vitals": list(active_masks.keys()) if active_masks else []
        }
        
        # Add ML details if available
        if ml_response:
            response["ml_prediction"] = {
                "lstm_risk": ml_response.get("lstm_risk"),
                "correlation_risk": ml_response.get("correlation_risk"),
                "combined_risk": ml_response.get("combined_risk"),
                "risk_level": ml_response.get("risk_level"),
                "detected_patterns": ml_response.get("detected_patterns", []),
                "risk_factors": ml_response.get("risk_factors", [])
            }
        
        # Phase 4: Add alerts to response
        if generated_alerts:
            # Include unsuppressed alerts in response
            unsuppressed_alerts = [a for a in generated_alerts if not a.suppressed]
            if unsuppressed_alerts:
                response["alerts"] = [a.model_dump() for a in unsuppressed_alerts]
                response["alert_count"] = len(unsuppressed_alerts)
        
        # Phase 4: Add treatment failure alerts
        if treatment_failure_alerts:
            response["treatment_failures"] = treatment_failure_alerts
            response["treatment_failure_count"] = len(treatment_failure_alerts)
            # Treatment failures are critical - always show
            if "alerts" not in response:
                response["alerts"] = []
            response["alerts"].extend([{
                "alert_type": "treatment_failure",
                "severity": "high",
                "message": alert["message"],
                "details": alert
            } for alert in treatment_failure_alerts])
        
        # Phase 5: Broadcast vitals update via WebSocket
        await state.connection_manager.send_vitals_update(patient_id, response)
        
        # Phase 5: Broadcast alerts if any
        if response.get("alerts"):
            for alert in response["alerts"]:
                await state.connection_manager.send_alert(patient_id, alert)
        
        return response


@app.delete("/patients/{patient_id}/discharge")
async def discharge_patient(patient_id: str):
    """
    Discharge a patient and remove from active monitoring.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Discharge confirmation
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    patient_state = state.active_patients[patient_id]
    admission_duration = datetime.now() - patient_state.admission_time
    
    # Remove from active patients
    del state.active_patients[patient_id]
    
    # Clear intervention and alert history
    state.intervention_tracker.clear_patient_interventions(patient_id)
    state.alert_manager.clear_patient_alerts(patient_id)
    
    logger.info(f"Patient {patient_id} discharged after {admission_duration}")
    
    # Phase 5: Broadcast patient discharge to dashboard
    discharge_message = {
        "type": "patient_discharged",
        "patient_id": patient_id,
        "discharge_time": datetime.now().isoformat()
    }
    await state.connection_manager.broadcast_to_dashboard(discharge_message)
    
    return {
        "message": "Patient discharged successfully",
        "patient_id": patient_id,
        "admission_time": patient_state.admission_time,
        "discharge_time": datetime.now(),
        "duration": str(admission_duration),
        "baseline_was_calibrated": patient_state.baseline_vitals is not None
    }


@app.post("/patients/{patient_id}/recalibrate")
async def recalibrate_baseline(patient_id: str):
    """
    Initiate baseline recalibration for a patient.
    
    Used when patient condition stabilizes at a new normal level
    (e.g., after surgery, medication changes, etc.)
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Recalibration status
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    patient_state = state.active_patients[patient_id]
    
    # Clear existing baseline and restart cold-start
    patient_state.calibration_status = CalibrationStatus.RECALIBRATING
    patient_state.vitals_buffer = []
    patient_state.last_update = datetime.now()
    
    logger.info(f"Patient {patient_id}: Baseline recalibration initiated")
    
    return {
        "message": "Baseline recalibration initiated",
        "patient_id": patient_id,
        "new_status": patient_state.calibration_status,
        "previous_baseline_cleared": True
    }


# ============================================================================
# Phase 3: Intervention-Aware Masking Endpoints
# ============================================================================

@app.post("/patients/{patient_id}/interventions")
async def log_intervention(patient_id: str, intervention: InterventionRequest):
    """
    Log a clinical intervention for a patient.
    
    This activates intervention masks to prevent false alarms from expected
    physiological responses.
    
    Args:
        patient_id: Patient identifier
        intervention: Intervention details
    
    Returns:
        Intervention record with active masks
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found. Admit patient first."
        )
    
    # Validate intervention type
    try:
        intervention_type = InterventionType(intervention.type.lower())
    except ValueError:
        valid_types = [t.value for t in InterventionType]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid intervention type. Valid types: {valid_types}"
        )
    
    # Log intervention
    intervention_record = state.intervention_tracker.log_intervention(
        patient_id=patient_id,
        intervention_type=intervention_type,
        timestamp=intervention.timestamp,
        dosage=intervention.dosage,
        administered_by=intervention.administered_by,
        notes=intervention.notes
    )
    
    logger.info(f"Intervention logged for {patient_id}: {intervention_type.value}")
    
    return {
        "message": "Intervention logged successfully",
        "intervention_id": intervention_record.intervention_id,
        "patient_id": patient_id,
        "type": intervention_record.intervention_type.value,
        "timestamp": intervention_record.timestamp.isoformat(),
        "active_masks": [
            {
                "vital": mask.vital_name,
                "expected_direction": mask.expected_direction,
                "duration_minutes": mask.mask_duration_minutes,
                "threshold_change": mask.threshold_change
            }
            for mask in intervention_record.expected_effects
        ],
        "response_window_minutes": state.intervention_tracker.INTERVENTION_PROFILES[intervention_type]["response_window_minutes"],
        "expected_response": state.intervention_tracker.INTERVENTION_PROFILES[intervention_type]["expected_response"]
    }


@app.get("/patients/{patient_id}/interventions")
async def get_interventions(patient_id: str, limit: Optional[int] = 10):
    """
    Get intervention history for a patient.
    
    Args:
        patient_id: Patient identifier
        limit: Maximum number of interventions to return
    
    Returns:
        List of interventions with effectiveness data
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    interventions = state.intervention_tracker.get_intervention_history(
        patient_id=patient_id,
        limit=limit
    )
    
    return {
        "patient_id": patient_id,
        "intervention_count": len(interventions),
        "interventions": interventions
    }


@app.get("/patients/{patient_id}/interventions/active")
async def get_active_masks(patient_id: str):
    """
    Get currently active intervention masks for a patient.
    
    Shows which vitals are currently masked due to recent interventions.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Active intervention masks
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    active_masks = state.intervention_tracker.get_active_masks(patient_id)
    
    return {
        "patient_id": patient_id,
        "has_active_masks": len(active_masks) > 0,
        "masked_vitals": list(active_masks.keys()),
        "active_masks": {
            vital: [
                {
                    "expected_direction": mask.expected_direction,
                    "duration_minutes": mask.mask_duration_minutes,
                    "threshold_change": mask.threshold_change
                }
                for mask in masks
            ]
            for vital, masks in active_masks.items()
        }
    }


@app.get("/patients/{patient_id}/alerts")
async def get_patient_alerts(
    patient_id: str,
    include_suppressed: bool = False,
    limit: int = 10
):
    """
    Get recent alerts for a patient.
    
    Args:
        patient_id: Patient identifier
        include_suppressed: Include suppressed alerts
        limit: Maximum number of alerts to return
    
    Returns:
        List of recent alerts
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    alerts = state.alert_manager.get_recent_alerts(
        patient_id=patient_id,
        include_suppressed=include_suppressed,
        limit=limit
    )
    
    stats = state.alert_manager.get_alert_statistics(patient_id)
    
    return {
        "patient_id": patient_id,
        "alert_count": len(alerts),
        "alerts": alerts,
        "statistics": stats
    }


@app.post("/patients/{patient_id}/alerts/{alert_id}/outcome")
async def record_alert_outcome(
    patient_id: str,
    alert_id: str,
    was_true_positive: bool
):
    """
    Record the outcome of an alert (true or false positive).
    
    Used to track alert accuracy and improve suppression logic.
    
    Args:
        patient_id: Patient identifier
        alert_id: Alert identifier
        was_true_positive: Whether alert correctly predicted deterioration
    
    Returns:
        Confirmation
    """
    if patient_id not in state.active_patients:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient {patient_id} not found"
        )
    
    # Use Phase 6 acknowledgment with backward compatibility
    outcome = "true_positive" if was_true_positive else "false_positive"
    alert = state.alert_manager.acknowledge_alert(
        alert_id=alert_id,
        clinician_id="system",  # Legacy endpoint
        outcome=outcome
    )
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )
    
    return {
        "message": "Alert outcome recorded",
        "alert_id": alert_id,
        "outcome": outcome
    }


# ============================================================================
# Phase 6: Alert Acknowledgment & Analytics Endpoints
# ============================================================================

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    clinician_id: str,
    outcome: str,
    outcome_notes: Optional[str] = None
):
    """
    Acknowledge an alert and record clinical outcome.
    
    Phase 6: Enables clinicians to provide feedback on alert accuracy.
    
    Args:
        alert_id: Alert identifier
        clinician_id: ID of acknowledging clinician
        outcome: One of ["true_positive", "false_positive", "intervention_needed", "no_action"]
        outcome_notes: Optional notes
    
    Returns:
        Updated alert
    """
    valid_outcomes = ["true_positive", "false_positive", "intervention_needed", "no_action"]
    
    if outcome not in valid_outcomes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid outcome. Must be one of: {valid_outcomes}"
        )
    
    alert = state.alert_manager.acknowledge_alert(
        alert_id=alert_id,
        clinician_id=clinician_id,
        outcome=outcome,
        outcome_notes=outcome_notes
    )
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} not found"
        )
    
    # Broadcast acknowledgment via WebSocket
    ack_message = {
        "type": "alert_acknowledged",
        "alert_id": alert_id,
        "acknowledged_by": clinician_id,
        "outcome": outcome,
        "timestamp": datetime.now().isoformat()
    }
    await state.connection_manager.broadcast_to_dashboard(ack_message)
    
    return {
        "message": "Alert acknowledged successfully",
        "alert": alert.model_dump()
    }


@app.get("/analytics/alerts")
async def get_alert_analytics(
    patient_id: Optional[str] = None,
    days: int = 7
):
    """
    Get alert analytics and accuracy metrics.
    
    Phase 6: Provides insights into system performance.
    
    Args:
        patient_id: Specific patient or None for all patients
        days: Number of days to analyze (default 7)
    
    Returns:
        Analytics dictionary with metrics
    """
    start_time = datetime.now() - timedelta(days=days)
    end_time = datetime.now()
    
    analytics = state.alert_manager.get_analytics(
        patient_id=patient_id,
        start_time=start_time,
        end_time=end_time
    )
    
    return analytics


@app.get("/analytics/patients/{patient_id}/statistics")
async def get_patient_statistics(patient_id: str):
    """
    Get detailed statistics for a specific patient.
    
    Phase 6: Patient-specific performance metrics.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        Patient statistics
    """
    stats = state.alert_manager.get_patient_statistics(patient_id)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No statistics found for patient {patient_id}"
        )
    
    return stats


@app.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """
    Get comprehensive analytics for dashboard display.
    
    Phase 6: Overview of system performance across all patients.
    
    Returns:
        Dashboard analytics
    """
    # Get overall analytics
    overall = state.alert_manager.get_analytics(days=7)
    
    # Get per-patient statistics
    patient_stats = []
    for patient_id in state.active_patients.keys():
        stats = state.alert_manager.get_patient_statistics(patient_id)
        if stats:
            patient_stats.append(stats)
    
    # Calculate system-wide metrics
    total_patients = len(state.active_patients)
    
    return {
        "overall_metrics": overall,
        "patient_statistics": patient_stats,
        "system_status": {
            "active_patients": total_patients,
            "timestamp": datetime.now().isoformat()
        }
    }