"""
Backend API - Orchestrator
WebSocket streaming + REST endpoints + RAG chat integration
"""

import asyncio
import logging
import math
import random
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Set
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .stream_merger import StreamMerger
from .rag_chat_agent import get_chat_agent
from .pdf_report_generator import ICUReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend-api")

# Global stream merger instance
stream_merger: StreamMerger = None

# Active WebSocket connections
websocket_connections: Set[WebSocket] = set()

# RAG Service API URL (GROQ-powered)
PATHWAY_API_URL = "http://rag-service:8002"

# =========================================================
# MOCK DATA (fallback when Kafka pipeline is not running)
# =========================================================

_MOCK_BASELINES = [
    {"patient_id": "P1", "name": "James Wilson",   "floor_id": "ICU-1", "bed": "1A", "age": 58, "hr": 78,  "sbp": 124, "spo2": 97, "rr": 15, "temp": 37.1, "state": "STABLE",              "risk": 0.12},
    {"patient_id": "P2", "name": "Maria Garcia",   "floor_id": "ICU-1", "bed": "1B", "age": 72, "hr": 88,  "sbp": 108, "spo2": 94, "rr": 19, "temp": 37.6, "state": "EARLY_DETERIORATION", "risk": 0.52},
    {"patient_id": "P3", "name": "Robert Chen",    "floor_id": "ICU-2", "bed": "2A", "age": 45, "hr": 105, "sbp": 92,  "spo2": 91, "rr": 24, "temp": 38.4, "state": "LATE_DETERIORATION",  "risk": 0.73},
    {"patient_id": "P4", "name": "Sarah Johnson",  "floor_id": "ICU-2", "bed": "2B", "age": 63, "hr": 72,  "sbp": 130, "spo2": 98, "rr": 14, "temp": 36.8, "state": "STABLE",              "risk": 0.11},
    {"patient_id": "P5", "name": "Michael Brown",  "floor_id": "ICU-2", "bed": "2C", "age": 51, "hr": 118, "sbp": 88,  "spo2": 89, "rr": 28, "temp": 38.9, "state": "CRITICAL",            "risk": 0.88},
    {"patient_id": "P6", "name": "Emily Davis",    "floor_id": "ICU-3", "bed": "3A", "age": 66, "hr": 82,  "sbp": 118, "spo2": 96, "rr": 17, "temp": 37.3, "state": "RECOVERING",          "risk": 0.28},
    {"patient_id": "P7", "name": "David Martinez", "floor_id": "ICU-1", "bed": "1C", "age": 79, "hr": 95,  "sbp": 100, "spo2": 93, "rr": 22, "temp": 38.1, "state": "EARLY_DETERIORATION", "risk": 0.55},
    {"patient_id": "P8", "name": "Lisa Thompson",  "floor_id": "ICU-3", "bed": "3B", "age": 44, "hr": 68,  "sbp": 135, "spo2": 99, "rr": 13, "temp": 36.7, "state": "STABLE",              "risk": 0.09},
]

def _build_mock_vitals(b: dict, tick: int) -> dict:
    phase = tick * 0.05
    hr  = b["hr"]  + 3 * math.sin(phase) + random.gauss(0, 1.5)
    sbp = b["sbp"] + 4 * math.cos(phase) + random.gauss(0, 2.0)
    dbp = sbp * 0.62 + random.gauss(0, 1.5)
    spo2 = min(100.0, b["spo2"] + 0.5 * math.sin(phase + 1) + random.gauss(0, 0.4))
    rr   = b["rr"]   + 1.5 * math.sin(phase + 2) + random.gauss(0, 0.8)
    temp = b["temp"] + 0.1 * math.sin(phase * 0.3) + random.gauss(0, 0.05)
    risk = max(0.0, min(1.0, b["risk"] + random.uniform(-0.03, 0.03)))
    shock_index = hr / max(sbp, 1.0)
    return {
        "patient_id":   b["patient_id"],
        "name":         b["name"],
        "bed_number":   b["bed"],
        "age":          b["age"],
        "floor_id":     b["floor_id"],
        "state":        b["state"],
        "timestamp":    datetime.now().isoformat(),
        "heart_rate":   round(hr, 1),
        "systolic_bp":  round(sbp, 1),
        "diastolic_bp": round(dbp, 1),
        "spo2":         round(spo2, 1),
        "respiratory_rate": round(rr, 1),
        "temperature":  round(temp, 2),
        "shock_index":  round(shock_index, 3),
        "anomaly_flag": risk >= 0.7,
        "hr_anomaly":   hr > 110 or hr < 50,
        "sbp_anomaly":  sbp < 90 or sbp > 160,
        "spo2_anomaly": spo2 < 92,
        "shock_index_anomaly": shock_index > 1.0,
        "lactate_anomaly": False,
    }, risk

def _seed_mock_patients(merger: StreamMerger):
    """Populate stream_merger with baseline mock patients."""
    for b in _MOCK_BASELINES:
        vitals, risk = _build_mock_vitals(b, 0)
        merger.patient_state[b["patient_id"]]["vitals"] = vitals
        merger.patient_state[b["patient_id"]]["prediction"] = {
            "patient_id": b["patient_id"],
            "risk_score": round(risk, 4),
            "risk_level": "HIGH" if risk >= 0.7 else ("MEDIUM" if risk >= 0.3 else "LOW"),
            "timestamp": datetime.now().isoformat(),
        }
        merger.patient_state[b["patient_id"]]["last_vitals_update"] = datetime.now()
        merger.patient_state[b["patient_id"]]["last_prediction_update"] = datetime.now()

async def _run_mock_updater(merger: StreamMerger):
    """Background task: refresh mock vitals every 2 s (stops when real Kafka data arrives)."""
    tick = 0
    while True:
        await asyncio.sleep(2)
        tick += 1
        for b in _MOCK_BASELINES:
            # If real Kafka data has overwritten this patient, stop mocking them
            state = merger.patient_state.get(b["patient_id"], {})
            vitals = state.get("vitals", {})
            if vitals.get("_kafka_live"):
                continue
            new_vitals, risk = _build_mock_vitals(b, tick)
            merger.patient_state[b["patient_id"]]["vitals"] = new_vitals
            merger.patient_state[b["patient_id"]]["prediction"] = {
                "patient_id":  b["patient_id"],
                "risk_score":  round(risk, 4),
                "risk_level":  "HIGH" if risk >= 0.7 else ("MEDIUM" if risk >= 0.3 else "LOW"),
                "timestamp":   datetime.now().isoformat(),
            }
            merger.patient_state[b["patient_id"]]["last_vitals_update"] = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    
    global stream_merger
    
    # Startup
    logger.info("Starting Backend API...")
    
    stream_merger = StreamMerger(kafka_servers='kafka:9092')
    stream_merger.register_listener(on_patient_update)
    stream_merger.start()
    
    logger.info("Stream merger started")
    
    # Seed mock patients so the dashboard is never empty (overridden by live Kafka data)
    _seed_mock_patients(stream_merger)
    logger.info("Mock patient data seeded as baseline")

    # Start background mock updater
    _mock_updater = asyncio.create_task(_run_mock_updater(stream_merger))
    
    logger.info("Backend API ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Backend API...")
    _mock_updater.cancel()
    stream_merger.stop()
    logger.info("Backend API stopped")

# Create FastAPI app
app = FastAPI(
    title="VitalX Backend API",
    description="Orchestrator for streaming ICU data",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models

class ChatRequest(BaseModel):
    """Chat request schema"""
    patient_id: str = Field(..., description="Patient identifier")
    question: str = Field(..., description="User question")
    
    class Config:
        schema_extra = {
            "example": {
                "patient_id": "P001",
                "question": "Why is the patient's shock index increasing?"
            }
        }

class ChatResponse(BaseModel):
    """Chat response schema"""
    patient_id: str
    question: str
    answer: str
    sources: List[dict]

# Callback for patient updates

def on_patient_update(patient_id: str, unified_view: dict):
    """
    Callback for patient state updates
    Broadcasts to all connected WebSocket clients
    """
    
    # Broadcast to all WebSocket connections
    for ws in list(websocket_connections):
        try:
            asyncio.create_task(ws.send_json(unified_view))
        except:
            websocket_connections.discard(ws)

# WebSocket endpoint

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming
    Pushes unified patient views to connected clients
    """
    
    await websocket.accept()
    websocket_connections.add(websocket)
    
    logger.info(f"WebSocket client connected (total: {len(websocket_connections)})")
    
    try:
        # Send initial state for all patients
        for patient_id in stream_merger.get_all_patients():
            unified_view = stream_merger.get_unified_view(patient_id)
            if unified_view:
                await websocket.send_json(unified_view)
        
        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        websocket_connections.discard(websocket)

# REST endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    if stream_merger is None:
        return {
            "status": "unhealthy",
            "message": "Stream merger not initialized"
        }
    
    merger_status = stream_merger.get_status()
    
    return {
        "status": "healthy" if merger_status['is_running'] else "degraded",
        "stream_merger": merger_status,
        "active_websockets": len(websocket_connections)
    }

@app.get("/patients")
async def get_patients():
    """Get list of all active patients"""
    
    if stream_merger is None:
        raise HTTPException(status_code=503, detail="Stream merger not initialized")
    
    patient_ids = stream_merger.get_all_patients()
    
    # Get brief summary for each patient
    patients = []
    for patient_id in patient_ids:
        unified_view = stream_merger.get_unified_view(patient_id)
        if unified_view:
            patients.append({
                'patient_id': patient_id,
                'risk_score': unified_view.get('risk_score', 0.0),
                'last_updated': unified_view.get('timestamp'),
                'anomaly_flag': unified_view.get('features', {}).get('anomaly_flag', False)
            })
    
    return {
        "total_patients": len(patients),
        "patients": patients
    }

@app.get("/patients/{patient_id}")
async def get_patient_details(patient_id: str):
    """Get detailed unified view for specific patient"""
    
    if stream_merger is None:
        raise HTTPException(status_code=503, detail="Stream merger not initialized")
    
    unified_view = stream_merger.get_unified_view(patient_id)
    
    if not unified_view:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    return unified_view

@app.get("/patients/{patient_id}/history")
async def get_patient_history(patient_id: str, hours: int = 4):
    """
    Get patient history for last N hours
    
    Args:
        patient_id: Patient identifier
        hours: Number of hours to retrieve (default: 4)
    """
    
    if stream_merger is None:
        raise HTTPException(status_code=503, detail="Stream merger not initialized")
    
    history = stream_merger.get_patient_history(patient_id, hours=hours)
    
    return {
        "patient_id": patient_id,
        "hours": hours,
        "data_points": len(history),
        "history": history
    }

@app.get("/patients/{patient_id}/latest")
async def get_patient_latest(patient_id: str):
    """Get latest data for patient (alias for /patients/{id})"""
    return await get_patient_details(patient_id)

# Floors API

@app.get("/floors")
async def get_floors():
    """Get list of all active ICU floors/units"""
    
    if stream_merger is None:
        raise HTTPException(status_code=503, detail="Stream merger not initialized")
    
    # Get patient counts per floor
    patient_ids = stream_merger.get_all_patients()
    floor_patient_counts = {}
    
    for patient_id in patient_ids:
        unified_view = stream_merger.get_unified_view(patient_id)
        if unified_view:
            floor_id = unified_view.get('floor_id', 'ICU-1')
            floor_patient_counts[floor_id] = floor_patient_counts.get(floor_id, 0) + 1
    
    # Generate floor data
    floors = []
    for floor_id, count in floor_patient_counts.items():
        floors.append({
            'id': floor_id,
            'name': f'ICU Floor {floor_id.split("-")[-1]}',
            'capacity': 10,
            'current_patients': count,
            'available_beds': max(0, 10 - count)
        })
    
    # Ensure at least one floor exists
    if not floors:
        floors.append({
            'id': 'ICU-1',
            'name': 'ICU Floor 1',
            'capacity': 10,
            'current_patients': 0,
            'available_beds': 10
        })
    
    return {"floors": floors}

@app.get("/floors/{floor_id}/patients")
async def get_floor_patients(floor_id: str):
    """Get all patients on a specific floor"""
    
    if stream_merger is None:
        raise HTTPException(status_code=503, detail="Stream merger not initialized")
    
    patient_ids = stream_merger.get_all_patients()
    
    floor_patients = []
    for patient_id in patient_ids:
        unified_view = stream_merger.get_unified_view(patient_id)
        if unified_view and unified_view.get('floor_id', 'ICU-1') == floor_id:
            floor_patients.append({
                'patient_id': patient_id,
                'floor_id': floor_id,
                'risk_score': unified_view.get('risk_score', 0.0),
                'last_updated': unified_view.get('timestamp'),
                'anomaly_flag': unified_view.get('features', {}).get('anomaly_flag', False),
                'vitals': {
                    'heart_rate': unified_view.get('vitals', {}).get('heart_rate'),
                    'systolic_bp': unified_view.get('vitals', {}).get('systolic_bp'),
                    'diastolic_bp': unified_view.get('vitals', {}).get('diastolic_bp'),
                    'spo2': unified_view.get('vitals', {}).get('spo2')
                }
            })
    
    return {"patients": floor_patients}

# Stats API

@app.get("/stats/overview")
async def get_stats_overview():
    """Get system-wide statistics overview"""
    
    if stream_merger is None:
        raise HTTPException(status_code=503, detail="Stream merger not initialized")
    
    patient_ids = stream_merger.get_all_patients()
    
    total_patients = len(patient_ids)
    high_risk_count = 0
    anomaly_count = 0
    stable_patients = 0
    unstable_patients = 0
    floors_active = set()
    
    # Analyze each patient
    for patient_id in patient_ids:
        unified_view = stream_merger.get_unified_view(patient_id)
        if unified_view:
            risk_score = unified_view.get('risk_score', 0.0)
            anomaly_flag = unified_view.get('features', {}).get('anomaly_flag', False)
            floor_id = unified_view.get('floor_id', 'ICU-1')
            
            floors_active.add(floor_id)
            
            if risk_score > 0.5:
                high_risk_count += 1
                unstable_patients += 1
            else:
                stable_patients += 1
            
            if anomaly_flag:
                anomaly_count += 1
    
    return {
        "total_patients": total_patients,
        "high_risk_count": high_risk_count,
        "anomaly_count": anomaly_count,
        "stable_patients": stable_patients,
        "unstable_patients": unstable_patients,
        "floors_active": len(floors_active),
        "data_source": "kafka_stream"
    }

# RAG Chat endpoint

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    RAG-powered chat endpoint using GROQ-powered rag-service
    
    Flow:
    1. Query RAG service with GROQ LLM
    2. Return response with sources
    """
    
    if stream_merger is None:
        raise HTTPException(status_code=503, detail="Stream merger not initialized")
    
    # Query RAG service (GROQ-powered)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            rag_response = await client.post(
                f"{PATHWAY_API_URL}/api/handoff/query",
                json={
                    "patient_id": request.patient_id,
                    "question": request.question
                }
            )
            
        if rag_response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"RAG service query failed: {rag_response.status_code}"
            )
        
        rag_data = rag_response.json()
        answer = rag_data.get('answer', 'No response from RAG service')
        context_used = rag_data.get('context_used', [])
        
    except httpx.RequestError as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to RAG service: {str(e)}"
        )
    
    return ChatResponse(
        patient_id=request.patient_id,
        question=request.question,
        answer=answer,
        sources=context_used[:3]  # Return top 3 sources
    )

# ================== PDF Report Generation ==================

class ReportRequest(BaseModel):
    """Request model for PDF report generation"""
    time_range_hours: int = Field(default=3, ge=1, le=48, description="Hours of data to include (1-48, default 3hrs)")
    include_ai_summary: bool = Field(default=True, description="Include AI-generated clinical summary")

@app.post("/patients/{patient_id}/reports/generate")
async def generate_patient_report(
    patient_id: str,
    request: ReportRequest = ReportRequest()
) -> StreamingResponse:
    """
    Generate comprehensive PDF report for a patient
    
    Args:
        patient_id: Patient identifier
        request: Report configuration
        
    Returns:
        PDF file download
    """
    
    try:
        # Get current patient data
        patient_data = stream_merger.get_unified_view(patient_id)
        if not patient_data:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
        
        # Get historical data
        vitals_history = stream_merger.get_patient_history(
            patient_id,
            hours=request.time_range_hours
        )
        
        logger.info(f"PDF Report: Patient {patient_id}, History entries: {len(vitals_history)}")
        
        # Get risk history (using same vitals history with predictions)
        risk_history = []
        for entry in vitals_history:
            if 'prediction' in entry or 'risk_score' in entry:
                risk_entry = {
                    'timestamp': entry.get('timestamp'),
                    'risk_score': entry.get('prediction', {}).get('risk_score', 0) if 'prediction' in entry else entry.get('risk_score', 0)
                }
                risk_history.append(risk_entry)
        
        # Generate AI summary if requested
        ai_summary = None
        if request.include_ai_summary:
            try:
                # Query RAG for patient context
                async with httpx.AsyncClient() as client:
                    payload = {
                        "patient_id": patient_id,
                        "query_text": f"Provide a comprehensive clinical summary for patient {patient_id} including current status, trends, and any concerns.",
                        "top_k": 10
                    }
                    logger.info(f"Querying RAG for {patient_id}: {payload}")
                    
                    rag_response = await client.post(
                        f"{PATHWAY_API_URL}/query",
                        json=payload,
                        timeout=10.0
                    )
                    
                    logger.info(f"RAG response status: {rag_response.status_code}")
                    
                    if rag_response.status_code == 200:
                        rag_data = rag_response.json()
                        retrieved_context = rag_data.get('retrieved_context', [])
                        logger.info(f"Retrieved {len(retrieved_context)} context items for {patient_id}")
                        
                        # Generate AI summary using LangChain
                        chat_agent = get_chat_agent()
                        ai_summary = chat_agent.generate_response(
                            patient_id=patient_id,
                            question="Provide a comprehensive clinical summary including current status, vital signs trends, risk assessment, and clinical recommendations.",
                            retrieved_context=retrieved_context
                        )
                        logger.info(f"Generated AI summary for {patient_id}: {len(ai_summary)} characters")
                    else:
                        logger.error(f"RAG query failed with status {rag_response.status_code}: {rag_response.text}")
                        ai_summary = f"AI summary unavailable (RAG returned {rag_response.status_code})"
            except Exception as e:
                logger.error(f"Could not generate AI summary: {e}", exc_info=True)
                ai_summary = "AI summary unavailable. The RAG system is still indexing patient data."
        
        # Generate PDF report
        report_generator = ICUReportGenerator()
        pdf_buffer = report_generator.generate_report(
            patient_id=patient_id,
            patient_data=patient_data,
            vitals_history=vitals_history,
            risk_history=risk_history,
            ai_summary=ai_summary,
            time_range_hours=request.time_range_hours
        )
        
        # Return PDF as streaming response
        filename = f"ICU_Report_{patient_id}_{asyncio.get_event_loop().time():.0f}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VitalX Backend API",
        "version": "2.1.0",
        "status": "operational",
        "architecture": "streaming-first",
        "features": ["Real-time monitoring", "AI Chat", "PDF Reports"],
        "endpoints": {
            "health": "/health",
            "patients": "/patients",
            "websocket": "/ws",
            "chat": "/chat",
            "pdf_report": "/patients/{patient_id}/reports/generate"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
