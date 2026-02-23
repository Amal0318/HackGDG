"""
Backend API - Orchestrator
WebSocket streaming + REST endpoints + RAG chat integration
"""

import asyncio
import logging
from contextlib import asynccontextmanager
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

# Pathway Query API URL
PATHWAY_API_URL = "http://pathway-engine:8080"

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
    logger.info("Backend API ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Backend API...")
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
    RAG-powered chat endpoint
    
    Flow:
    1. Query Pathway RAG for relevant context
    2. Pass context + question to LLM
    3. Return grounded response with sources
    """
    
    if stream_merger is None:
        raise HTTPException(status_code=503, detail="Stream merger not initialized")
    
    # Step 1: Query Pathway RAG
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            rag_response = await client.post(
                f"{PATHWAY_API_URL}/query",
                json={
                    "patient_id": request.patient_id,
                    "query_text": request.question,
                    "top_k": 5
                }
            )
            
        if rag_response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Pathway RAG query failed: {rag_response.status_code}"
            )
        
        rag_data = rag_response.json()
        retrieved_context = rag_data.get('retrieved_context', [])
        
    except httpx.RequestError as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to Pathway RAG: {str(e)}"
        )
    
    # Step 2: Generate LLM response using LangChain
    chat_agent = get_chat_agent()
    
    if not retrieved_context:
        answer = f"No recent data available for patient {request.patient_id} to answer this question."
    else:
        # Generate intelligent response using LangChain
        answer = chat_agent.generate_response(
            patient_id=request.patient_id,
            question=request.question,
            retrieved_context=retrieved_context
        )
    
    return ChatResponse(
        patient_id=request.patient_id,
        question=request.question,
        answer=answer,
        sources=retrieved_context[:3]  # Return top 3 sources
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
                    rag_response = await client.post(
                        f"{PATHWAY_API_URL}/query",
                        json={
                            "patient_id": patient_id,
                            "query": f"Provide a comprehensive clinical summary for patient {patient_id} including current status, trends, and any concerns.",
                            "top_k": 10
                        },
                        timeout=5.0
                    )
                    
                    if rag_response.status_code == 200:
                        rag_data = rag_response.json()
                        retrieved_context = rag_data.get('results', [])
                        
                        # Generate AI summary using LangChain
                        chat_agent = get_chat_agent()
                        ai_summary = chat_agent.generate_response(
                            patient_id=patient_id,
                            question="Provide a comprehensive clinical summary including current status, vital trends, and recommendations.",
                            retrieved_context=retrieved_context
                        )
            except Exception as e:
                logger.warning(f"Could not generate AI summary: {e}")
                ai_summary = "AI summary unavailable due to system limitations."
        
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
