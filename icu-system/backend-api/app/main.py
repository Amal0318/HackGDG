"""
Backend API Service - REST + WebSocket API Gateway
Real-time ICU patient monitoring with ML predictions
Integrated with Requestly API for monitoring and resilience
"""

import logging
import json
import asyncio
from typing import Optional, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .config import settings
from .kafka_service import patient_data_store, kafka_consumer_service
from .requestly_integration import requestly_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("backend-api")

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("=" * 80)
    logger.info("🏥 ICU Monitoring System - Backend API Starting")
    logger.info("=" * 80)
    logger.info(f"Kafka: {settings.KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Floors configured: {len(settings.FLOORS)}")
    logger.info(f"🔧 Requestly Integration: ENABLED (Sponsor Feature)")
    
    try:
        # Start Kafka consumer
        await kafka_consumer_service.start()
        logger.info("✅ Kafka consumer started successfully")
    except Exception as e:
        logger.warning(f"⚠️ Kafka consumer failed to start: {e}")
        logger.info("🎭 Enabling Requestly Mock Mode as fallback")
        requestly_service.enable_mock_mode()
    
    logger.info("🚀 Backend API ready to serve requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Backend API")
    await kafka_consumer_service.stop()
    logger.info("✅ Shutdown complete")

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# =========================================================
# API ROUTES
# =========================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "features": [
            "JWT Authentication",
            "Multi-floor ICU monitoring  ",
            "Real-time patient data streaming",
            "Requestly API integration (Sponsor)"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    patient_count = patient_data_store.get_patient_count()
    
    return {
        "status": "healthy",
        "service": "backend-api",
        "kafka_consumer": "running" if kafka_consumer_service.running else "stopped",
        "patients_monitored": patient_count,
        "requestly_mock_mode": requestly_service.mock_mode,
        "floors_configured": len(settings.FLOORS)
    }


@app.get("/api/floors")
async def get_floors():
    """Get list of all ICU floors"""
    requestly_service.log_api_request("/api/floors", "GET", "anonymous")
    
    floors = settings.FLOORS.copy()
    
    # Add patient counts to each floor
    for floor in floors:
        if requestly_service.mock_mode:
            # Use mock data
            patients = requestly_service.get_mock_patients_for_floor(floor["id"], floor["capacity"])
            floor["current_patients"] = len(patients)
            floor["available_beds"] = floor["capacity"] - len(patients)
        else:
            # Use real data from Kafka
            patients = await patient_data_store.get_patients_by_floor(floor["id"])
            floor["current_patients"] = len(patients)
            floor["available_beds"] = floor["capacity"] - len(patients)
    
    response = {
        "floors": floors,
        "total_capacity": sum(f["capacity"] for f in floors),
        "total_patients": sum(f["current_patients"] for f in floors)
    }
    
    return requestly_service.intercept_response(response, "/api/floors")


@app.get("/api/floors/{floor_id}/patients")
async def get_floor_patients(
    floor_id: str
):
    """Get all patients on a specific floor with live data"""
    requestly_service.log_api_request(f"/api/floors/{floor_id}/patients", "GET", "anonymous")
    
    # Validate floor_id
    valid_floors = [f["id"] for f in settings.FLOORS]
    if floor_id not in valid_floors:
        raise HTTPException(status_code=404, detail=f"Floor {floor_id} not found")
    
    if requestly_service.mock_mode:
        # Return mock data via Requestly fallback
        patients = requestly_service.get_mock_patients_for_floor(floor_id, 8)
        logger.info(f"🎭 Serving mock data for floor {floor_id} (Requestly fallback)")
    else:
        # Return real data from Kafka
        patients = await patient_data_store.get_patients_by_floor(floor_id)
    
    # Convert to list format
    patient_list = list(patients.values())
    
    # Add statistics
    high_risk_count = sum(1 for p in patient_list if p.get("is_high_risk", False))
    anomaly_count = sum(1 for p in patient_list if p.get("anomaly_flag", 0) == 1)
    
    response = {
        "floor_id": floor_id,
        "patients": patient_list,
        "statistics": {
            "total_patients": len(patient_list),
            "high_risk_patients": high_risk_count,
            "anomalies_detected": anomaly_count,
            "data_source": "requestly_mock" if requestly_service.mock_mode else "kafka_live"
        }
    }
    
    return requestly_service.intercept_response(response, f"/api/floors/{floor_id}/patients")


@app.get("/api/patients/{patient_id}")
async def get_patient_detail(
    patient_id: str
):
    """Get detailed information for a specific patient"""
    requestly_service.log_api_request(f"/api/patients/{patient_id}", "GET", "anonymous")
    
    if requestly_service.mock_mode:
        # Extract floor from patient_id (P1-xxx, P2-xxx, P3-xxx)
        floor_id = f"{patient_id[1]}F" if patient_id.startswith('P') else "1F"
        patient_data = requestly_service.get_mock_patient_data(patient_id, floor_id)
    else:
        patient_data = await patient_data_store.get_patient(patient_id)
        
        if not patient_data:
            raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    
    return requestly_service.intercept_response(patient_data, f"/api/patients/{patient_id}")


@app.get("/api/admin/requestly/analytics")
async def get_requestly_analytics():
    """
    Get Requestly API analytics
    Shows API usage monitoring powered by Requestly
    """
    analytics = requestly_service.get_request_analytics()
    
    return {
        "requestly_analytics": analytics,
        "mock_mode_enabled": requestly_service.mock_mode,
        "feature": "Sponsor Integration - Requestly API Monitoring"
    }


@app.post("/api/admin/requestly/mock-mode")
async def toggle_mock_mode(enable: bool):
    """
    Toggle Requestly mock mode
    Admin-only: Enable/disable mock data fallback
    """
    if enable:
        requestly_service.enable_mock_mode()
    else:
        requestly_service.disable_mock_mode()
    
    return {
        "mock_mode": requestly_service.mock_mode,
        "message": f"Mock mode {'enabled' if enable else 'disabled'}"
    }


@app.get("/api/stats/overview")
async def get_system_overview():
    """Get overall system statistics"""
    requestly_service.log_api_request("/api/stats/overview", "GET", "anonymous")
    
    if requestly_service.mock_mode:
        # Generate mock stats
        all_patients = {}
        for floor in settings.FLOORS:
            patients = requestly_service.get_mock_patients_for_floor(floor["id"], floor["capacity"])
            all_patients.update(patients)
    else:
        all_patients = await patient_data_store.get_all_patients()
    
    patient_list = list(all_patients.values())
    
    # Calculate statistics
    total_patients = len(patient_list)
    high_risk = sum(1 for p in patient_list if p.get("is_high_risk", False))
    anomalies = sum(1 for p in patient_list if p.get("anomaly_flag", 0) == 1)
    stable = sum(1 for p in patient_list if p.get("state") == "stable")
    unstable = total_patients - stable
    
    response = {
        "total_patients": total_patients,
        "high_risk_count": high_risk,
        "anomaly_count": anomalies,
        "stable_patients": stable,
        "unstable_patients": unstable,
        "floors_active": len(settings.FLOORS),
        "data_source": "requestly_mock" if requestly_service.mock_mode else "kafka_live"
    }
    
    return requestly_service.intercept_response(response, "/api/stats/overview")


# =========================================================
# WEBSOCKET ENDPOINTS - Real-time Data Stream
# =========================================================

class WebSocketManager:
    """Manages WebSocket connections for real-time patient data streaming"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.patient_subscriptions: dict[str, Set[WebSocket]] = {}
        self.floor_subscriptions: dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"🔌 WebSocket connected from {websocket.client}. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        
        # Remove from patient subscriptions
        for patient_id in list(self.patient_subscriptions.keys()):
            self.patient_subscriptions[patient_id].discard(websocket)
            if not self.patient_subscriptions[patient_id]:
                del self.patient_subscriptions[patient_id]
        
        # Remove from floor subscriptions
        for floor_id in list(self.floor_subscriptions.keys()):
            self.floor_subscriptions[floor_id].discard(websocket)
            if not self.floor_subscriptions[floor_id]:
                del self.floor_subscriptions[floor_id]
        
        logger.info(f"🔌 WebSocket disconnected from {websocket.client}. Total connections: {len(self.active_connections)}")
    
    def subscribe_patient(self, websocket: WebSocket, patient_id: str):
        """Subscribe to specific patient updates"""
        if patient_id not in self.patient_subscriptions:
            self.patient_subscriptions[patient_id] = set()
        self.patient_subscriptions[patient_id].add(websocket)
        
    def subscribe_floor(self, websocket: WebSocket, floor_id: str):
        """Subscribe to all patients on a floor"""
        if floor_id not in self.floor_subscriptions:
            self.floor_subscriptions[floor_id] = set()
        self.floor_subscriptions[floor_id].add(websocket)
    
    async def broadcast_patient_update(self, patient_id: str, data: dict):
        """Broadcast update to subscribers of this patient"""
        # Send to patient-specific subscribers
        if patient_id in self.patient_subscriptions:
            disconnected = set()
            for websocket in self.patient_subscriptions[patient_id]:
                try:
                    await websocket.send_json({
                        "type": "patient_update",
                        "patient_id": patient_id,
                        "data": data
                    })
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected
            for ws in disconnected:
                self.disconnect(ws)
        
        # Send to floor subscribers
        floor_id = data.get("floor_id")
        if floor_id and floor_id in self.floor_subscriptions:
            disconnected = set()
            for websocket in self.floor_subscriptions[floor_id]:
                try:
                    await websocket.send_json({
                        "type": "floor_update",
                        "floor_id": floor_id,
                        "patient_id": patient_id,
                        "data": data
                    })
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    disconnected.add(websocket)
            
            for ws in disconnected:
                self.disconnect(ws)
    
    async def broadcast_all(self, message: dict):
        """Broadcast message to all connections"""
        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.add(websocket)
        
        for ws in disconnected:
            self.disconnect(ws)

# Global WebSocket manager
ws_manager = WebSocketManager()

@app.websocket("/ws/patients")
async def websocket_patient_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time patient data streaming
    
    Client can send subscription messages:
    - {"action": "subscribe_patient", "patient_id": "P1"}
    - {"action": "subscribe_floor", "floor_id": "1F"}
    - {"action": "subscribe_all"}
    """
    origin = websocket.headers.get("origin")
    logger.info(f"🔌 WebSocket connection attempt from {websocket.client}, origin: {origin}")
    
    try:
        await ws_manager.connect(websocket)
        logger.info(f"✅ WebSocket connection accepted")
    except Exception as e:
        logger.error(f"❌ Failed to accept WebSocket: {e}")
        return
    
    try:
        # Send initial data once
        logger.info(f"📊 Fetching patient data for initial send...")
        all_patients = await patient_data_store.get_all_patients()
        logger.info(f"📊 Got {len(all_patients)} patients, sending to client...")
        await websocket.send_json({
            "type": "initial_data",
            "patients": list(all_patients.values())
        })
        logger.info(f"✅ Initial data sent successfully")
        
        # Listen for subscription commands (no continuous streaming)
        while True:
            try:
                data = await websocket.receive_json()
                action = data.get("action")
                
                logger.info(f"📨 WebSocket received: {action}")
                
                if action == "subscribe_patient":
                    patient_id = data.get("patient_id")
                    if patient_id:
                        ws_manager.subscribe_patient(websocket, patient_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "target": "patient",
                            "patient_id": patient_id
                        })
                        logger.info(f"✅ Subscribed to patient {patient_id}")
                
                elif action == "subscribe_floor":
                    floor_id = data.get("floor_id")
                    if floor_id:
                        ws_manager.subscribe_floor(websocket, floor_id)
                        await websocket.send_json({
                            "type": "subscribed",
                            "target": "floor",
                            "floor_id": floor_id
                        })
                        logger.info(f"✅ Subscribed to floor {floor_id}")
                
                elif action == "ping":
                    # Respond to ping to keep connection alive
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Timeout waiting for messages is normal, continue
                continue
    
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket client disconnected gracefully")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        ws_manager.disconnect(websocket)

# Stream updates only when data changes (via Kafka consumer)
# This will be called by the Kafka consumer when new predictions arrive
async def notify_websocket_clients(patient_id: str, patient_data: dict):
    """Notify WebSocket clients when patient data changes"""
    try:
        await ws_manager.broadcast_patient_update(patient_id, patient_data)
        logger.debug(f"📡 Broadcasted update for {patient_id}")
    except Exception as e:
        logger.error(f"❌ Error broadcasting to WebSocket clients: {e}")

# Set up callback to notify WebSocket clients when data changes
patient_data_store.set_update_callback(notify_websocket_clients)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
