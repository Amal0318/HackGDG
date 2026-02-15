#!/usr/bin/env python3
"""
Simple file-based backend service for ICU data
Reads simulator_output.json and serves real patient data via FastAPI
"""

import json
import re
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ICU Data Service", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientDataManager:
    def __init__(self):
        self.patients: Dict[str, dict] = {}
        self.vitals_history: Dict[str, List[dict]] = {}
        self.risk_history: Dict[str, List[dict]] = {}
        self.websocket_connections: List[WebSocket] = []
    
    def parse_simulator_data(self, file_path: str = "../data/simulator_output.json"):
        """Parse the simulator output file to extract patient data"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract JSON objects from log lines
            json_pattern = r'\{[^}]+\}'
            matches = re.findall(json_pattern, content)
            
            latest_patients = {}
            
            for match in matches:
                try:
                    data = json.loads(match)
                    if 'patient_id' in data and 'vitals' in data:
                        patient_id = data['patient_id']
                        
                        # Parse vitals
                        vitals = data['vitals']
                        timestamp = data.get('timestamp', datetime.now().isoformat())
                        
                        # Create patient record
                        patient_data = {
                            'patient_id': patient_id,
                            'name': f'Patient {patient_id}',
                            'bed_number': patient_id,
                            'floor': '3',
                            'vitals': {
                                'heart_rate': vitals.get('HR', 0),
                                'systolic_bp': vitals.get('SBP', 0),
                                'diastolic_bp': vitals.get('DBP', 0),
                                'spo2': vitals.get('SpO2', 0),
                                'respiratory_rate': vitals.get('RR', 0),
                                'temperature': vitals.get('Temp', 37.0)
                            },
                            'latest_risk_score': data.get('computed_risk', data.get('risk_score', 0.1)),
                            'timestamp': timestamp,
                            'abnormal_vitals': []
                        }
                        
                        latest_patients[patient_id] = patient_data
                        
                        # Add to history
                        if patient_id not in self.vitals_history:
                            self.vitals_history[patient_id] = []
                        if patient_id not in self.risk_history:
                            self.risk_history[patient_id] = []
                        
                        # Add vitals to history
                        vitals_point = {
                            'timestamp': timestamp,
                            'heart_rate': vitals.get('HR', 0),
                            'systolic_bp': vitals.get('SBP', 0),
                            'diastolic_bp': vitals.get('DBP', 0),
                            'spo2': vitals.get('SpO2', 0),
                            'respiratory_rate': vitals.get('RR', 0),
                            'temperature': vitals.get('Temp', 37.0)
                        }
                        self.vitals_history[patient_id].append(vitals_point)
                        
                        # Add risk to history
                        risk_point = {
                            'timestamp': timestamp,
                            'risk_score': data.get('computed_risk', data.get('risk_score', 0.1))
                        }
                        self.risk_history[patient_id].append(risk_point)
                        
                        # Keep only last 60 points
                        self.vitals_history[patient_id] = self.vitals_history[patient_id][-60:]
                        self.risk_history[patient_id] = self.risk_history[patient_id][-60:]
                
                except json.JSONDecodeError:
                    continue
            
            self.patients = latest_patients
            logger.info(f"Parsed {len(self.patients)} patients from simulator data")
            
        except Exception as e:
            logger.error(f"Error parsing simulator data: {e}")
    
    async def notify_websockets(self, patient_id: str, data: dict):
        """Notify all connected WebSocket clients of patient updates"""
        if not self.websocket_connections:
            return
        
        message = {
            'type': 'patient_update',
            'patient_id': patient_id,
            'data': data
        }
        
        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)

# Initialize data manager
data_manager = PatientDataManager()

@app.on_event("startup")
async def startup_event():
    """Load initial data on startup"""
    data_manager.parse_simulator_data()

@app.get("/api/patients")
async def get_patients():
    """Get all patients"""
    # Refresh data from file
    data_manager.parse_simulator_data()
    return list(data_manager.patients.values())

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: str):
    """Get specific patient data"""
    data_manager.parse_simulator_data()
    if patient_id in data_manager.patients:
        return data_manager.patients[patient_id]
    return {"error": "Patient not found"}

@app.get("/api/patients/{patient_id}/vitals-history")
async def get_patient_vitals_history(patient_id: str):
    """Get patient vitals history"""
    data_manager.parse_simulator_data()
    return data_manager.vitals_history.get(patient_id, [])

@app.get("/api/patients/{patient_id}/risk-history") 
async def get_patient_risk_history(patient_id: str):
    """Get patient risk history"""
    data_manager.parse_simulator_data()
    return data_manager.risk_history.get(patient_id, [])

@app.websocket("/ws/patients")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time patient data updates"""
    await websocket.accept()
    data_manager.websocket_connections.append(websocket)
    
    try:
        # Send initial data
        await websocket.send_json({
            'type': 'subscribed',
            'message': 'Connected to patient data stream'
        })
        
        while True:
            # Periodically refresh and send updates
            data_manager.parse_simulator_data()
            
            for patient_id, patient_data in data_manager.patients.items():
                await data_manager.notify_websockets(patient_id, patient_data)
            
            await asyncio.sleep(2)  # Update every 2 seconds
            
    except WebSocketDisconnect:
        data_manager.websocket_connections.remove(websocket)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ICU Data Service Running", "patients": len(data_manager.patients)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")