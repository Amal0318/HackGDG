# VitalX Streaming Refactor — Implementation Templates

**Quick Reference for Developers**

---

## 🚀 PHASE 1: Vital Simulator

### File: `icu-system/vital-simulator/app/drift_model.py`

```python
"""
Realistic Physiological Drift Model
No state machines, no spikes - just gradual changes
"""

import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class PatientBaseline:
    """Individual patient baseline parameters"""
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    spo2: float
    respiratory_rate: float
    temperature: float
    lactate: float

class PhysiologicalDriftModel:
    """Models gradual physiological changes using Brownian motion"""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        
        # Initialize baselines (realistic ranges)
        self.baseline = PatientBaseline(
            heart_rate=random.uniform(65, 85),
            systolic_bp=random.uniform(110, 130),
            diastolic_bp=random.uniform(60, 80),
            spo2=random.uniform(96, 99),
            respiratory_rate=random.uniform(12, 18),
            temperature=random.uniform(36.2, 37.2),
            lactate=random.uniform(0.5, 1.5)
        )
        
        # Drift rates (change per minute)
        self.hr_drift = 0.0
        self.sbp_drift = 0.0
        self.lactate_drift = 0.0
        
        # Current values
        self.current_hr = self.baseline.heart_rate
        self.current_sbp = self.baseline.systolic_bp
        self.current_dbp = self.baseline.diastolic_bp
        self.current_spo2 = self.baseline.spo2
        self.current_rr = self.baseline.respiratory_rate
        self.current_temp = self.baseline.temperature
        self.current_lactate = self.baseline.lactate
        
        # Deterioration flag
        self.is_deteriorating = False
        
    def trigger_deterioration(self):
        """Probabilistic deterioration trigger (5% chance per call)"""
        if not self.is_deteriorating and random.random() < 0.05:
            self.is_deteriorating = True
            self.hr_drift = random.uniform(0.1, 0.5)  # HR increases
            self.sbp_drift = -random.uniform(0.05, 0.3)  # BP decreases
            self.lactate_drift = random.uniform(0.01, 0.05)  # Lactate increases
            
    def update(self, dt: float = 1.0) -> dict:
        """
        Update vital signs
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Dictionary of current vital signs
        """
        # Check for deterioration trigger
        self.trigger_deterioration()
        
        # Apply drift
        dt_minutes = dt / 60.0
        self.current_hr += self.hr_drift * dt_minutes
        self.current_sbp += self.sbp_drift * dt_minutes
        self.current_lactate += self.lactate_drift * dt_minutes
        
        # Add Gaussian noise (realistic variability)
        noise_hr = np.random.normal(0, 1.5)
        noise_sbp = np.random.normal(0, 2.0)
        noise_spo2 = np.random.normal(0, 0.5)
        noise_rr = np.random.normal(0, 0.8)
        noise_temp = np.random.normal(0, 0.1)
        
        # Update with noise
        hr = self.current_hr + noise_hr
        sbp = self.current_sbp + noise_sbp
        spo2 = self.current_spo2 + noise_spo2
        rr = self.current_rr + noise_rr
        temp = self.current_temp + noise_temp
        lactate = self.current_lactate
        
        # Calculate derived values
        dbp = sbp * 0.65  # Approximate DBP
        map_value = dbp + (sbp - dbp) / 3.0  # Mean Arterial Pressure
        shock_index = hr / max(sbp, 1.0)  # Prevent division by zero
        
        # Enforce physiological bounds
        hr = np.clip(hr, 40, 180)
        sbp = np.clip(sbp, 70, 200)
        dbp = np.clip(dbp, 40, 120)
        spo2 = np.clip(spo2, 88, 100)
        rr = np.clip(rr, 8, 35)
        temp = np.clip(temp, 35.0, 40.0)
        lactate = np.clip(lactate, 0.5, 8.0)
        
        return {
            'patient_id': self.patient_id,
            'timestamp': None,  # Will be set by caller
            'heart_rate': round(hr, 1),
            'systolic_bp': round(sbp, 1),
            'diastolic_bp': round(dbp, 1),
            'map': round(map_value, 1),
            'spo2': round(spo2, 1),
            'respiratory_rate': round(rr, 1),
            'temperature': round(temp, 1),
            'lactate': round(lactate, 2),
            'shock_index': round(shock_index, 2)
        }
```

### File: `icu-system/vital-simulator/app/main.py` (Refactored)

```python
"""
Vital Simulator Service - Clean Drift Model Implementation
No state machines, no spikes, just realistic physiology
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timezone
from kafka import KafkaProducer
from drift_model import PhysiologicalDriftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vital-simulator")

class VitalSimulator:
    """Clean vital signs simulator using drift model"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
        self.kafka_topic = 'vitals_raw'
        self.num_patients = 8
        
        # Initialize drift models for each patient
        self.patients = {
            f'P{str(i).zfill(3)}': PhysiologicalDriftModel(f'P{str(i).zfill(3)}')
            for i in range(1, self.num_patients + 1)
        }
        
        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            enable_idempotence=True,
            acks='all',
            retries=3
        )
        
        logger.info(f"Initialized simulator for {self.num_patients} patients")
        logger.info(f"Publishing to topic: {self.kafka_topic}")
        
    async def run(self):
        """Main simulation loop"""
        logger.info("Starting simulation loop (1 Hz per patient)")
        
        while True:
            try:
                # Update all patients
                for patient_id, model in self.patients.items():
                    # Generate vitals
                    vitals = model.update(dt=1.0)
                    vitals['timestamp'] = datetime.now(timezone.utc).isoformat()
                    
                    # Publish to Kafka
                    self.producer.send(self.kafka_topic, value=vitals)
                    
                # Wait 1 second
                await asyncio.sleep(1.0)
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(1.0)
                
        self.producer.close()

if __name__ == "__main__":
    simulator = VitalSimulator()
    asyncio.run(simulator.run())
```

---

## 🧠 PHASE 3: Pathway Feature Engineering

### File: `icu-system/pathway-engine/app/feature_engineering.py`

```python
"""
Streaming Feature Engineering
Deterministic feature computation only - NO risk scoring
"""

import pathway as pw
import logging

logger = logging.getLogger(__name__)

def create_feature_pipeline(vitals_stream: pw.Table) -> pw.Table:
    """
    Transform raw vitals into enriched features
    
    Input: vitals_raw topic
    Output: vitals_enriched topic
    
    Features computed:
    - Rolling statistics (mean, std)
    - Deltas (change from window start)
    - Trends (lactate slope)
    - Anomaly flags (z-score based)
    """
    
    logger.info("Creating feature engineering pipeline")
    
    # Define sliding window (30 minutes, hop 10 seconds)
    windowed = vitals_stream.windowby(
        vitals_stream.patient_id,
        window=pw.temporal.sliding(
            hop=pw.Duration.seconds(10),
            duration=pw.Duration.minutes(30)
        ),
        instance=vitals_stream.timestamp
    )
    
    # Compute aggregates per window
    features = windowed.reduce(
        patient_id=pw.this.patient_id,
        timestamp=pw.reducers.latest(pw.this.timestamp),
        
        # Original vitals (latest)
        heart_rate=pw.reducers.latest(pw.this.heart_rate),
        systolic_bp=pw.reducers.latest(pw.this.systolic_bp),
        diastolic_bp=pw.reducers.latest(pw.this.diastolic_bp),
        map=pw.reducers.latest(pw.this.map),
        spo2=pw.reducers.latest(pw.this.spo2),
        respiratory_rate=pw.reducers.latest(pw.this.respiratory_rate),
        temperature=pw.reducers.latest(pw.this.temperature),
        lactate=pw.reducers.latest(pw.this.lactate),
        shock_index=pw.reducers.latest(pw.this.shock_index),
        
        # Rolling statistics
        rolling_mean_hr=pw.reducers.avg(pw.this.heart_rate),
        rolling_std_hr=pw.reducers.stddev(pw.this.heart_rate),
        rolling_mean_sbp=pw.reducers.avg(pw.this.systolic_bp),
        rolling_std_sbp=pw.reducers.stddev(pw.this.systolic_bp),
        rolling_mean_spo2=pw.reducers.avg(pw.this.spo2),
        
        # Deltas (latest - earliest in window)
        hr_delta=(
            pw.reducers.latest(pw.this.heart_rate) - 
            pw.reducers.earliest(pw.this.heart_rate)
        ),
        sbp_delta=(
            pw.reducers.latest(pw.this.systolic_bp) - 
            pw.reducers.earliest(pw.this.systolic_bp)
        ),
        shock_index_delta=(
            pw.reducers.latest(pw.this.shock_index) - 
            pw.reducers.earliest(pw.this.shock_index)
        ),
        
        # Max/min for range
        max_hr=pw.reducers.max(pw.this.heart_rate),
        min_hr=pw.reducers.min(pw.this.heart_rate),
        max_shock_index=pw.reducers.max(pw.this.shock_index),
    )
    
    # Add anomaly flags (z-score based)
    features_with_anomalies = features.select(
        pw.this.patient_id,
        pw.this.timestamp,
        pw.this.heart_rate,
        pw.this.systolic_bp,
        pw.this.spo2,
        pw.this.shock_index,
        pw.this.rolling_mean_hr,
        pw.this.rolling_std_hr,
        pw.this.hr_delta,
        pw.this.sbp_delta,
        pw.this.shock_index_delta,
        pw.this.max_shock_index,
        
        # Anomaly detection
        anomaly_flag=(
            (pw.this.heart_rate > pw.this.rolling_mean_hr + 2.5 * pw.this.rolling_std_hr) |
            (pw.this.systolic_bp < 90) |
            (pw.this.shock_index > 1.3)
        )
    )
    
    logger.info("Feature pipeline configured")
    return features_with_anomalies
```

---

## 🔍 PHASE 4: Pathway Streaming RAG

### File: `icu-system/pathway-engine/app/streaming_rag.py`

```python
"""
Live Streaming Vector Index for RAG
Real-time embeddings with sliding window expiry
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class StreamingRAGIndex:
    """
    Patient-isolated streaming vector index
    Maintains last N hours of embeddings per patient
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', window_hours=3):
        self.model = SentenceTransformer(model_name)
        self.window_hours = window_hours
        
        # Per-patient storage: {patient_id: [(text, embedding, timestamp), ...]}
        self.patient_indices = defaultdict(list)
        
        logger.info(f"Initialized RAG index with {model_name}, {window_hours}h window")
        
    def add_enriched_event(self, event: dict):
        """Add enriched vital event to index"""
        
        # Convert event to text chunk
        chunk = self._event_to_text(event)
        
        # Embed
        embedding = self.model.encode(chunk)
        
        # Store
        patient_id = event['patient_id']
        timestamp = datetime.fromisoformat(event['timestamp'])
        
        self.patient_indices[patient_id].append({
            'text': chunk,
            'embedding': embedding,
            'timestamp': timestamp,
            'raw_data': event
        })
        
        # Cleanup old entries
        self._cleanup_expired(patient_id)
        
    def _event_to_text(self, event: dict) -> str:
        """Convert enriched event to structured text"""
        
        time_str = event['timestamp'].split('T')[1][:5]  # HH:MM
        
        text_parts = [
            f"Time {time_str}",
            f"Patient {event['patient_id']}",
            f"HR {event['heart_rate']:.0f}",
        ]
        
        if 'hr_delta' in event:
            delta_sign = "↑" if event['hr_delta'] > 0 else "↓"
            text_parts.append(f"({delta_sign}{abs(event['hr_delta']):.1f})")
            
        text_parts.extend([
            f"SBP {event['systolic_bp']:.0f}",
            f"ShockIndex {event['shock_index']:.2f}",
        ])
        
        if 'anomaly_flag' in event and event['anomaly_flag']:
            text_parts.append("ANOMALY")
            
        return " | ".join(text_parts)
        
    def _cleanup_expired(self, patient_id: str):
        """Remove embeddings older than window"""
        
        cutoff = datetime.now() - timedelta(hours=self.window_hours)
        
        self.patient_indices[patient_id] = [
            entry for entry in self.patient_indices[patient_id]
            if entry['timestamp'] > cutoff
        ]
        
    def query(
        self, 
        patient_id: str, 
        query_text: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Query patient's index
        
        Args:
            patient_id: Patient identifier
            query_text: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of retrieved contexts with relevance scores
        """
        
        if patient_id not in self.patient_indices:
            logger.warning(f"No index for patient {patient_id}")
            return []
            
        # Embed query
        query_embedding = self.model.encode(query_text)
        
        # Compute similarities
        patient_index = self.patient_indices[patient_id]
        similarities = []
        
        for entry in patient_index:
            similarity = self._cosine_similarity(
                query_embedding, 
                entry['embedding']
            )
            similarities.append((similarity, entry))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k
        results = []
        for similarity, entry in similarities[:top_k]:
            results.append({
                'text': entry['text'],
                'timestamp': entry['timestamp'].isoformat(),
                'relevance_score': float(similarity),
                'raw_data': entry['raw_data']
            })
            
        return results
        
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_patients': len(self.patient_indices),
            'total_embeddings': sum(len(idx) for idx in self.patient_indices.values()),
            'per_patient': {
                pid: len(idx) for pid, idx in self.patient_indices.items()
            }
        }
```

### File: `icu-system/pathway-engine/app/query_api.py`

```python
"""
HTTP Query Interface for Pathway RAG
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Pathway RAG Query API")

# Global reference to RAG index (set by main.py)
rag_index = None

class QueryRequest(BaseModel):
    patient_id: str
    query_text: str
    top_k: int = 5

class QueryResponse(BaseModel):
    patient_id: str
    retrieved_context: List[Dict]

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query patient's streaming RAG index"""
    
    if rag_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
        
    try:
        results = rag_index.query(
            request.patient_id,
            request.query_text,
            request.top_k
        )
        
        return QueryResponse(
            patient_id=request.patient_id,
            retrieved_context=results
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "rag_index": rag_index.get_stats() if rag_index else None
    }
```

---

## 🤖 PHASE 6: ML Service

### File: `icu-system/ml-service/app/main.py`

```python
"""
ML Service - Risk Score Authority
Consumes enriched data, publishes predictions
"""

import json
import logging
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
from collections import deque, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-service")

class PatientSequenceBuffer:
    """Maintain rolling sequence buffer per patient"""
    
    def __init__(self, window_size=60):
        self.window_size = window_size
        self.buffers = defaultdict(lambda: deque(maxlen=window_size))
        
    def add(self, patient_id: str, features: dict):
        """Add feature vector to patient's buffer"""
        
        # Extract features
        feature_vector = [
            features.get('heart_rate', 0),
            features.get('systolic_bp', 0),
            features.get('spo2', 0),
            features.get('shock_index', 0),
            features.get('rolling_mean_hr', 0),
            # Add more features as needed
        ]
        
        self.buffers[patient_id].append(feature_vector)
        
    def is_ready(self, patient_id: str) -> bool:
        """Check if buffer has enough data"""
        return len(self.buffers[patient_id]) >= self.window_size
        
    def get_sequence(self, patient_id: str) -> np.ndarray:
        """Get full sequence as numpy array"""
        return np.array(list(self.buffers[patient_id]))

def predict_risk(sequence: np.ndarray) -> float:
    """
    Placeholder prediction function
    
    Args:
        sequence: [window_size, num_features] array
        
    Returns:
        risk_score: Float between 0 and 1
    """
    
    # PLACEHOLDER: Replace with actual model inference
    # For now, use weighted average of shock index
    shock_index_values = sequence[:, 3]  # Assuming shock_index is 4th feature
    mean_shock_index = np.mean(shock_index_values)
    
    # Simple heuristic: map shock index to risk
    risk_score = np.clip((mean_shock_index - 0.5) / 1.5, 0.0, 1.0)
    
    return float(risk_score)

def main():
    kafka_servers = 'kafka:9092'
    
    # Initialize Kafka consumer
    consumer = KafkaConsumer(
        'vitals_enriched',
        bootstrap_servers=kafka_servers,
        group_id='ml-service',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    # Initialize Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=kafka_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        enable_idempotence=True
    )
    
    # Initialize sequence buffer
    sequence_buffer = PatientSequenceBuffer(window_size=60)
    
    logger.info("ML Service started, consuming from vitals_enriched")
    
    for message in consumer:
        try:
            enriched_data = message.value
            patient_id = enriched_data['patient_id']
            
            # Add to buffer
            sequence_buffer.add(patient_id, enriched_data)
            
            # Predict if buffer is ready
            if sequence_buffer.is_ready(patient_id):
                sequence = sequence_buffer.get_sequence(patient_id)
                risk_score = predict_risk(sequence)
                
                # Create prediction message
                prediction = {
                    'patient_id': patient_id,
                    'timestamp': enriched_data['timestamp'],
                    'risk_score': risk_score
                }
                
                # Publish to predictions topic
                producer.send('vitals_predictions', value=prediction)
                
                logger.debug(f"{patient_id}: risk_score={risk_score:.3f}")
                
        except Exception as e:
            logger.error(f"Processing error: {e}")

if __name__ == "__main__":
    main()
```

---

## 🌐 PHASE 7: Backend API

### File: `icu-system/backend-api/app/stream_merger.py`

```python
"""
Stream Merger - Join vitals_enriched + vitals_predictions
"""

import json
import logging
import threading
from kafka import KafkaConsumer
from collections import defaultdict
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class StreamMerger:
    """Merge enriched vitals with ML predictions"""
    
    def __init__(self, kafka_servers='kafka:9092'):
        self.kafka_servers = kafka_servers
        self.patient_state = defaultdict(dict)
        self.listeners = []
        
        # Start consumer threads
        threading.Thread(target=self._consume_vitals, daemon=True).start()
        threading.Thread(target=self._consume_predictions, daemon=True).start()
        
        logger.info("Stream merger initialized")
        
    def _consume_vitals(self):
        """Consume vitals_enriched topic"""
        
        consumer = KafkaConsumer(
            'vitals_enriched',
            bootstrap_servers=self.kafka_servers,
            group_id='backend-vitals',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        for message in consumer:
            data = message.value
            patient_id = data['patient_id']
            
            self.patient_state[patient_id]['vitals'] = data
            self._notify_listeners(patient_id)
            
    def _consume_predictions(self):
        """Consume vitals_predictions topic"""
        
        consumer = KafkaConsumer(
            'vitals_predictions',
            bootstrap_servers=self.kafka_servers,
            group_id='backend-predictions',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        for message in consumer:
            data = message.value
            patient_id = data['patient_id']
            
            self.patient_state[patient_id]['prediction'] = data
            self._notify_listeners(patient_id)
            
    def get_unified_view(self, patient_id: str) -> Optional[Dict]:
        """Get merged view for patient"""
        
        state = self.patient_state.get(patient_id, {})
        
        if 'vitals' not in state:
            return None
            
        vitals = state['vitals']
        prediction = state.get('prediction', {})
        
        return {
            'patient_id': patient_id,
            'timestamp': vitals.get('timestamp'),
            'vitals': {
                'heart_rate': vitals.get('heart_rate'),
                'systolic_bp': vitals.get('systolic_bp'),
                'spo2': vitals.get('spo2'),
                'shock_index': vitals.get('shock_index'),
            },
            'features': {
                'rolling_mean_hr': vitals.get('rolling_mean_hr'),
                'hr_delta': vitals.get('hr_delta'),
                'anomaly_flag': vitals.get('anomaly_flag'),
            },
            'risk_score': prediction.get('risk_score', 0.0)
        }
        
    def register_listener(self, callback):
        """Register callback for updates"""
        self.listeners.append(callback)
        
    def _notify_listeners(self, patient_id: str):
        """Notify all listeners of update"""
        unified_view = self.get_unified_view(patient_id)
        if unified_view:
            for callback in self.listeners:
                callback(unified_view)
```

### File: `icu-system/backend-api/app/main.py`

```python
"""
Backend API - Orchestrator
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import logging

from stream_merger import StreamMerger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend-api")

app = FastAPI(title="VitalX Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize stream merger
stream_merger = StreamMerger()

# WebSocket connections
websocket_connections = []

def broadcast_update(unified_view):
    """Broadcast update to all WebSocket clients"""
    for ws in websocket_connections:
        try:
            ws.send_json(unified_view)
        except:
            websocket_connections.remove(ws)

stream_merger.register_listener(broadcast_update)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except:
        websocket_connections.remove(websocket)

@app.get("/patients")
async def get_patients():
    """Get list of all patients"""
    return {
        "patients": list(stream_merger.patient_state.keys())
    }

@app.get("/patients/{patient_id}/latest")
async def get_patient_latest(patient_id: str):
    """Get latest unified view for patient"""
    view = stream_merger.get_unified_view(patient_id)
    if not view:
        return {"error": "Patient not found"}
    return view

class ChatRequest(BaseModel):
    patient_id: str
    question: str

@app.post("/chat")
async def chat(request: ChatRequest):
    """RAG-powered chat endpoint"""
    
    # Query Pathway RAG
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://pathway-engine:8080/query',
            json={
                'patient_id': request.patient_id,
                'query_text': request.question
            }
        )
        
    context = response.json()['retrieved_context']
    
    # TODO: Call LLM with context
    # For now, return context directly
    answer = f"Based on recent vitals: {context[0]['text']}" if context else "No context available"
    
    return {
        "answer": answer,
        "sources": context
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "patients_tracked": len(stream_merger.patient_state)
    }
```

---

## 📝 DOCKER-COMPOSE UPDATES

### File: `icu-system/docker-compose.yml` (Key changes)

```yaml
services:
  pathway-engine:
    build:
      context: ./pathway-engine
    ports:
      - "8080:8080"  # Expose query API
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      RAG_WINDOW_HOURS: 3
      
  ml-service:
    build:
      context: ./ml-service
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      SEQUENCE_WINDOW_SIZE: 60
      
  backend-api:
    build:
      context: ./backend-api
    ports:
      - "8000:8000"
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      PATHWAY_API_URL: http://pathway-engine:8080
```

---

## ✅ TESTING COMMANDS

### Test Vital Simulator
```bash
# Check Kafka messages
docker exec -it icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_raw \
  --from-beginning --max-messages 10
```

### Test Pathway Features
```bash
docker exec -it icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_enriched \
  --from-beginning --max-messages 10
```

### Test RAG Query
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "query_text": "Why is shock index increasing?"
  }'
```

### Test ML Predictions
```bash
docker exec -it icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_predictions \
  --from-beginning --max-messages 10
```

### Test Backend API
```bash
curl http://localhost:8000/patients
curl http://localhost:8000/patients/P001/latest
```

---

**Ready to implement!** 🚀
