# VitalX - Production Deployment Guide

## Architecture Overview

**Streaming-First, ML-Authoritative Architecture**

```
Vital Simulator → Pathway Engine → ML Service → Backend API → Frontend
     ↓                  ↓              ↓             ↓
   kafka          kafka (RAG)      kafka         WebSocket
 vitals_raw    vitals_enriched  vitals_predictions
                                     ↓
                               Alert Engine
                              alerts_stream
```

## Service Responsibilities

### 1. **Vital Simulator** (`vital-simulator/`)
- **Purpose**: Generates realistic physiological vitals using drift model
- **Input**: None
- **Output**: Kafka topic `vitals_raw`
- **Key Features**: 
  - Mean-reversion drift with stress episodes
  - Clinical bounds enforcement
  - NO state machine logic
- **Port**: None (internal)

### 2. **Pathway Engine** (`pathway-engine/`)
- **Purpose**: Feature engineering + Streaming RAG
- **Input**: Kafka topic `vitals_raw`
- **Output**: Kafka topic `vitals_enriched`
- **Key Features**:
  - Sliding window aggregation (30min duration, 10sec hop)
  - 25 computed features (rolling stats, deltas, anomalies)
  - Live vector index with patient isolation (3-hour memory)
  - Query API for RAG retrieval
  - **NO RISK SCORING** (ML Service only)
- **Port**: 8080 (Query API)

### 3. **ML Service** (`ml-service/`)
- **Purpose**: **SOLE RISK AUTHORITY** - Predicts patient risk
- **Input**: Kafka topic `vitals_enriched`
- **Output**: Kafka topic `vitals_predictions`
- **Key Features**:
  - PatientSequenceBuffer (60-step history)
  - Placeholder inference (weighted heuristic for now)
  - Sole source of `risk_score`
- **Port**: None (internal)

### 4. **Backend API** (`backend-api/`)
- **Purpose**: Stream merger + orchestration
- **Input**: 
  - Kafka topic `vitals_enriched`
  - Kafka topic `vitals_predictions`
- **Output**: 
  - WebSocket `/ws`
  - REST endpoints
- **Key Features**:
  - Merges vitals + features + risk_score into unified view
  - WebSocket broadcasting for real-time updates
  - REST API for queries
  - Chat endpoint (queries Pathway RAG API)
  - **NO RISK CALCULATION** (consumes ML predictions only)
- **Port**: 8000

### 5. **Alert Engine** (`alert-engine/`)
- **Purpose**: Threshold-based alerting
- **Input**: Kafka topic `vitals_predictions`
- **Output**: Kafka topic `alerts_stream`
- **Key Features**:
  - Triggers alerts when `risk_score > threshold` (default 0.75)
  - Alert cooldown (5 minutes)
  - Severity levels (MEDIUM/HIGH/CRITICAL)
  - **NO RULE-BASED RISK LOGIC**
- **Port**: None (internal)

## Kafka Topics

| Topic | Producer | Consumer(s) | Schema |
|-------|----------|-------------|--------|
| `vitals_raw` | vital-simulator | pathway-engine | {patient_id, timestamp, vitals} |
| `vitals_enriched` | pathway-engine | ml-service, backend-api | {patient_id, timestamp, vitals, features, anomaly_flags} |
| `vitals_predictions` | ml-service | backend-api, alert-engine | {patient_id, timestamp, risk_score} |
| `alerts_stream` | alert-engine | (optional consumers) | {alert_id, patient_id, timestamp, risk_score, severity} |

## Folder Structure

```
icu-system/
├── docker-compose.yml          # Updated with new services
├── vital-simulator/
│   ├── Dockerfile             # Updated to use main_new.py
│   ├── requirements.txt
│   └── app/
│       ├── drift_model.py     # NEW: PhysiologicalDriftModel
│       ├── main_new.py        # NEW: Drift-based simulator
│       └── main.py            # OLD: Can be removed
├── pathway-engine/
│   ├── Dockerfile             # Updated to use main_new.py
│   ├── requirements.txt       # Already includes sentence-transformers, fastapi
│   └── app/
│       ├── feature_engineering.py  # NEW: Feature pipeline
│       ├── streaming_rag.py        # NEW: Live vector index
│       ├── query_api.py            # NEW: RAG query API
│       ├── main_new.py             # NEW: Orchestrator
│       └── main.py                 # OLD: Can be removed
├── ml-service/
│   ├── Dockerfile             # Updated to use main_new.py
│   ├── requirements.txt
│   └── app/
│       ├── main_new.py        # NEW: Streaming ML service
│       └── main.py            # OLD: Can be removed
├── backend-api/
│   ├── Dockerfile             # Updated to use main_new.py
│   ├── requirements.txt
│   └── app/
│       ├── stream_merger.py   # NEW: Dual Kafka consumer
│       ├── main_new.py        # NEW: FastAPI with WebSocket
│       └── main.py            # OLD: Can be removed
├── alert-engine/
│   ├── Dockerfile             # NEW: Created
│   ├── requirements.txt       # Existing (kafka-python)
│   └── app/
│       └── main.py            # NEW: Threshold-based alerting
└── alert-system/              # LEGACY: Commented out in docker-compose
    └── (old email alert service with LLM)
```

## Files Removed/Deprecated

### Files to Delete (After Testing)

**vital-simulator/**
- `app/main.py` (replaced by `app/main_new.py`)

**pathway-engine/**
- `app/main.py` (replaced by `app/main_new.py`)
- Any old risk engine files

**ml-service/**
- `app/main.py` (replaced by `app/main_new.py`)
- Old REST API endpoints (if any)

**backend-api/**
- `app/main.py` (replaced by `app/main_new.py`)
- `app/risk_engine.py` (if exists)
- Any file with rule-based risk calculations

### Services Deprecated

**alert-system/** (Commented out in docker-compose)
- Old email alert service with LLM integration
- Can be re-enabled if email alerts needed

**rag-service/** (Commented out in docker-compose)
- Standalone RAG service removed
- RAG functionality now inside pathway-engine

## Deployment Instructions

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- 4 CPU cores recommended

### Environment Variables

Create `.env` file in `icu-system/` directory:

```bash
# Number of patients to simulate
NUM_PATIENTS=5

# Update interval (seconds)
UPDATE_INTERVAL_SECONDS=1

# Alert threshold (0.0 - 1.0)
ALERT_THRESHOLD=0.75

# Optional: Legacy email alerts (if alert-system enabled)
# GEMINI_API_KEY=your_key_here
# ENABLE_EMAIL_ALERTS=true
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USERNAME=your_email
# SMTP_PASSWORD=your_password
# ALERT_EMAIL_TO=admin@example.com
```

### Step 1: Build Services

```bash
cd icu-system
docker-compose build
```

### Step 2: Start Services

```bash
docker-compose up -d
```

This starts:
- Zookeeper (port 2181)
- Kafka (port 29092 external, 9092 internal)
- Vital Simulator
- Pathway Engine (port 8080)
- ML Service
- Backend API (port 8000)
- Alert Engine

### Step 3: Verify Services

**Check container status:**
```bash
docker-compose ps
```

Expected output:
```
NAME                      STATUS        PORTS
icu-zookeeper            Up            2181/tcp
icu-kafka                Up            29092->29092/tcp
icu-vital-simulator      Up            
icu-pathway-engine       Up            8080->8080/tcp
icu-ml-service           Up            
icu-backend-api          Up            8000->8000/tcp
icu-alert-engine         Up            
```

**Check logs:**
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f vital-simulator
docker-compose logs -f pathway-engine
docker-compose logs -f ml-service
docker-compose logs -f backend-api
docker-compose logs -f alert-engine
```

### Step 4: Verify Kafka Topics

**List topics:**
```bash
docker exec icu-kafka kafka-topics --bootstrap-server kafka:9092 --list
```

Expected topics:
- vitals_raw
- vitals_enriched
- vitals_predictions
- alerts_stream

**Consume topic (verify data flow):**
```bash
# Vitals raw data
docker exec icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_raw \
  --from-beginning \
  --max-messages 5

# Enriched vitals with features
docker exec icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_enriched \
  --from-beginning \
  --max-messages 5

# ML predictions
docker exec icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_predictions \
  --from-beginning \
  --max-messages 5

# Alerts
docker exec icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic alerts_stream \
  --from-beginning
```

## API Testing

### Backend API Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "backend-api",
  "stream_merger": "running"
}
```

### List Patients

```bash
curl http://localhost:8000/patients
```

Expected response:
```json
{
  "patients": [
    {
      "patient_id": "P001",
      "risk_score": 0.42,
      "last_update": "2024-01-15T10:30:45"
    },
    ...
  ]
}
```

### Get Patient Details

```bash
curl http://localhost:8000/patients/P001
```

Expected response includes:
- Vitals (heart_rate, blood_pressure, etc.)
- Features (rolling_mean_hr, hr_delta, etc.)
- Anomaly flags
- Risk score (from ML Service)

### Query RAG System

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "query": "What is the current trend for heart rate?"
  }'
```

### Query Pathway RAG API Directly

```bash
# Query patient context
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "query_text": "heart rate trends",
    "top_k": 5
  }'

# Get RAG statistics
curl http://localhost:8080/stats

# Health check
curl http://localhost:8080/health
```

## WebSocket Testing

Using `wscat` (install: `npm install -g wscat`):

```bash
wscat -c ws://localhost:8000/ws
```

You should receive real-time patient updates:
```json
{
  "event": "patient_update",
  "patient_id": "P001",
  "data": {
    "vitals": {...},
    "features": {...},
    "risk_score": 0.42,
    "timestamp": "2024-01-15T10:30:45"
  }
}
```

## Troubleshooting

### Services Not Starting

**Check Kafka health:**
```bash
docker exec icu-kafka kafka-broker-api-versions --bootstrap-server kafka:9092
```

**Restart specific service:**
```bash
docker-compose restart vital-simulator
```

### No Data in Topics

**Check simulator logs:**
```bash
docker-compose logs vital-simulator | grep "Published"
```

**Verify Kafka connectivity:**
```bash
docker exec icu-vital-simulator ping kafka
```

### ML Service Not Producing Predictions

**Check buffer size:**
```bash
docker-compose logs ml-service | grep "buffer size"
```

ML Service waits for 60 samples before predicting. With 1Hz updates, this takes ~60 seconds.

### Pathway Engine Errors

**Check for Pathway compilation issues:**
```bash
docker-compose logs pathway-engine | grep "ERROR"
```

**Verify sentence-transformers model download:**
```bash
docker-compose logs pathway-engine | grep "all-MiniLM"
```

### Backend API Not Receiving Data

**Check stream merger:**
```bash
docker-compose logs backend-api | grep "StreamMerger"
```

**Verify both consumers:**
```bash
docker-compose logs backend-api | grep "consume_vitals\|consume_predictions"
```

## Performance Validation

### Expected Metrics

- **Vitals Generation**: 1Hz per patient (~5 msg/sec for 5 patients)
- **Feature Enrichment**: <100ms latency (Pathway sliding window)
- **ML Predictions**: <50ms inference (placeholder)
- **End-to-End Latency**: <200ms (simulator → frontend)

### Monitoring Commands

**CPU/Memory usage:**
```bash
docker stats
```

**Message rate:**
```bash
docker exec icu-kafka kafka-consumer-groups \
  --bootstrap-server kafka:9092 \
  --describe \
  --group pathway-engine
```

## Shutdown

**Graceful shutdown:**
```bash
docker-compose down
```

**Full cleanup (including volumes):**
```bash
docker-compose down -v
```

## Next Steps

1. **Deploy Frontend**: Update frontend to connect to `ws://localhost:8000/ws`
2. **Replace Placeholder ML**: Integrate trained LSTM model in ml-service
3. **Add Observability**: Prometheus + Grafana for metrics
4. **Load Testing**: Simulate 50+ patients
5. **Production Kafka**: Replace local Kafka with managed cluster (Confluent Cloud, AWS MSK)

## Architecture Validation Checklist

- ✅ NO state machine logic in vital simulator
- ✅ NO risk scoring in Pathway Engine (features only)
- ✅ NO risk scoring in Backend API (merge only)
- ✅ ML Service is SOLE source of risk_score
- ✅ Streaming RAG inside Pathway (no standalone service)
- ✅ Clean separation of concerns
- ✅ Production logging (no emojis)
- ✅ Health check endpoints
- ✅ Graceful shutdown handlers
- ✅ Kafka retry logic
