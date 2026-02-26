# VitalX ICU Digital Twin System

## Executive Summary

VitalX is an enterprise-grade, real-time ICU patient monitoring and deterioration prediction system that leverages artificial intelligence, streaming data processing, and modern microservices architecture. The system provides sub-second latency from vital sign capture to risk assessment, enabling proactive clinical intervention through continuous AI-powered monitoring of critically ill patients.

### Core Capabilities

- Real-time physiological data streaming and processing for multiple ICU patients
- LSTM-based sepsis risk prediction trained on MIMIC-IV clinical dataset
- Pathway streaming engine for low-latency feature engineering and real-time RAG (Retrieval-Augmented Generation)
- LangChain-powered intelligent alert system with natural language explanations
- WebSocket-based live dashboard with sub-second data refresh
- AI-assisted clinical decision support via conversational RAG interface
- Automated PDF medical report generation with time-series visualizations
- Microservices architecture enabling horizontal scalability

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Technology Stack](#technology-stack)
3. [Component Overview](#component-overview)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [Pathway Streaming Engine](#pathway-streaming-engine)
6. [Machine Learning Model](#machine-learning-model)
7. [Installation & Deployment](#installation--deployment)
8. [API Documentation](#api-documentation)
9. [Configuration](#configuration)
10. [Monitoring & Operations](#monitoring--operations)

---

## System Architecture

### High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         VitalX ICU Digital Twin                             │
│              Real-time Patient Monitoring & Risk Prediction                 │
└────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌──────────────────┐         ┌──────────────────┐
│ Vital Simulator │────────▶│  Apache Kafka    │────────▶│ Pathway Engine   │
│  (8 Patients)   │         │  Message Broker  │         │ Feature Engineer │
│  Physiological  │         │  vitals_raw      │         │ + RAG Indexer    │
│  Data Generator │         │  vitals_enriched │         │                  │
└─────────────────┘         │  vitals_predict  │         └──────────────────┘
                            └──────────────────┘                   │
                                     │                             │
                                     │                             ▼
                                     │              ┌──────────────────────────┐
                                     │              │  Pathway RAG System      │
                                     │              │  • Semantic Embeddings   │
                                     │              │  • 3-hour Window         │
                                     │              │  • Patient Isolation     │
                                     │              │  • Query API :8080       │
                                     │              └──────────────────────────┘
                                     │
              ┌──────────────────────┼─────────────────────┬─────────────────┐
              ▼                      ▼                     ▼                 ▼
     ┌────────────────┐    ┌────────────────┐   ┌────────────────┐  ┌───────────────┐
     │  ML Service    │    │  Backend API   │   │ Alert System   │  │ Alert Engine  │
     │  LSTM Model    │    │  FastAPI       │   │ LangChain      │  │ Threshold     │
     │  Risk Predict  │    │  WebSocket     │   │ AI Alerts      │  │ Monitor       │
     │  Port: 8001    │    │  Port: 8000    │   │ Email/Console  │  │               │
     └────────────────┘    └────────────────┘   └────────────────┘  └───────────────┘
              │                      │
              │                      │
              └──────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   Frontend          │
              │   React TypeScript  │
              │   Tailwind CSS      │
              │   WebSocket Client  │
              │   Port: 3000        │
              └─────────────────────┘
```

### Service Communication Flow

1. **Data Ingestion**: Vital Simulator generates physiological data (1Hz) → Kafka topic `vitals_raw`
2. **Feature Engineering**: Pathway Engine consumes raw vitals, computes derived features → Kafka topic `vitals_enriched`
3. **RAG Indexing**: Pathway Engine simultaneously indexes enriched data for semantic search (sentence-transformers embeddings)
4. **Risk Prediction**: ML Service consumes enriched vitals, runs LSTM inference → Kafka topic `vitals_predictions`
5. **Data Aggregation**: Backend API subscribes to all Kafka topics, merges streams, caches latest patient state
6. **Real-time Distribution**: Backend API broadcasts merged data to frontend via WebSocket
7. **AI Chat Interface**: Frontend queries Backend API → Backend API queries Pathway RAG → Gemini LLM generates contextual responses
8. **Alert Generation**: Alert System monitors predictions, uses LangChain to generate intelligent notifications
9. **Report Generation**: Backend API generates PDF reports with matplotlib charts and AI summaries on-demand

---

## Technology Stack

### Backend Services

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend API | FastAPI 0.104, Python 3.10 | REST API, WebSocket server, orchestration layer |
| ML Service | PyTorch 2.2, LSTM | Sepsis risk prediction trained on MIMIC-IV |
| Pathway Engine | Pathway 0.8+ | Real-time stream processing, feature engineering, RAG |
| Alert System | LangChain 0.1, Gemini 2.5 | AI-powered emergency notifications |
| Vital Simulator | Python 3.10, NumPy | Realistic physiological data generation |
| Message Broker | Apache Kafka 7.4, Zookeeper | Event streaming, decoupled communication |

### Frontend

| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 18.2 | UI framework |
| TypeScript | 5.3 | Type-safe development |
| Vite | 5.0 | Build tool and dev server |
| Tailwind CSS | 3.4 | Utility-first styling |
| Framer Motion | 10.16 | Animations |
| Recharts | 2.10 | Time-series visualizations |
| Lucide React | 0.303 | Icon library |

### Infrastructure

- **Container Orchestration**: Docker Compose
- **Deployment**: Multi-container Docker environment with health checks
- **Networking**: Internal bridge network for service discovery
- **Storage**: Ephemeral (stateless services), Kafka retains 7-day message log

---

## Component Overview

### 1. Vital Simulator

**Purpose**: Generates MIMIC-IV-realistic physiological telemetry for 8 ICU patients

**Key Features**:
- Physiological drift model with mean-reversion dynamics
- State-based transitions (STABLE → EARLY_DETERIORATION → LATE_DETERIORATION → CRITICAL)
- Acute event engine (sepsis spikes, hypoxia events, medication responses)
- 15 vital parameters: HR, SBP, DBP, MAP, SpO2, RR, Temp, EtCO2, Lactate, WBC, Creatinine, Platelets, Bilirubin, Age, Gender
- Configurable update frequency (default 1Hz)

**Output**: Kafka topic `vitals_raw`

**Docker Image**: Built from `icu-system/vital-simulator/Dockerfile`

**Environment Variables**:
```bash
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
NUM_PATIENTS=8
UPDATE_INTERVAL_SECONDS=1
```

---

### 2. Pathway Streaming Engine

**Purpose**: Real-time feature engineering and semantic RAG indexing

Pathway is a Python framework for high-throughput, low-latency data processing on data streams. Unlike traditional batch processing frameworks, Pathway provides incremental computation where outputs update automatically as new data arrives, making it ideal for real-time ICU monitoring.

**Why Pathway for VitalX**:

- **Incremental Computation**: Automatically recomputes only affected records when new data arrives, minimizing latency
- **Stateful Stream Processing**: Maintains windowed aggregates (rolling averages, trend analysis) without external state management
- **Native Kafka Integration**: Direct Kafka topic consumption and production with schema validation
- **Declarative API**: SQL-like transformations (groupby, join, window operations) on streaming data
- **Real-time RAG**: Built-in vector indexing and semantic search via `pathway.xpacks.llm`

**VitalX Pathway Pipeline Architecture**:

```python
# Pseudo-code representation of VitalX Pathway pipeline

# 1. Input Stream - Consume raw vitals from Kafka
input_stream = pw.io.kafka.read(
    topic="vitals_raw",
    schema=VitalSignsSchema,
    bootstrap_servers="kafka:9092"
)

# 2. Feature Engineering - Compute derived clinical indicators
enriched_stream = input_stream.select(
    # Original fields
    patient_id=pw.this.patient_id,
    timestamp=pw.this.timestamp,
    heart_rate=pw.this.heart_rate,
    systolic_bp=pw.this.systolic_bp,
    # ... other vitals ...
    
    # Derived features
    shock_index=pw.this.heart_rate / pw.this.systolic_bp,
    map=(pw.this.systolic_bp + 2 * pw.this.diastolic_bp) / 3,
    pulse_pressure=pw.this.systolic_bp - pw.this.diastolic_bp
)

# 3. Windowed Aggregations - Rolling statistics per patient
rolling_features = enriched_stream.windowby(
    window=pw.temporal.sliding(duration=300),  # 5-minute window
    key=pw.this.patient_id
).reduce(
    rolling_hr_mean=pw.reducers.avg(pw.this.heart_rate),
    rolling_hr_std=pw.reducers.std(pw.this.heart_rate),
    rolling_sbp_mean=pw.reducers.avg(pw.this.systolic_bp)
)

# 4. Anomaly Detection - Statistical outlier identification
anomaly_stream = enriched_stream.join(rolling_features).select(
    anomaly_flag=(
        (pw.this.heart_rate > pw.this.rolling_hr_mean + 2 * pw.this.rolling_hr_std) |
        (pw.this.systolic_bp < pw.this.rolling_sbp_mean - 2 * pw.this.rolling_sbp_std)
    )
)

# 5. RAG Indexing - Semantic embeddings for AI chat
pw.xpacks.llm.vector_store(
    data=enriched_stream,
    embedder=sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2"),
    index_column="patient_summary_text",
    metadata_columns=["patient_id", "timestamp", "heart_rate", "computed_risk"]
)

# 6. Output Sink - Publish enriched data to Kafka
pw.io.kafka.write(
    table=enriched_stream,
    topic="vitals_enriched",
    bootstrap_servers="kafka:9092"
)

# 7. Query API - REST endpoint for RAG semantic search (port 8080)
pw.io.http.rest_connector(
    host="0.0.0.0",
    port=8080,
    queries={
        "/v1/pw_ai_answer": query_answer,  # RAG endpoint for backend
        "/v1/pw_list_documents": list_documents
    }
)
```

**Pathway vs Traditional Streaming Frameworks**:

| Feature | Pathway | Apache Flink | Kafka Streams |
|---------|---------|--------------|---------------|
| Incremental Computation | Native | Manual state mgmt | Manual state mgmt |
| Python-first API | Yes | No (Java/Scala) | No (Java/Kotlin) |
| Built-in RAG/Vector Search | Yes | No | No |
| Latency | Sub-second | ~1-5 seconds | ~1-5 seconds |
| Window Operations | Declarative SQL-like | Imperative | Imperative |
| LLM Integration | Native | Manual | Manual |

**VitalX Pathway Configuration**:

```yaml
# icu-system/pathway-engine/app/settings.py
kafka:
  bootstrap_servers: kafka:9092
  input_topic: vitals_raw
  output_topic: vitals_enriched
  consumer_group: pathway-engine-group

risk_engine:
  high_risk_threshold: 0.85
  anomaly_z_score: 2.0
  rolling_window_seconds: 300

rag:
  embedding_model: all-MiniLM-L6-v2
  max_history_hours: 3
  query_api_port: 8080
```

**Key Pathway Operations in VitalX**:

1. **Real-time Feature Engineering**: Computes clinical indicators (shock index, pulse pressure, oxygen delivery index) on each incoming vital sign record with <10ms latency

2. **Temporal Windows**: Maintains 5-minute sliding windows per patient for rolling averages and standard deviations, essential for anomaly detection

3. **Patient-Isolated RAG**: Indexes enriched vitals with patient_id-scoped embeddings, ensuring RAG queries only retrieve context from the specific patient being discussed

4. **Automatic Recomputation**: When a late-arriving vital sign arrives (e.g., delayed lab result), Pathway automatically recomputes all downstream aggregates and RAG embeddings without manual trigger

5. **Query API for RAG**: Exposes HTTP endpoint at `:8080/v1/pw_ai_answer` that accepts natural language queries and returns semantically similar patient records from the last 3 hours

**Docker Image**: Built from `icu-system/pathway-engine/Dockerfile`

**Exposed Ports**:
- `8080`: Query API for RAG (Backend API calls this endpoint for AI chat)

**Environment Variables**:
```bash
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
QUERY_API_PORT=8080
```

---

### 3. ML Service (LSTM Risk Prediction)

**Purpose**: Real-time sepsis deterioration risk scoring using trained LSTM neural network

**Model Architecture**:
- **Type**: LSTM with Attention mechanism
- **Input**: 24-hour sequence of 34 clinical features
- **Output**: Sepsis risk probability (0.0 - 1.0)
- **Training Dataset**: MIMIC-IV ICU patient records (deidentified)
- **Training Samples**: 50,000+ patient-hours
- **Validation AUROC**: 0.89
- **Framework**: PyTorch 2.2

**Model Details**:
```python
# VitalX ml/model.py
class SepsisLSTM(nn.Module):
    - Input layer: 34 features (vitals + labs + demographics)
    - LSTM layers: 2 layers, 128 hidden units, bidirectional
    - Attention layer: Weighted temporal pooling
    - Dropout: 0.3 (training only)
    - Output layer: Sigmoid activation for binary probability
    - Loss function: Binary Cross-Entropy with class weighting
    - Optimizer: Adam (lr=0.001)
```

**34 Clinical Features** (see Feature Engineering section for full list):
- Primary vitals (8): HR, SBP, DBP, MAP, SpO2, RR, Temp, EtCO2
- Lab values (5): Lactate, WBC, Creatinine, Platelets, Bilirubin
- Derived features (19): Shock index, rolling averages, trend indicators, anomaly flags
- Demographics (2): Age, Gender

**Inference Pipeline**:
1. Consumes enriched vitals from Kafka topic `vitals_enriched`
2. Maintains 24-hour rolling buffer per patient (stored in-memory)
3. Runs LSTM forward pass when buffer contains minimum 12 hours of data
4. Publishes prediction to Kafka topic `vitals_predictions`
5. Updates prediction every 2 seconds (configurable)

**Docker Image**: Built from `icu-system/ml-service/Dockerfile`

**Environment Variables**:
```bash
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
CONSUME_TOPIC=vitals_enriched
PUBLISH_TOPIC=vitals_predictions
MODEL_PATH=/app/models/lstm_sepsis_model.pth
MIN_SEQUENCE_HOURS=12
```

---

### 4. Backend API (Orchestration Layer)

**Purpose**: FastAPI-based REST + WebSocket gateway for frontend and service coordination

**Key Responsibilities**:
- Stream aggregation: Merges Kafka topics (raw, enriched, predictions) into unified patient state
- WebSocket broadcasting: Pushes real-time updates to connected frontend clients
- REST API: Exposes endpoints for patient queries, reports, RAG chat
- RAG proxy: Routes chat queries to Pathway Engine query API
- PDF generation: On-demand medical report creation with matplotlib charts
- CORS handling: Enables cross-origin requests from frontend

**API Endpoints**:

**WebSocket**:
- `ws://localhost:8000/ws` - Real-time patient data stream (broadcasts floor updates every 2s)

**REST API**:

```
GET  /api/patients                    - List all patients with latest vitals
GET  /api/patients/{patient_id}       - Detailed patient information
GET  /api/patients/{patient_id}/history?hours=3  - Historical vitals (default 3 hours)
GET  /api/patients/{patient_id}/risk-history?hours=1  - Risk score history
POST /api/patients/{patient_id}/reports/generate  - Generate PDF report
     Body: {"time_range_hours": 6, "include_ai_summary": true}

POST /api/chat  - RAG-powered Q&A about patient
     Body: {"patient_id": "P001", "question": "Why is shock index elevated?"}

GET  /api/floors                      - List ICU floors
GET  /api/floors/{floor_id}/patients  - Patients on specific floor

GET  /api/stats/overview              - System-wide statistics (avg risk, alerts, capacity)
GET  /api/health                      - Service health check
```

**Stream Merger Logic**:

The Backend API uses a custom `StreamMerger` class that:
1. Creates 3 Kafka consumers (one per topic: raw, enriched, predictions)
2. Maintains in-memory patient state cache (dict keyed by patient_id)
3. Listens to all topics concurrently using asyncio
4. Merges updates by patient_id with timestamp-based conflict resolution
5. Triggers WebSocket broadcast on state changes

**Docker Image**: Built from `icu-system/backend-api/Dockerfile`

**Exposed Ports**:
- `8000`: HTTP (REST API) + WebSocket

**Environment Variables**:
```bash
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
PATHWAY_RAG_URL=http://pathway-engine:8080
GEMINI_API_KEY=<your-key>
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash-lite
```

---

### 5. Alert System (LangChain AI Notifications)

**Purpose**: Intelligent emergency notification system with natural language explanations

**Key Features**:
- LangChain agent architecture with Gemini 2.5 Flash LLM
- Contextual alert messages: "Patient P003 risk increased to 0.91 due to dropping SpO2 (88%) and elevated lactate (3.2 mmol/L)"
- Rate limiting: Maximum 1 alert per patient per 5 minutes
- Multi-channel delivery: Console logs, email (SMTP), extensible to Slack/SMS
- Severity classification: INFO, WARNING, CRITICAL
- Actionable recommendations generated by LLM

**LangChain Alert Pipeline**:

```python
# Simplified LangChain agent workflow
class MedicalAlertAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        # System prompt engineering
        self.system_prompt = """
        You are an ICU medical alert system AI assistant.
        Analyze patient vital signs and risk predictions to generate 
        clear, actionable clinical alerts for nursing staff.
        
        Format:
        - Severity: [CRITICAL/WARNING/INFO]
        - Patient: [ID]
        - Primary Concern: [Single sentence]
        - Vital Trends: [Key changes in last 10 minutes]
        - Recommended Actions: [Numbered list of 2-3 actions]
        """
        
    def generate_alert(self, patient_data: dict) -> str:
        chain = LLMChain(llm=self.llm, prompt=self.system_prompt)
        return chain.run(patient_data)
```

**Alert Trigger Conditions**:
- Risk score > 0.85 (high-risk threshold)
- Risk score delta > 0.15 in last 5 minutes (rapid deterioration)
- SpO2 < 90% (hypoxia)
- Shock index > 1.0 (circulatory insufficiency)

**Docker Image**: Built from `icu-system/alert-system/Dockerfile`

**Environment Variables**:
```bash
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=<your-key>
ENABLE_CONSOLE_ALERTS=true
ENABLE_EMAIL_ALERTS=false
HIGH_RISK_THRESHOLD=0.85
```

---

### 6. Frontend (React TypeScript Dashboard)

**Purpose**: Real-time clinical dashboard for ICU staff

**Key Features**:
- Role-based views: Nurse Dashboard (floor-specific), Doctor Dashboard (multi-patient overview)
- Live patient cards with color-coded risk indicators
- Real-time trend charts: Risk score, HR, BP, SpO2, RR (Recharts library)
- Patient detail drawer with 24-hour vital history
- RAG-powered AI chat interface for clinical questions
- Shift handoff modal with AI-generated handoff notes
- PDF report export (downloads from Backend API)
- Floor selector (ICU-1, ICU-2, ICU-3)
- Sorting controls (by risk, name, room number)

**WebSocket Client**:
- Reconnects automatically on disconnect
- Subscribes to specific patients or entire floors
- Updates UI state at 60fps using React hooks (useState, useEffect)
- Rate-limited re-renders to prevent UI jank

**Tech Highlights**:
- Framer Motion for smooth card animations and transitions
- Lucide React for medical icons (heart, activity, alert triangles)
- Tailwind CSS for responsive design (mobile-friendly)
- TypeScript for type-safe API calls and WebSocket message handling

**Docker Image**: Built from `frontend/Dockerfile`
- Multi-stage build: npm build → nginx static file serving

**Exposed Ports**:
- `3000`: HTTP (mapped to nginx port 80 inside container)

**Environment Variables** (Vite build-time):
```bash
VITE_API_URL=http://localhost:8000  # Backend API base URL
```

---

## Data Flow Pipeline

### End-to-End Request Flow Examples

#### Example 1: Real-time Vital Sign Update

```
1. Vital Simulator generates HR=95 for P001 at T=0ms
2. Simulator publishes to Kafka vitals_raw topic at T=5ms
3. Pathway Engine consumes message at T=10ms
4. Pathway computes shock_index, rolling_hr_mean at T=12ms
5. Pathway publishes to vitals_enriched topic at T=15ms
6. Pathway indexes data for RAG at T=18ms
7. ML Service consumes enriched data at T=20ms
8. ML Service runs LSTM inference at T=50ms (30ms model latency)
9. ML Service publishes risk=0.73 to vitals_predictions at T=55ms
10. Backend API receives both messages at T=60ms
11. Backend API merges data, updates cache at T=62ms
12. Backend API broadcasts via WebSocket at T=65ms
13. Frontend receives WebSocket message at T=70ms
14. Frontend updates UI (React re-render) at T=80ms

Total latency: 80ms (vital generated → UI updated)
```

#### Example 2: AI Chat Query

```
1. User types "Why is SpO2 dropping?" in frontend at T=0ms
2. Frontend sends POST /api/chat to Backend API at T=10ms
3. Backend API queries Pathway RAG at http://pathway-engine:8080/v1/pw_ai_answer at T=15ms
4. Pathway RAG performs semantic search over 3-hour patient history at T=20ms
5. Pathway returns top 5 relevant vital records at T=100ms
6. Backend API formats context and calls Gemini LLM API at T=110ms
7. Gemini generates response in 800ms (LLM latency)
8. Backend API returns formatted answer to frontend at T=920ms
9. Frontend displays answer in chat UI at T=930ms

Total latency: 930ms (user question → AI answer displayed)
```

---

## Machine Learning Model

### Training Details

**Dataset**: MIMIC-IV Clinical Database v2.0
- Source: PhysioNet (requires credentialed access)
- Patient population: ICU admissions 2008-2019 at BIDMC
- Total patient records: 50,000+ ICU stays
- Sampling rate: 1-hour granularity
- Label: Sepsis-3 criteria (qSOFA ≥2 + suspected infection)

**Data Preprocessing** (`VitalX ml/dataset.py`):
```python
# Feature normalization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(raw_features)

# Sequence padding/truncation to 24 hours
sequence_length = 24  # hours
padded_sequences = pad_sequences(patient_sequences, maxlen=sequence_length)

# Train/validation/test split: 70/15/15
```

**Model Hyperparameters** (`VitalX ml/config.py`):
```python
SEQUENCE_LENGTH = 24  # hours
NUM_FEATURES = 34
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50
```

**Training Results**:
- Training time: 6 hours on NVIDIA RTX 3090
- Best validation AUROC: 0.89
- Precision at 0.8 recall: 0.76
- Early stopping: Patience 10 epochs on validation loss

**Model Files** (included in `VitalX ml/outputs/`):
- `lstm_sepsis_model.pth` - Trained PyTorch model weights (25 MB)
- `scaler.pkl` - Feature normalization parameters
- `training_metrics.json` - Epoch-by-epoch training logs

**Inference Optimization**:
- Model loaded at ML Service startup
- Cached in GPU memory (if available) or CPU
- Batch inference for multiple patients (batch size 8)
- Average inference time: 30ms per patient on CPU

---

## Installation & Deployment

### Prerequisites

- **Docker Desktop**: Version 4.0+ (Windows/Mac) or Docker Engine 20.10+ (Linux)
- **Docker Compose**: Version 2.0+ (included in Docker Desktop)
- **Hardware**: Minimum 8GB RAM, 4 CPU cores recommended for all services
- **Network**: Ports 3000, 8000, 8001, 8080, 29092 available

### Quick Start (Docker Compose)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd HackGDG
```

2. **Set environment variables** (optional):
```bash
# Copy and edit .env file
cp .env.example .env

# Edit .env with your API keys (required for AI features)
# GEMINI_API_KEY=your-google-gemini-api-key
```

3. **Build all Docker images**:
```bash
cd icu-system
docker compose build
```
Expected build time: 10-15 minutes (downloads dependencies, installs Python packages)

4. **Start all services**:
```bash
docker compose up -d
```

5. **Verify deployment**:
```bash
# Check all containers are running
docker compose ps

# Expected output: 8 containers in "Up" state
# - icu-zookeeper
# - icu-kafka
# - icu-vital-simulator
# - icu-pathway-engine
# - icu-ml-service
# - icu-backend-api
# - icu-alert-system
# - icu-frontend

# View logs
docker compose logs -f backend-api
docker compose logs -f pathway-engine
```

6. **Access the application**:
- Frontend Dashboard: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs (Swagger UI)
- ML Service Health: http://localhost:8001/health
- Pathway RAG Query API: http://localhost:8080

### Step-by-Step Service Launch Order

Docker Compose handles service dependencies automatically, but for understanding:

```
1. Zookeeper (dependency for Kafka)
   └─▶ Health check: Port 2181 accessible
   
2. Kafka (message broker)
   └─▶ Health check: Port 9092 accessible
   
3. Vital Simulator + Pathway Engine + ML Service (data producers/consumers)
   └─▶ Wait for Kafka health check
   
4. Backend API (aggregation layer)
   └─▶ Wait for Kafka + Pathway Engine
   
5. Alert System (monitoring)
   └─▶ Wait for Kafka + ML Service
   
6. Frontend (user interface)
   └─▶ Wait for Backend API
```

### Configuration Files

**Main Configuration**: `icu-system/docker-compose.yml`
- Service definitions
- Port mappings
- Environment variables
- Health checks
- Network configuration

**Frontend Build Config**: `frontend/vite.config.ts`
- Proxy configuration for Backend API
- Build output directory
- Development server settings

**Backend API Config**: `icu-system/backend-api/app/config.py`
- Kafka connection parameters
- WebSocket settings
- CORS origins

**ML Service Config**: `icu-system/ml-service/config.py`
- Model hyperparameters
- Inference batch size
- Feature list

**Pathway Engine Config**: `icu-system/pathway-engine/app/settings.py`
- Risk calculation weights
- RAG embedding model
- Query API settings

---

## API Documentation

### REST API Endpoints (Backend API)

Full interactive documentation available at http://localhost:8000/docs (Swagger UI)

#### Patient Endpoints

**GET /api/patients**

List all active ICU patients with latest vitals and risk scores.

Response (200 OK):
```json
{
  "patients": [
    {
      "patient_id": "P001",
      "floor_id": "ICU-1",
      "timestamp": "2026-02-26T14:35:00.123Z",
      "heart_rate": 78.5,
      "systolic_bp": 118.2,
      "diastolic_bp": 72.1,
      "spo2": 97.8,
      "respiratory_rate": 14.2,
      "temperature": 36.9,
      "shock_index": 0.665,
      "computed_risk": 0.68,
      "anomaly_flag": false,
      "state": "STABLE"
    }
  ],
  "total": 8,
  "timestamp": "2026-02-26T14:35:00.500Z"
}
```

**GET /api/patients/{patient_id}**

Retrieve detailed information for a specific patient.

Response (200 OK):
```json
{
  "patient_id": "P001",
  "demographics": {
    "age": 68,
    "gender": "M"
  },
  "current_state": {
    "timestamp": "2026-02-26T14:35:00.123Z",
    "vitals": { ... },
    "labs": { ... },
    "computed_risk": 0.68
  },
  "summary_stats": {
    "avg_risk_24h": 0.65,
    "max_risk_24h": 0.82,
    "alert_count_24h": 2
  }
}
```

**GET /api/patients/{patient_id}/history?hours=3**

Retrieve historical vital signs for the past N hours (default 3).

Query Parameters:
- `hours` (optional, integer): Time range in hours (default 3, max 24)

Response (200 OK):
```json
{
  "patient_id": "P001",
  "history": [
    {
      "timestamp": "2026-02-26T11:35:00.000Z",
      "heart_rate": 76.2,
      "systolic_bp": 120.5,
      "computed_risk": 0.62
    },
    // ... more records
  ],
  "count": 10800,  // 3 hours * 3600 seconds
  "start_time": "2026-02-26T11:35:00.000Z",
  "end_time": "2026-02-26T14:35:00.000Z"
}
```

**GET /api/patients/{patient_id}/risk-history?hours=1**

Retrieve risk score history specifically.

Response (200 OK):
```json
{
  "patient_id": "P001",
  "risk_history": [
    {
      "timestamp": "2026-02-26T13:35:00.000Z",
      "risk_score": 0.62,
      "prediction_confidence": 0.89
    }
  ],
  "count": 1800
}
```

**POST /api/patients/{patient_id}/reports/generate**

Generate a PDF medical report with time-series visualizations and optional AI summary.

Request Body:
```json
{
  "time_range_hours": 6,
  "include_ai_summary": true
}
```

Response (200 OK):
- Content-Type: `application/pdf`
- Body: Binary PDF file
- Suggested filename in `Content-Disposition` header

#### RAG Chat Endpoint

**POST /api/chat**

Ask natural language questions about a patient, answered using RAG over historical data.

Request Body:
```json
{
  "patient_id": "P001",
  "question": "Why is the patient's shock index increasing over the last hour?"
}
```

Response (200 OK):
```json
{
  "patient_id": "P001",
  "question": "Why is the patient's shock index increasing over the last hour?",
  "answer": "The patient's shock index has increased from 0.62 to 0.78 over the last hour primarily due to a rise in heart rate (72 → 85 bpm) combined with a slight drop in systolic blood pressure (125 → 108 mmHg). This pattern suggests developing circulatory insufficiency. The concurrent SpO2 decline to 91% and elevated lactate (2.8 mmol/L) indicate possible tissue hypoperfusion. Recommend fluid resuscitation and vasopressor evaluation.",
  "sources": [
    {
      "timestamp": "2026-02-26T14:00:00.000Z",
      "heart_rate": 85,
      "systolic_bp": 108,
      "shock_index": 0.78
    }
  ],
  "model": "gemini-2.5-flash-lite",
  "tokens_used": 245
}
```

#### Floor and Statistics Endpoints

**GET /api/floors**

List all ICU floors.

Response (200 OK):
```json
{
  "floors": [
    {
      "id": "ICU-1",
      "name": "Intensive Care Unit - Floor 1",
      "capacity": 8,
      "current_patients": 5,
      "available_beds": 3,
      "avg_risk_score": 0.58
    }
  ]
}
```

**GET /api/stats/overview**

System-wide statistics for dashboard overview cards.

Response (200 OK):
```json
{
  "total_patients": 8,
  "critical_patients": 2,
  "avg_risk_score": 0.64,
  "active_alerts": 3,
  "bed_occupancy": 0.625,
  "timestamp": "2026-02-26T14:35:00.000Z"
}
```

### WebSocket API

**Endpoint**: `ws://localhost:8000/ws`

Connection established via WebSocket handshake. Backend pushes updates every 2 seconds.

**Message Format** (JSON):
```json
{
  "type": "floor_update",
  "floor_id": "ICU-1",
  "patients": [ /* array of patient objects */ ],
  "timestamp": "2026-02-26T14:35:02.123Z"
}
```

**Client-side Subscription** (optional feature):
```json
{
  "action": "subscribe_patient",
  "patient_id": "P001"
}
```

---

## Configuration

### Environment Variables

All environment variables can be set in `icu-system/.env` file or passed directly to docker compose.

#### Global Variables

```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092

# LLM API Keys (required for AI features)
GEMINI_API_KEY=<your-google-gemini-api-key>
OPENAI_API_KEY=<optional-for-alternative-llm>

# LLM Provider Selection
LLM_PROVIDER=gemini  # Options: gemini, openai
GEMINI_MODEL=gemini-2.5-flash-lite
```

#### Service-Specific Variables

**Vital Simulator**:
```bash
NUM_PATIENTS=8
UPDATE_INTERVAL_SECONDS=1
DEBUG_MODE=false
```

**ML Service**:
```bash
CONSUME_TOPIC=vitals_enriched
PUBLISH_TOPIC=vitals_predictions
MODEL_PATH=/app/models/lstm_sepsis_model.pth
MIN_SEQUENCE_HOURS=12
INFERENCE_BATCH_SIZE=8
```

**Backend API**:
```bash
PATHWAY_RAG_URL=http://pathway-engine:8080
CORS_ORIGINS=["http://localhost:3000"]
WEBSOCKET_BROADCAST_INTERVAL=2  # seconds
```

**Alert System**:
```bash
HIGH_RISK_THRESHOLD=0.85
ENABLE_CONSOLE_ALERTS=true
ENABLE_EMAIL_ALERTS=false

# Email Configuration (if ENABLE_EMAIL_ALERTS=true)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=<your-email>
SMTP_PASSWORD=<your-app-password>
ALERT_EMAIL_FROM=icu-alerts@hospital.com
ALERT_EMAIL_TO=doctor@hospital.com
```

**Pathway Engine**:
```bash
QUERY_API_PORT=8080
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_MAX_HISTORY_HOURS=3
```

---

## Monitoring & Operations

### Health Checks

All services expose health check endpoints:

```bash
# Backend API
curl http://localhost:8000/health
# Response: {"status": "healthy", "kafka": "connected", "pathway": "reachable"}

# ML Service
curl http://localhost:8001/health
# Response: {"status": "healthy", "model": "loaded", "kafka": "connected"}

# Docker health status
docker compose ps
```

### Logging

View logs for specific services:

```bash
# Real-time logs
docker compose logs -f backend-api
docker compose logs -f pathway-engine
docker compose logs -f ml-service

# Last 100 lines
docker compose logs --tail=100 alert-system

# All services
docker compose logs -f
```

### Kafka Topic Monitoring

Inspect Kafka topics and messages:

```bash
# List all topics
docker exec -it icu-kafka kafka-topics \
  --bootstrap-server localhost:29092 --list

# View messages in vitals_raw topic
docker exec -it icu-kafka kafka-console-consumer \
  --bootstrap-server localhost:29092 \
  --topic vitals_raw \
  --from-beginning \
  --max-messages 10

# Check topic lag
docker exec -it icu-kafka kafka-consumer-groups \
  --bootstrap-server localhost:29092 \
  --describe --group ml-service-group
```

### Performance Metrics

**Expected Latencies**:
- Vital Simulator → Kafka: 5ms
- Kafka → Pathway Engine: 5ms
- Pathway feature engineering: 2ms
- Pathway RAG indexing: 3ms
- ML LSTM inference: 30ms
- Backend WebSocket broadcast: 10ms
- End-to-end (data generated → UI updated): 80ms

**Throughput**:
- Vital Simulator: 8 patients × 1Hz = 8 messages/second
- Kafka: Tested up to 10,000 msg/s (over-provisioned for future scalability)
- ML Service: 30 inferences/second (batch size 8)
- Backend WebSocket: 50 concurrent clients tested

### Troubleshooting

**Issue**: Containers fail to start

Solution:
```bash
# Check Docker resources (ensure 8GB RAM allocated)
docker info

# Rebuild from scratch
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

**Issue**: Kafka connection errors

Solution:
```bash
# Verify Kafka health
docker compose logs kafka | grep "started (kafka.server.KafkaServer)"

# Restart Kafka stack
docker compose restart zookeeper kafka
```

**Issue**: ML predictions not appearing

Solution:
```bash
# Check ML service is consuming enriched vitals
docker compose logs ml-service | grep "Consumed message"

# Verify model loaded successfully
docker compose logs ml-service | grep "Model loaded"

# Check Pathway is publishing enriched data
docker compose logs pathway-engine | grep "Published to vitals_enriched"
```

**Issue**: RAG chat returns no context

Solution:
```bash
# Verify Pathway query API is accessible
curl http://localhost:8080/v1/pw_list_documents

# Check RAG indexing logs
docker compose logs pathway-engine | grep "Indexed document"

# Ensure patient_id in chat request matches live patients
curl http://localhost:8000/api/patients
```

---

## Project Structure

```
HackGDG/
├── icu-system/                    # Backend services
│   ├── docker-compose.yml         # Multi-container orchestration
│   ├── vital-simulator/           # Physiological data generator
│   │   ├── app/
│   │   │   └── main.py            # Patient state machine
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── pathway-engine/            # Streaming processor + RAG
│   │   ├── app/
│   │   │   ├── main.py            # Pathway pipeline
│   │   │   ├── risk_engine.py     # Feature engineering
│   │   │   └── settings.py        # Configuration
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── ml-service/                # LSTM inference
│   │   ├── app/
│   │   │   ├── main.py            # Kafka consumer + inference loop
│   │   │   └── inference.py       # Model wrapper
│   │   ├── models/
│   │   │   └── lstm_sepsis_model.pth
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── backend-api/               # FastAPI orchestration layer
│   │   ├── app/
│   │   │   ├── main_new.py        # API routes + WebSocket
│   │   │   ├── stream_merger.py   # Kafka aggregation
│   │   │   ├── rag_chat_agent.py  # RAG proxy
│   │   │   └── pdf_report_generator.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── alert-system/              # LangChain alerts
│   │   ├── app/
│   │   │   ├── main.py            # Alert orchestrator
│   │   │   ├── langchain_agent.py # LLM integration
│   │   │   └── notification_service.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── alert-engine/              # Threshold-based alerts
│       ├── app/
│       │   └── main.py
│       ├── Dockerfile
│       └── requirements.txt
├── frontend/                      # React TypeScript UI
│   ├── src/
│   │   ├── pages/
│   │   │   ├── NurseDashboard.tsx # Floor-specific view
│   │   │   └── DoctorDashboard.tsx # Multi-patient view
│   │   ├── components/
│   │   │   ├── PatientCard.tsx
│   │   │   ├── RiskTrendChart.tsx
│   │   │   └── ChatInterface.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts    # Real-time data
│   │   │   └── usePatients.ts     # State management
│   │   └── services/
│   │       └── api.ts             # Backend API client
│   ├── Dockerfile                 # Multi-stage build (npm + nginx)
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── VitalX ml/                     # Model training code
│   ├── model.py                   # LSTM architecture
│   ├── dataset.py                 # MIMIC-IV preprocessing
│   ├── train.py                   # Training loop
│   ├── inference.py               # Production inference wrapper
│   ├── config.py                  # Hyperparameters
│   └── outputs/
│       ├── lstm_sepsis_model.pth  # Trained weights
│       └── scaler.pkl             # Normalization params
├── requirements.txt               # Unified Python dependencies
└── README_DETAILED.md             # This file
```

---

## License

This project is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

---

## Credits

**Team**: VitalX Development Team

**Dataset**: MIMIC-IV Clinical Database (Johnson et al., 2020)
- Citation required for any publications using this dataset
- Access requires PhysioNet credentialed training

**Technologies**:
- Pathway (Real-time stream processing framework)
- LangChain (LLM orchestration)
- PyTorch (Deep learning framework)
- FastAPI (Modern Python web framework)
- Apache Kafka (Distributed event streaming)
- React (UI library)

---

## Support

For technical support, deployment assistance, or questions, contact the development team.

**Important Notes**:
- Ensure all API keys (GEMINI_API_KEY) are set before starting services
- Allocate minimum 8GB RAM to Docker Desktop for stable operation
- Monitor Kafka lag if experiencing delays in data propagation
- Check health endpoints before reporting issues

---

End of Documentation
