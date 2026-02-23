# VitalX ICU Digital Twin — Streaming Architecture Refactor

**Architectural Transformation Plan**  
*From Rule-Based State Machine → Streaming ML-First Production System*

---

## 🎯 TRANSFORMATION OBJECTIVES

### Current Architecture (Problems)
- ❌ State-machine ICU simulation with hardcoded transitions
- ❌ Rule-based risk scoring (duplicated logic)
- ❌ Standalone ChromaDB RAG (batch indexing)
- ❌ Multiple risk authorities (Pathway + Backend)
- ❌ Unrealistic vital sign spikes
- ❌ Topic explosion issues

### Target Architecture (Solutions)
- ✅ Streaming-first data flow
- ✅ ML-only risk authority (single source of truth)
- ✅ Pathway-based live streaming RAG
- ✅ Clean separation of concerns
- ✅ Realistic physiological simulation
- ✅ Production-grade microservices

---

## 📊 FINAL ARCHITECTURE FLOW

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMING DATA FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ Vital Simulator  │  → Realistic physiological drift model
│  (No States)     │  → Gaussian noise only
└────────┬─────────┘  → Gradual deterioration
         │
         ↓ 
    Kafka Topic: vitals_raw
    Schema: {patient_id, timestamp, heart_rate, systolic_bp, 
             diastolic_bp, map, spo2, respiratory_rate, 
             temperature, lactate, shock_index}
         │
         ↓
┌────────┴──────────────────────────────────────────────┐
│           PATHWAY ENGINE (3 Functions)                 │
├────────────────────────────────────────────────────────┤
│                                                         │
│  A) FEATURE ENGINEERING                                │
│     • Sliding window (30-60 min)                       │
│     • Rolling statistics (mean_hr, mean_sbp, etc.)     │
│     • Deltas (hr_delta, sbp_delta, shock_delta)       │
│     • Anomaly flags (z-score based)                    │
│     • NO risk scoring                                  │
│     Output → Kafka: vitals_enriched                    │
│                                                         │
│  B) STREAMING VECTOR INDEX (Live RAG Memory)           │
│     • Convert enriched events → text chunks            │
│     • Embed using sentence-transformers                │
│     • Maintain sliding-window index (1-3 hrs)          │
│     • Auto-expire old embeddings                       │
│     • Per-patient isolation                            │
│                                                         │
│  C) QUERY INTERFACE                                    │
│     • Expose HTTP endpoint                             │
│     • Input: patient_id + query_text                   │
│     • Process: embed query → retrieve top-k            │
│     • Output: structured context (no LLM)              │
│                                                         │
└────────┬───────────────────────────────────┬──────────┘
         │                                   │
         ↓                                   │
    Kafka Topic: vitals_enriched            │ Query API
    Schema: {patient_id, timestamp,         │ (HTTP)
             + rolling_mean_hr,              │
             + hr_delta,                     │
             + shock_index_delta,            │
             + lactate_trend,                │
             + anomaly_flag}                 │
         │                                   │
         ↓                                   │
┌────────┴─────────────┐                    │
│    ML SERVICE        │                    │
│  (Placeholder Mode)  │                    │
├──────────────────────┤                    │
│ • Consume enriched   │                    │
│ • Sequence buffer    │                    │
│ • predict(seq) →     │                    │
│   risk_score         │                    │
│ • Publish to Kafka   │                    │
└────────┬─────────────┘                    │
         │                                   │
         ↓                                   │
    Kafka Topic: vitals_predictions         │
    Schema: {patient_id, timestamp,         │
             risk_score}                     │
         │                                   │
         ↓                                   │
┌────────┴─────────────────────────────────┴┐
│          BACKEND API (Orchestrator)       │
├───────────────────────────────────────────┤
│ 1. Merge vitals_enriched + predictions    │
│ 2. WebSocket /ws → stream unified view    │
│ 3. REST endpoints                         │
│    • GET /patients                        │
│    • GET /patients/{id}/latest            │
│    • GET /patients/{id}/history           │
│ 4. Chat endpoint                          │
│    • POST /chat                           │
│    • Call Pathway query API               │
│    • Pass context → LLM                   │
│    • Return grounded response             │
└────────┬──────────────────────────────────┘
         │
         ↓
┌────────┴─────────────┐
│   React Frontend     │
│ • Live vitals        │
│ • Risk score display │
│ • Risk trend chart   │
│ • Alert banner       │
│ • RAG chat panel     │
└──────────────────────┘

         ↑
         │ (parallel)
┌────────┴─────────────┐
│   Alert Engine       │
│ • Watch predictions  │
│ • Trigger: risk >    │
│   threshold          │
│ • Generate context   │
│   from enriched data │
└──────────────────────┘
```

---

## 🔧 SERVICE RESPONSIBILITIES BREAKDOWN

### 1. VITAL SIMULATOR

**Role:** Realistic Physiological Data Generator

**Responsibilities:**
- Generate baseline physiological parameters per patient
- Apply gradual drift model:
  - Heart rate drift (±0.1-0.5 bpm/min)
  - Blood pressure drift (±0.05-0.3 mmHg/min)
  - Lactate accumulation (gradual increase)
  - Shock index calculation (HR/SBP)
- Add Gaussian noise only (no spikes)
- Probabilistic deterioration trigger (5-10% chance/interval)
- Publish to Kafka `vitals_raw`

**What to REMOVE:**
- `PatientState` enum (STABLE, EARLY, CRITICAL, etc.)
- State transition logic
- Hardcoded deterioration labels
- Acute event system (SEPSIS_SPIKE, etc.)
- Scripted spike profiles
- Emoji logs

**What to KEEP:**
- Kafka producer
- Individual patient baselines
- JSON serialization

**Output Schema:**
```json
{
  "patient_id": "P001",
  "timestamp": "2026-02-23T14:32:00Z",
  "heart_rate": 102.3,
  "systolic_bp": 92.1,
  "diastolic_bp": 61.4,
  "map": 71.6,
  "spo2": 94.2,
  "respiratory_rate": 22.1,
  "temperature": 37.8,
  "lactate": 2.3,
  "shock_index": 1.11
}
```

---

### 2. KAFKA INFRASTRUCTURE

**Role:** Event Streaming Backbone

**Topics:**
```
vitals_raw           → Raw physiological data
vitals_enriched      → Feature-engineered data
vitals_predictions   → ML risk scores
alerts_stream        → Alert events
```

**Configuration Requirements:**
- **Idempotent producers:** `enable.idempotence=true`
- **Controlled throughput:** 1 msg/sec per patient
- **Retention policy:** 7 days
- **Replication factor:** 1 (dev), 3 (prod)
- **Consumer groups:** Properly configured
- **Auto-create topics:** Enabled

**What to FIX:**
- Ensure no topic explosion (linear growth only)
- Add monitoring for consumer lag
- Configure proper cleanup policies

---

### 3. PATHWAY ENGINE (CORE REFACTOR)

**Role:** Streaming Feature Engineering + Live RAG Memory

**Three Responsibilities:**

#### A) FEATURE ENGINEERING

**Input:** Kafka `vitals_raw`

**Processing:**
- Maintain sliding window (30-60 min per patient)
- Compute rolling statistics:
  - `rolling_mean_hr`, `rolling_std_hr`
  - `rolling_mean_sbp`, `rolling_std_sbp`
  - `rolling_mean_spo2`
- Compute deltas:
  - `hr_delta` (current - 5min ago)
  - `sbp_delta`
  - `shock_index_delta`
- Compute trends:
  - `lactate_trend` (slope over window)
- Anomaly detection:
  - `anomaly_flag` (z-score > 2.5)

**Output:** Kafka `vitals_enriched`

**CRITICAL:** 
- ❌ Do NOT compute risk scores
- ❌ Do NOT assign medical states
- ❌ Do NOT trigger alerts
- ✅ Deterministic feature layer ONLY

**Output Schema:**
```json
{
  "patient_id": "P001",
  "timestamp": "2026-02-23T14:32:00Z",
  "heart_rate": 102.3,
  "systolic_bp": 92.1,
  "spo2": 94.2,
  "shock_index": 1.11,
  "rolling_mean_hr": 98.7,
  "rolling_std_hr": 4.2,
  "hr_delta": 3.2,
  "sbp_delta": -8.1,
  "shock_index_delta": 0.15,
  "lactate_trend": 0.2,
  "anomaly_flag": true
}
```

#### B) STREAMING VECTOR INDEX (Live RAG)

**Purpose:** Maintain real-time queryable memory per patient

**Implementation:**
1. **Chunk Generation:**
   - Convert each enriched event into structured text:
     ```
     "Time 14:32 | Patient P001 | HR 102 (↑3.2) | SBP 92 (↓8.1) | ShockIndex 1.11 (↑0.15) | Lactate rising 0.2/hr | Anomaly detected"
     ```

2. **Embedding:**
   - Use sentence-transformers (e.g., `all-MiniLM-L6-v2`)
   - Embed in real-time as events arrive

3. **Index Management:**
   - Maintain per-patient vector index
   - Sliding window: Keep last 1-3 hours
   - Auto-expire old embeddings
   - Patient isolation (no cross-contamination)

4. **Storage:**
   - Use Pathway's native indexing capabilities
   - No external ChromaDB dependency

**Key Properties:**
- Event-driven updates (no batch rebuild)
- No periodic re-indexing
- Memory-efficient (bounded buffer)

#### C) QUERY INTERFACE

**Endpoint:** `POST /pathway/query`

**Input:**
```json
{
  "patient_id": "P001",
  "query_text": "Why is shock index increasing?"
}
```

**Processing:**
1. Embed query using same model
2. Retrieve top-k (k=5) relevant chunks from patient's index
3. Return structured context

**Output:**
```json
{
  "patient_id": "P001",
  "retrieved_context": [
    {
      "text": "Time 14:32 | HR 102 (↑3.2) | SBP 92 (↓8.1) | ShockIndex 1.11 (↑0.15)",
      "timestamp": "2026-02-23T14:32:00Z",
      "relevance_score": 0.87
    },
    ...
  ]
}
```

**CRITICAL:**
- ❌ No LLM inside Pathway
- ✅ Pure retrieval only

---

### 4. ML SERVICE (PLACEHOLDER MODE)

**Role:** Risk Score Authority (Single Source of Truth)

**Responsibilities:**
- Consume `vitals_enriched` from Kafka
- Maintain sequence buffer per patient (e.g., last 60 timesteps)
- Call prediction function:
  ```python
  def predict(sequence: np.ndarray) -> float:
      # Placeholder: return random risk score
      # Production: load LSTM model and infer
      return np.random.uniform(0.1, 0.9)
  ```
- Publish to Kafka `vitals_predictions`

**Output Schema:**
```json
{
  "patient_id": "P001",
  "timestamp": "2026-02-23T14:32:00Z",
  "risk_score": 0.73
}
```

**Design Notes:**
- ML training is handled separately (offline pipeline)
- This service is inference-only
- Stateless (except for sequence buffer)
- Health check endpoint returns model info

**Future Enhancement:**
- Load trained LSTM model from disk
- Include confidence intervals
- Support model versioning

---

### 5. BACKEND API (ORCHESTRATOR)

**Role:** Data Fusion + Client Interface

**Responsibilities:**

#### A) Stream Merging
- Consume both `vitals_enriched` and `vitals_predictions`
- Join by (patient_id, timestamp)
- Maintain unified in-memory state per patient
- Handle timing mismatches (e.g., prediction arrives 100ms after vitals)

#### B) WebSocket Streaming
**Endpoint:** `ws://backend/ws`

**Behavior:**
- Push unified patient view to connected clients
- Schema:
  ```json
  {
    "patient_id": "P001",
    "timestamp": "2026-02-23T14:32:00Z",
    "vitals": {
      "heart_rate": 102.3,
      "systolic_bp": 92.1,
      "spo2": 94.2,
      "shock_index": 1.11
    },
    "features": {
      "rolling_mean_hr": 98.7,
      "hr_delta": 3.2,
      "anomaly_flag": true
    },
    "risk_score": 0.73
  }
  ```

#### C) REST Endpoints

**GET /health**
```json
{
  "status": "healthy",
  "services": {
    "kafka": "connected",
    "ml_service": "healthy",
    "pathway_engine": "healthy"
  }
}
```

**GET /patients**
```json
{
  "patients": [
    {
      "patient_id": "P001",
      "latest_risk_score": 0.73,
      "last_update": "2026-02-23T14:32:00Z"
    },
    ...
  ]
}
```

**GET /patients/{id}/latest**
```json
{
  "patient_id": "P001",
  "vitals": {...},
  "features": {...},
  "risk_score": 0.73
}
```

**GET /patients/{id}/history?hours=4**
```json
{
  "patient_id": "P001",
  "history": [
    {
      "timestamp": "2026-02-23T14:32:00Z",
      "risk_score": 0.73,
      "heart_rate": 102.3
    },
    ...
  ]
}
```

#### D) Chat Endpoint

**POST /chat**

**Input:**
```json
{
  "patient_id": "P001",
  "question": "Why is the patient's risk score increasing?"
}
```

**Flow:**
1. Call Pathway query API:
   ```python
   context = requests.post('http://pathway-engine:8080/query', json={
       'patient_id': patient_id,
       'query_text': question
   })
   ```

2. Pass context + question to LLM:
   ```python
   prompt = f"""
   Context (patient data):
   {context['retrieved_context']}
   
   Question: {question}
   
   Instructions:
   - Use only the provided context
   - Explain clinical trends analytically
   - Do NOT diagnose
   - Do NOT hallucinate
   """
   
   response = llm.generate(prompt)
   ```

3. Return grounded response

**Output:**
```json
{
  "answer": "The risk score is increasing because shock index has risen from 0.96 to 1.11 over the last 15 minutes, driven by rising heart rate (↑3.2) and falling systolic BP (↓8.1).",
  "sources": [
    {
      "timestamp": "2026-02-23T14:32:00Z",
      "text": "Time 14:32 | HR 102 (↑3.2) | SBP 92 (↓8.1)"
    }
  ]
}
```

---

### 6. ALERT ENGINE

**Role:** Threshold-Based Alerting

**Trigger Logic:**
```python
if risk_score > ALERT_THRESHOLD:
    generate_alert()
```

**Alert Generation:**
1. Consume `vitals_predictions` topic
2. Check threshold (e.g., risk_score > 0.75)
3. Fetch recent enriched data for context
4. Generate alert message:
   ```json
   {
     "patient_id": "P001",
     "timestamp": "2026-02-23T14:32:00Z",
     "alert_type": "HIGH_RISK",
     "risk_score": 0.87,
     "context": {
       "hr": 108,
       "sbp": 88,
       "shock_index": 1.23,
       "anomaly_flag": true
     },
     "message": "Patient P001 risk score exceeded threshold (0.87 > 0.75)"
   }
   ```
5. Publish to `alerts_stream` topic
6. Send to notification service (email, Slack, etc.)

**What NOT to do:**
- ❌ No independent rule-based risk calculation
- ❌ No duplicate scoring logic
- ✅ Alert engine is reactive only

---

### 7. FRONTEND

**Role:** Real-Time Visualization

**Display Components:**

#### A) Patient Card
- Patient ID
- Live vitals (HR, BP, SpO2, Temp)
- Risk score (large numeric display)
- Risk badge (color-coded)

#### B) Risk Trend Chart
- Time series graph of risk_score
- Last 4 hours
- Highlight threshold line

#### C) Vitals Trend Chart
- Multi-line chart (HR, SBP, SpO2)
- Show rolling averages

#### D) Alert Banner
- Top of screen
- Show when alert triggers
- Display context from alert message

#### E) RAG Chat Panel
- Text input for questions
- Call `POST /chat` endpoint
- Display grounded responses
- Show source timestamps

**What to REMOVE:**
- Hardcoded state labels (STABLE, CRITICAL)
- Rule-based UI badges
- Mock data generators

**What to ADD:**
- Real-time WebSocket connection
- Risk score visualization
- Chat interface

---

## 📁 FILE STRUCTURE CHANGES

### Before (Current)
```
icu-system/
├── vital-simulator/
│   └── app/
│       └── main.py (1102 lines, state machine)
├── pathway-engine/
│   └── app/
│       ├── main.py
│       ├── risk_engine.py (risk calculation)
│       └── ...
├── ml-service/
│   └── app/
│       └── main.py
├── backend-api/
│   └── main.py (model inference logic)
├── rag-service/ (standalone ChromaDB)
│   └── app/
│       ├── main.py
│       ├── vector_store.py
│       └── kafka_indexer.py
└── alert-system/
```

### After (Refactored)
```
icu-system/
├── vital-simulator/
│   └── app/
│       ├── main.py (NEW: clean drift model)
│       ├── drift_model.py (NEW)
│       └── config.py (NEW)
│
├── pathway-engine/
│   └── app/
│       ├── main.py
│       ├── feature_engineering.py (NEW: replaces risk_engine.py)
│       ├── streaming_rag.py (NEW: live vector index)
│       ├── query_api.py (NEW: HTTP query interface)
│       ├── embeddings.py (NEW: sentence-transformers)
│       └── config.py
│
├── ml-service/
│   └── app/
│       ├── main.py (NEW: Kafka consumer + producer)
│       ├── inference.py (NEW: placeholder predict())
│       ├── sequence_buffer.py (NEW)
│       └── config.py
│
├── backend-api/
│   └── app/
│       ├── main.py (NEW: FastAPI app)
│       ├── stream_merger.py (NEW: join vitals + predictions)
│       ├── websocket_handler.py (NEW)
│       ├── rest_endpoints.py (NEW)
│       ├── chat_handler.py (NEW: RAG integration)
│       └── config.py
│
├── alert-engine/ (NEW: replaces old alert-system)
│   └── app/
│       ├── main.py (NEW: threshold-based)
│       ├── alert_generator.py (NEW)
│       └── config.py
│
└── [REMOVE] rag-service/ (functionality moved to pathway-engine)
```

---

## 🚀 IMPLEMENTATION ROADMAP

### PHASE 1: Vital Simulator Refactor

**Goal:** Replace state machine with realistic drift model

**Tasks:**
1. Create new drift model:
   ```python
   # drift_model.py
   class PhysiologicalDriftModel:
       def __init__(self, patient_id):
           self.baseline_hr = random.uniform(65, 85)
           self.baseline_sbp = random.uniform(110, 130)
           self.hr_drift_rate = 0  # bpm/minute
           self.sbp_drift_rate = 0  # mmHg/minute
           
       def apply_drift(self, dt):
           # Brownian motion with occasional regime shifts
           pass
           
       def simulate_deterioration(self):
           # Probabilistic trigger (5% per interval)
           if random.random() < 0.05:
               self.hr_drift_rate += random.uniform(0.1, 0.5)
               self.sbp_drift_rate -= random.uniform(0.05, 0.3)
   ```

2. Remove state machine:
   - Delete `PatientState` enum
   - Delete transition logic
   - Delete acute event system

3. Update output schema:
   - Add `shock_index` calculation
   - Add `lactate` field
   - Add `map` (mean arterial pressure)

4. Rename Kafka topic:
   ```python
   KAFKA_TOPIC = 'vitals_raw'
   ```

**Files to modify:**
- `vital-simulator/app/main.py` (major rewrite)
- Create `vital-simulator/app/drift_model.py`
- Create `vital-simulator/app/config.py`

**Testing:**
- Run simulator for 10 minutes
- Verify gradual trends (no spikes)
- Check Kafka messages in `vitals_raw`

---

### PHASE 2: Kafka Cleanup

**Goal:** Ensure proper topic configuration

**Tasks:**
1. Update `docker-compose.yml`:
   ```yaml
   environment:
     KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
     KAFKA_LOG_RETENTION_HOURS: 168  # 7 days
     KAFKA_LOG_SEGMENT_BYTES: 1073741824  # 1GB
   ```

2. Create topic initialization script:
   ```bash
   # init-topics.sh
   kafka-topics --create --topic vitals_raw --partitions 3 --replication-factor 1
   kafka-topics --create --topic vitals_enriched --partitions 3 --replication-factor 1
   kafka-topics --create --topic vitals_predictions --partitions 3 --replication-factor 1
   kafka-topics --create --topic alerts_stream --partitions 1 --replication-factor 1
   ```

3. Enable idempotent producers in all services:
   ```python
   producer = KafkaProducer(
       enable_idempotence=True,
       acks='all',
       retries=3
   )
   ```

**Files to modify:**
- `icu-system/docker-compose.yml`
- Create `icu-system/init-topics.sh`

---

### PHASE 3: Pathway Engine — Feature Engineering

**Goal:** Replace risk calculation with feature engineering only

**Tasks:**
1. Create `feature_engineering.py`:
   ```python
   import pathway as pw
   
   def create_feature_pipeline(vitals_stream: pw.Table) -> pw.Table:
       # Sliding window aggregation
       windowed = vitals_stream.windowby(
           pw.this.patient_id,
           window=pw.temporal.sliding(
               hop=pw.Duration.seconds(10),
               duration=pw.Duration.minutes(30)
           ),
           instance=pw.this.timestamp
       )
       
       # Compute rolling statistics
       features = windowed.reduce(
           patient_id=pw.this.patient_id,
           timestamp=pw.reducers.latest(pw.this.timestamp),
           rolling_mean_hr=pw.reducers.avg(pw.this.heart_rate),
           rolling_std_hr=pw.reducers.stddev(pw.this.heart_rate),
           rolling_mean_sbp=pw.reducers.avg(pw.this.systolic_bp),
           hr_delta=pw.reducers.latest(pw.this.heart_rate) - pw.reducers.earliest(pw.this.heart_rate),
           # ... more features
       )
       
       return features
   ```

2. Remove risk calculation:
   - Delete/archive `risk_engine.py`
   - Remove all risk score computations
   - Remove medical state assignments

3. Update output topic:
   ```python
   OUTPUT_TOPIC = 'vitals_enriched'
   ```

**Files to modify:**
- `pathway-engine/app/main.py`
- Create `pathway-engine/app/feature_engineering.py`
- Archive `pathway-engine/app/risk_engine.py`

**Testing:**
- Consume `vitals_enriched` topic
- Verify feature values (rolling_mean_hr, etc.)
- Check for no risk_score field

---

### PHASE 4: Pathway Engine — Streaming RAG

**Goal:** Implement live vector index inside Pathway

**Tasks:**
1. Create `streaming_rag.py`:
   ```python
   import pathway as pw
   from sentence_transformers import SentenceTransformer
   import numpy as np
   
   class StreamingRAGIndex:
       def __init__(self):
           self.model = SentenceTransformer('all-MiniLM-L6-v2')
           self.patient_indices = {}  # patient_id -> index
           
       def add_enriched_event(self, event):
           # Convert to text
           chunk = self._event_to_text(event)
           
           # Embed
           embedding = self.model.encode(chunk)
           
           # Add to patient's index
           self._add_to_index(event['patient_id'], chunk, embedding, event['timestamp'])
           
       def _event_to_text(self, event):
           return f"Time {event['timestamp']} | HR {event['heart_rate']} (Δ{event['hr_delta']}) | SBP {event['systolic_bp']} (Δ{event['sbp_delta']}) | ShockIndex {event['shock_index']}"
           
       def query(self, patient_id, query_text, top_k=5):
           query_embedding = self.model.encode(query_text)
           # Retrieve top-k from patient's index
           return self._retrieve(patient_id, query_embedding, top_k)
   ```

2. Integrate with Pathway stream:
   ```python
   # In main.py
   rag_index = StreamingRAGIndex()
   
   enriched_stream.apply(
       lambda row: rag_index.add_enriched_event(row),
       result=pw.Table.empty()
   )
   ```

3. Add sliding window expiry:
   ```python
   def _cleanup_old_embeddings(self, patient_id, max_age_hours=3):
       now = datetime.now()
       cutoff = now - timedelta(hours=max_age_hours)
       # Remove embeddings older than cutoff
   ```

**Files to modify:**
- Create `pathway-engine/app/streaming_rag.py`
- Create `pathway-engine/app/embeddings.py`
- Update `pathway-engine/requirements.txt`:
  ```
  sentence-transformers==2.2.2
  numpy>=1.24.0
  ```

**Testing:**
- Add test events to `vitals_enriched`
- Call query function
- Verify retrieval works

---

### PHASE 5: Pathway Engine — Query API

**Goal:** Expose HTTP endpoint for RAG queries

**Tasks:**
1. Create `query_api.py`:
   ```python
   from fastapi import FastAPI
   from pydantic import BaseModel
   
   app = FastAPI()
   
   class QueryRequest(BaseModel):
       patient_id: str
       query_text: str
       top_k: int = 5
   
   @app.post("/query")
   async def query_rag(request: QueryRequest):
       results = rag_index.query(
           request.patient_id,
           request.query_text,
           request.top_k
       )
       return {
           "patient_id": request.patient_id,
           "retrieved_context": results
       }
   ```

2. Run FastAPI alongside Pathway:
   ```python
   # In main.py
   import threading
   import uvicorn
   
   def run_api():
       uvicorn.run(query_api.app, host="0.0.0.0", port=8080)
   
   api_thread = threading.Thread(target=run_api, daemon=True)
   api_thread.start()
   
   pw.run()  # Run Pathway pipeline
   ```

**Files to modify:**
- Create `pathway-engine/app/query_api.py`
- Update `pathway-engine/app/main.py`

**Testing:**
- `curl -X POST http://localhost:8080/query -d '{"patient_id":"P001","query_text":"shock index"}'`
- Verify context returned

---

### PHASE 6: ML Service Refactor

**Goal:** Consume enriched data, publish risk scores

**Tasks:**
1. Create Kafka consumer:
   ```python
   # main.py
   from kafka import KafkaConsumer, KafkaProducer
   
   consumer = KafkaConsumer(
       'vitals_enriched',
       bootstrap_servers='kafka:9092',
       group_id='ml-service'
   )
   
   producer = KafkaProducer(
       bootstrap_servers='kafka:9092',
       value_serializer=lambda v: json.dumps(v).encode('utf-8')
   )
   ```

2. Create sequence buffer:
   ```python
   # sequence_buffer.py
   class PatientSequenceBuffer:
       def __init__(self, window_size=60):
           self.buffers = {}  # patient_id -> deque
           self.window_size = window_size
           
       def add(self, patient_id, features):
           if patient_id not in self.buffers:
               self.buffers[patient_id] = deque(maxlen=self.window_size)
           self.buffers[patient_id].append(features)
           
       def get_sequence(self, patient_id):
           return np.array(self.buffers[patient_id])
   ```

3. Placeholder prediction:
   ```python
   # inference.py
   def predict_risk(sequence: np.ndarray) -> float:
       # Placeholder: return weighted average of shock index
       # Production: load LSTM model
       return np.random.uniform(0.1, 0.9)
   ```

4. Kafka producer:
   ```python
   for message in consumer:
       enriched_data = json.loads(message.value)
       
       # Add to buffer
       sequence_buffer.add(enriched_data['patient_id'], enriched_data)
       
       # Predict if buffer full
       if sequence_buffer.is_ready(enriched_data['patient_id']):
           sequence = sequence_buffer.get_sequence(enriched_data['patient_id'])
           risk_score = predict_risk(sequence)
           
           # Publish
           producer.send('vitals_predictions', {
               'patient_id': enriched_data['patient_id'],
               'timestamp': enriched_data['timestamp'],
               'risk_score': risk_score
           })
   ```

**Files to modify:**
- Rewrite `ml-service/app/main.py`
- Create `ml-service/app/inference.py`
- Create `ml-service/app/sequence_buffer.py`

**Testing:**
- Check consumer reads `vitals_enriched`
- Verify `vitals_predictions` receives messages
- Check risk_score values

---

### PHASE 7: Backend API Refactor

**Goal:** Merge streams, expose WebSocket + REST + Chat

**Tasks:**
1. Create stream merger:
   ```python
   # stream_merger.py
   from kafka import KafkaConsumer
   from collections import defaultdict
   
   class StreamMerger:
       def __init__(self):
           self.vitals_consumer = KafkaConsumer('vitals_enriched')
           self.predictions_consumer = KafkaConsumer('vitals_predictions')
           self.patient_state = defaultdict(dict)
           
       def merge(self):
           for msg in self.vitals_consumer:
               data = json.loads(msg.value)
               patient_id = data['patient_id']
               self.patient_state[patient_id]['vitals'] = data
               
           for msg in self.predictions_consumer:
               data = json.loads(msg.value)
               patient_id = data['patient_id']
               self.patient_state[patient_id]['risk_score'] = data['risk_score']
               
               # Emit unified view
               yield self._get_unified_view(patient_id)
   ```

2. Create WebSocket handler:
   ```python
   # websocket_handler.py
   from fastapi import WebSocket
   
   @app.websocket("/ws")
   async def websocket_endpoint(websocket: WebSocket):
       await websocket.accept()
       
       for unified_view in stream_merger.merge():
           await websocket.send_json(unified_view)
   ```

3. Create REST endpoints:
   ```python
   # rest_endpoints.py
   @app.get("/patients")
   async def get_patients():
       return {"patients": list(stream_merger.patient_state.keys())}
   
   @app.get("/patients/{patient_id}/latest")
   async def get_patient_latest(patient_id: str):
       return stream_merger.patient_state[patient_id]
   ```

4. Create chat handler:
   ```python
   # chat_handler.py
   import httpx
   
   @app.post("/chat")
   async def chat(request: ChatRequest):
       # Query Pathway
       async with httpx.AsyncClient() as client:
           context_response = await client.post(
               'http://pathway-engine:8080/query',
               json={
                   'patient_id': request.patient_id,
                   'query_text': request.question
               }
           )
       
       context = context_response.json()['retrieved_context']
       
       # Generate LLM response
       prompt = f"Context: {context}\n\nQuestion: {request.question}"
       answer = llm_generate(prompt)  # Use OpenAI/Anthropic/etc.
       
       return {"answer": answer, "sources": context}
   ```

**Files to modify:**
- Rewrite `backend-api/main.py`
- Create `backend-api/app/stream_merger.py`
- Create `backend-api/app/websocket_handler.py`
- Create `backend-api/app/rest_endpoints.py`
- Create `backend-api/app/chat_handler.py`

**Testing:**
- Test WebSocket connection
- Test REST endpoints
- Test chat with sample question

---

### PHASE 8: Alert Engine

**Goal:** Threshold-based alerting from predictions

**Tasks:**
1. Create alert engine:
   ```python
   # main.py
   from kafka import KafkaConsumer, KafkaProducer
   
   ALERT_THRESHOLD = 0.75
   
   consumer = KafkaConsumer('vitals_predictions')
   producer = KafkaProducer()
   enriched_consumer = KafkaConsumer('vitals_enriched')
   
   for msg in consumer:
       prediction = json.loads(msg.value)
       
       if prediction['risk_score'] > ALERT_THRESHOLD:
           # Fetch context
           context = get_recent_enriched_data(prediction['patient_id'])
           
           # Generate alert
           alert = {
               'patient_id': prediction['patient_id'],
               'timestamp': prediction['timestamp'],
               'alert_type': 'HIGH_RISK',
               'risk_score': prediction['risk_score'],
               'context': context,
               'message': f"Risk score {prediction['risk_score']:.2f} exceeds threshold"
           }
           
           producer.send('alerts_stream', alert)
           send_notification(alert)  # Email, Slack, etc.
   ```

**Files to modify:**
- Create `alert-engine/app/main.py`
- Create `alert-engine/app/alert_generator.py`
- Create `alert-engine/app/config.py`

**Testing:**
- Manually publish high risk_score
- Verify alert generated
- Check alert message content

---

### PHASE 9: Frontend Cleanup

**Goal:** Remove hardcoded states, display ML-driven data

**Tasks:**
1. Remove state-based UI:
   ```typescript
   // PatientCard.tsx - REMOVE
   const stateColors = {
     STABLE: 'green',
     CRITICAL: 'red'
   }
   ```

2. Add risk score display:
   ```typescript
   // PatientCard.tsx
   <div className="risk-score">
     <span className="text-4xl font-bold">{riskScore.toFixed(2)}</span>
     <span className="text-sm">Risk Score</span>
   </div>
   ```

3. Add risk trend chart:
   ```typescript
   // RiskTrendChart.tsx
   import { LineChart } from 'recharts'
   
   const RiskTrendChart = ({ patient_id }) => {
     const history = usePatientRiskHistory(patient_id)
     
     return (
       <LineChart data={history}>
         <Line dataKey="risk_score" stroke="#f59e0b" />
         <ReferenceLine y={0.75} stroke="red" label="Threshold" />
       </LineChart>
     )
   }
   ```

4. Add chat panel:
   ```typescript
   // RAGChatPanel.tsx
   const RAGChatPanel = ({ patient_id }) => {
     const [question, setQuestion] = useState('')
     const [answer, setAnswer] = useState('')
     
     const handleSubmit = async () => {
       const response = await fetch('/api/chat', {
         method: 'POST',
         body: JSON.stringify({ patient_id, question })
       })
       const data = await response.json()
       setAnswer(data.answer)
     }
     
     return (
       <div className="chat-panel">
         <input value={question} onChange={e => setQuestion(e.target.value)} />
         <button onClick={handleSubmit}>Ask</button>
         <div className="answer">{answer}</div>
       </div>
     )
   }
   ```

**Files to modify:**
- `frontend/src/components/PatientCard.tsx`
- `frontend/src/components/RiskTrendChart.tsx`
- Create `frontend/src/components/RAGChatPanel.tsx`
- Update `frontend/src/hooks/usePatients.ts`

**Testing:**
- Check UI displays risk scores
- Verify charts render
- Test chat functionality

---

### PHASE 10: Logging & Production Hardening

**Goal:** Clean, production-ready logs

**Tasks:**
1. Standardize logging format:
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       handlers=[
           logging.StreamHandler(),
           logging.FileHandler('/var/log/service.log')
       ]
   )
   ```

2. Remove debug noise:
   - Delete emoji logs
   - Remove simulation event logs
   - Keep only critical events

3. Add structured logging:
   ```python
   import structlog
   
   logger = structlog.get_logger()
   logger.info("patient_update", patient_id="P001", risk_score=0.73)
   ```

4. Add health checks:
   ```python
   @app.get("/health")
   async def health():
       return {
           "status": "healthy",
           "kafka": check_kafka_connection(),
           "model_loaded": state.model_loaded
       }
   ```

5. Add monitoring:
   - Prometheus metrics
   - Grafana dashboards
   - Alert on consumer lag

**Files to modify:**
- All `main.py` files across services
- Add `requirements.txt` entry: `structlog`

---

## ✅ VALIDATION CHECKLIST

After completing all phases, verify:

### Architecture Validation
- [ ] Vital simulator produces realistic gradual trends (no spikes)
- [ ] Kafka topics have linear growth (no explosion)
- [ ] Pathway only publishes enriched features (no risk scores)
- [ ] ML service is sole risk authority
- [ ] Pathway RAG index updates in real-time
- [ ] Backend merges streams correctly
- [ ] Frontend displays ML-driven data

### Functional Validation
- [ ] WebSocket streams unified patient view
- [ ] REST endpoints return correct data
- [ ] Chat endpoint returns grounded responses
- [ ] Alerts trigger at threshold
- [ ] No hardcoded medical states in UI

### Performance Validation
- [ ] Kafka throughput: ~8 msgs/sec (8 patients × 1 msg/sec)
- [ ] Pathway latency: <100ms per event
- [ ] ML inference latency: <50ms
- [ ] WebSocket latency: <200ms end-to-end
- [ ] RAG query latency: <300ms

### Production Validation
- [ ] All services have health checks
- [ ] Logs are clean and structured
- [ ] Docker containers restart on failure
- [ ] No memory leaks after 24hr run
- [ ] Consumer lag < 1 second

---

## 🎯 SUCCESS METRICS

### Before Refactor
- **Risk Authority:** Duplicated (Pathway + Backend)
- **RAG System:** Batch indexing (ChromaDB)
- **Simulation:** Unrealistic spikes
- **Topic Growth:** Exponential (bug)
- **Code Quality:** Mixed concerns

### After Refactor
- **Risk Authority:** Single (ML Service)
- **RAG System:** Streaming (Pathway native)
- **Simulation:** Realistic drift
- **Topic Growth:** Linear (O(n))
- **Code Quality:** Clean separation

---

## 📚 ADDITIONAL RESOURCES

### Pathway Resources
- [Pathway Streaming Docs](https://pathway.com/developers/documentation)
- [Pathway Vector Index Tutorial](https://pathway.com/developers/showcases/rag-with-streaming-data)

### ML Resources
- [LSTM for Time Series](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [Medical AI Ethics](https://www.nature.com/articles/s41591-021-01614-0)

### Kafka Resources
- [Kafka Streams Tutorial](https://kafka.apache.org/documentation/streams/)
- [Idempotent Producers](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/)

---

## 🤝 NEXT STEPS

1. **Review this document** with your team
2. **Prioritize phases** based on dependencies
3. **Create GitHub issues** for each phase
4. **Set up development branches** for parallel work
5. **Begin with Phase 1** (Vital Simulator) as foundation
6. **Test incrementally** after each phase
7. **Document learnings** as you go

---

**Document Version:** 1.0  
**Last Updated:** February 23, 2026  
**Author:** Senior Distributed Systems Architect  
**Status:** Ready for Implementation
