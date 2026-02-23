# VitalX Architecture Refactor — Quick Reference

**TL;DR: From Rule-Based to ML-Driven Streaming Architecture**

---

## 📊 BEFORE vs AFTER COMPARISON

### Data Flow Architecture

#### ❌ BEFORE (Current State)

```
Vital Simulator (State Machine)
    ├── STABLE → EARLY → CRITICAL states
    ├── Hardcoded transitions
    └── Unrealistic spikes
         ↓
    Kafka "vitals"
         ↓
Pathway Engine
    ├── Feature engineering
    ├── ❌ Risk calculation (DUPLICATE LOGIC)
    └── ❌ Medical state assignment
         ↓
    Kafka "vitals_enriched" (with risk_score)
         ↓
ML Service (confused role)
    ├── Has model but doesn't use enriched data
    └── ❌ Disconnected from main flow
         ↓
Backend API
    ├── ❌ ALSO calculates risk (SECOND DUPLICATE)
    ├── Standalone model inference
    └── Conflicting risk authorities
         ↓
    Frontend (hardcoded states)
    
Separate:
RAG Service (ChromaDB)
    ├── ❌ Batch indexing (not real-time)
    ├── Kafka consumer → batch rebuild
    └── Standalone vector DB
```

**Problems:**
- 3 different risk calculation sources (Pathway, ML Service, Backend)
- State machine produces unrealistic vitals
- RAG is batch-updated, not streaming
- Topic explosion bug (1M+ messages/hour)
- Mixed concerns everywhere

---

#### ✅ AFTER (Target State)

```
Vital Simulator (Drift Model)
    ├── Physiological baselines
    ├── Gradual drift (Brownian motion)
    ├── Probabilistic deterioration
    └── NO states, NO spikes
         ↓
    Kafka "vitals_raw"
    {patient_id, timestamp, heart_rate, systolic_bp, ..., shock_index}
         ↓
Pathway Engine (3 Functions)
    ├─────────────────────────────────────┐
    │ A) Feature Engineering              │
    │    • Sliding window (30-60 min)     │
    │    • Rolling statistics             │
    │    • Deltas, trends                 │
    │    • Anomaly flags                  │
    │    • ✅ NO RISK SCORING             │
    │         ↓                            │
    │    Kafka "vitals_enriched"          │
    │    {+ rolling_mean_hr, + hr_delta,  │
    │     + anomaly_flag}                 │
    │                                      │
    │ B) Streaming Vector Index           │
    │    • Convert events → text chunks   │
    │    • Embed in real-time             │
    │    • Sliding window (3 hrs)         │
    │    • Per-patient isolation          │
    │    • Auto-expire old data           │
    │                                      │
    │ C) Query API (HTTP)                 │
    │    POST /query                       │
    │    {patient_id, query_text}         │
    │    → Retrieved context              │
    └─────────────────────────────────────┘
         ↓
    Kafka "vitals_enriched"
         ↓
ML Service (SOLE RISK AUTHORITY) ⭐
    ├── Consume enriched data
    ├── Sequence buffer (60 timesteps)
    ├── predict(sequence) → risk_score
    └── ✅ SINGLE SOURCE OF TRUTH
         ↓
    Kafka "vitals_predictions"
    {patient_id, timestamp, risk_score}
         ↓
Backend API (Orchestrator)
    ├── Merge vitals_enriched + vitals_predictions
    ├── WebSocket /ws → unified stream
    ├── REST endpoints
    └── Chat endpoint → Query Pathway RAG → LLM
         ↓
    Frontend (ML-driven)
    ├── Live vitals
    ├── Risk score display
    ├── Risk trend chart
    └── RAG chat panel
    
Alert Engine (Parallel)
    ├── Watch vitals_predictions
    ├── Trigger: risk_score > threshold
    └── Generate alert with context
```

**Solutions:**
- ✅ ML Service is sole risk authority
- ✅ Pathway does deterministic features only
- ✅ Streaming RAG inside Pathway (no batch indexing)
- ✅ Clean separation of concerns
- ✅ Linear topic growth (no explosion)

---

## 🎯 RESPONSIBILITY MATRIX

| Component | BEFORE | AFTER |
|-----------|--------|-------|
| **Vital Simulator** | State machine with transitions | Drift model, gradual changes |
| **Risk Calculation** | Pathway + Backend (duplicated) | ML Service ONLY |
| **Feature Engineering** | Mixed with risk logic | Pathway ONLY |
| **RAG System** | Standalone ChromaDB (batch) | Pathway native (streaming) |
| **Medical States** | Hardcoded everywhere | REMOVED |
| **Alert Logic** | Rule-based independent system | Threshold on ML risk_score |

---

## 🔑 KEY DESIGN PRINCIPLES

### 1. Single Source of Truth
- **Risk Score:** ML Service publishes to `vitals_predictions`
- No other service calculates risk

### 2. Separation of Concerns
- **Pathway:** Feature engineering + RAG memory
- **ML Service:** Risk inference
- **Backend:** Orchestration + UI serving

### 3. Streaming-First
- No batch processing
- Event-driven updates
- Real-time indexing

### 4. Clean Data Flow
```
Raw → Feature Engineering → ML Inference → Presentation
```

---

## 📦 KAFKA TOPICS

| Topic | Producer | Consumer | Schema | Purpose |
|-------|----------|----------|--------|---------|
| `vitals_raw` | Vital Simulator | Pathway | {patient_id, timestamp, heart_rate, systolic_bp, ...} | Raw physiological data |
| `vitals_enriched` | Pathway | ML Service, Backend | {+ rolling_mean_hr, + hr_delta, + anomaly_flag} | Feature-engineered data |
| `vitals_predictions` | ML Service | Backend, Alert Engine | {patient_id, timestamp, risk_score} | ML risk scores |
| `alerts_stream` | Alert Engine | Frontend, Notification | {patient_id, alert_type, risk_score, context} | Alert events |

---

## 🚀 MIGRATION CHECKLIST

### Phase 1: Foundation
- [ ] Refactor Vital Simulator (drift model)
- [ ] Configure Kafka topics
- [ ] Remove state machine code

### Phase 2: Pathway Refactor
- [ ] Create feature_engineering.py
- [ ] Remove risk calculation from Pathway
- [ ] Implement streaming RAG index
- [ ] Expose query API

### Phase 3: ML Service
- [ ] Kafka consumer for vitals_enriched
- [ ] Sequence buffer implementation
- [ ] Placeholder predict() function
- [ ] Kafka producer for vitals_predictions

### Phase 4: Backend Integration
- [ ] Stream merger (join enriched + predictions)
- [ ] WebSocket handler
- [ ] REST endpoints
- [ ] Chat endpoint with Pathway RAG

### Phase 5: Frontend
- [ ] Remove hardcoded state labels
- [ ] Add risk score display
- [ ] Risk trend chart component
- [ ] RAG chat panel

### Phase 6: Production
- [ ] Clean logging (remove emojis)
- [ ] Health checks
- [ ] Monitoring (Prometheus/Grafana)
- [ ] Load testing

---

## 🧪 VALIDATION SCENARIOS

### Scenario 1: Risk Score Authority
**Test:** Publish to vitals_enriched, check predictions
- ✅ ML Service publishes to vitals_predictions
- ✅ Backend serves risk_score from predictions topic
- ✅ Frontend displays ML-driven risk
- ❌ No risk_score in vitals_enriched

### Scenario 2: RAG Query
**Test:** Ask "Why is shock index increasing?"
- ✅ Backend calls Pathway query API
- ✅ Pathway retrieves recent chunks
- ✅ LLM generates grounded response
- ❌ No hallucination, no diagnosis

### Scenario 3: Realistic Vitals
**Test:** Run simulator for 10 minutes
- ✅ Gradual trends (no spikes)
- ✅ Drift rates realistic (<0.5 bpm/min)
- ✅ Occasional deterioration triggers
- ❌ No state transitions

### Scenario 4: Topic Growth
**Test:** Run system for 1 hour
- ✅ Linear growth (~28,800 events total)
  - 8 patients × 1 msg/sec × 3600 sec = 28,800
- ✅ vitals_enriched has ~same count as vitals_raw
- ❌ Not 1M+ messages (explosion bug)

---

## 🔧 QUICK TROUBLESHOOTING

### Issue: No risk scores appearing
**Check:**
1. ML Service consuming vitals_enriched?
2. Sequence buffer full? (needs 60 events)
3. ML Service publishing to vitals_predictions?

### Issue: RAG query returns empty
**Check:**
1. Pathway index receiving events?
2. Patient ID exists in index?
3. Embeddings not expired? (3 hour window)

### Issue: Kafka topic explosion
**Check:**
1. Pathway using latest-state materialization?
2. Multiple groupby operations? (reduce to one)
3. Consumer lag increasing?

### Issue: WebSocket not updating
**Check:**
1. Backend consuming both topics?
2. Stream merger joining correctly?
3. WebSocket connection alive?

---

## 📚 CRITICAL CODE LOCATIONS

### Remove Risk Calculation
**Files to modify:**
- `pathway-engine/app/risk_engine.py` → ARCHIVE
- `backend-api/main.py` → Remove model inference

### Remove State Machine
**Files to modify:**
- `vital-simulator/app/main.py` → Delete PatientState enum
- Remove transition logic (lines ~200-400)

### Remove Standalone RAG
**Files to remove:**
- `rag-service/` → DELETE entire folder
- Functionality moved to pathway-engine

### Add Streaming RAG
**New files:**
- `pathway-engine/app/streaming_rag.py`
- `pathway-engine/app/query_api.py`
- `pathway-engine/app/embeddings.py`

---

## 🎓 LEARNING RESOURCES

### Pathway Streaming
- [Pathway Documentation](https://pathway.com/developers/documentation)
- [Streaming RAG Tutorial](https://pathway.com/developers/showcases/rag-with-streaming-data)

### Kafka Best Practices
- [Idempotent Producers](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/)
- [Topic Design](https://www.confluent.io/blog/how-choose-number-topics-partitions-kafka-cluster/)

### Medical AI
- [Ethics in Medical AI](https://www.nature.com/articles/s41591-021-01614-0)
- [Physiological Modeling](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/expphysiol.2009.051748)

---

## 💡 COMMON PITFALLS TO AVOID

### ❌ Don't Do This
1. **Multiple Risk Calculations**
   - Only ML Service should publish risk_score
   
2. **Batch RAG Indexing**
   - Use streaming updates, not periodic rebuilds
   
3. **Cross-Patient Data Leakage**
   - Ensure patient isolation in RAG index
   
4. **Hardcoded State Labels**
   - Remove STABLE/CRITICAL from UI
   
5. **Duplicate Kafka Topics**
   - Stick to 4 topics: vitals_raw, vitals_enriched, vitals_predictions, alerts_stream

### ✅ Do This Instead
1. **Clear Ownership**
   - Each metric has one authoritative source
   
2. **Event-Driven Updates**
   - React to Kafka events, don't poll
   
3. **Patient Isolation**
   - Use patient_id as partition key
   
4. **Dynamic UI**
   - Display ML-driven risk scores
   
5. **Topic Hygiene**
   - Monitor growth, configure retention

---

## 🚦 GO-LIVE CHECKLIST

### Pre-Production
- [ ] All services have health checks
- [ ] Logging is production-ready (no debug noise)
- [ ] Kafka topics configured with proper retention
- [ ] Consumer groups properly named
- [ ] Docker containers restart on failure

### Production Monitoring
- [ ] Prometheus metrics exposed
- [ ] Grafana dashboards created
- [ ] Alert rules configured
- [ ] Consumer lag monitoring
- [ ] Error rate tracking

### Documentation
- [ ] API documentation (Swagger/OpenAPI)
- [ ] Deployment runbook
- [ ] Troubleshooting guide
- [ ] Architecture diagram updated

---

## 📞 SUPPORT CONTACTS

For implementation questions:
- **Architecture:** See [STREAMING_ARCHITECTURE_REFACTOR.md](STREAMING_ARCHITECTURE_REFACTOR.md)
- **Code Templates:** See [IMPLEMENTATION_TEMPLATES.md](IMPLEMENTATION_TEMPLATES.md)
- **Pathway Issues:** [Pathway Discord](https://discord.gg/pathway)
- **Kafka Issues:** [Confluent Community](https://forum.confluent.io/)

---

**Version:** 1.0  
**Last Updated:** February 23, 2026  
**Status:** Ready for Implementation

---

## 🎯 IMMEDIATE NEXT STEPS

1. **Start with Vital Simulator** (Phase 1)
   - Self-contained
   - No dependencies
   - Immediate visible impact

2. **Then Pathway Features** (Phase 3)
   - Remove risk calculation
   - Add feature engineering
   - Test with simulator

3. **Then ML Service** (Phase 6)
   - Add Kafka consumer
   - Implement placeholder predict()
   - Publish to vitals_predictions

4. **Integration** (Phases 7-9)
   - Backend stream merger
   - Frontend updates
   - Alert engine

5. **Polish** (Phase 10)
   - Clean logging
   - Health checks
   - Production hardening

**Estimated Timeline:**
- Phase 1: 2-3 days
- Phase 2-3: 3-4 days
- Phase 4-6: 4-5 days
- Phase 7-9: 3-4 days
- Phase 10: 2-3 days

**Total: ~2-3 weeks for complete refactor**

Good luck! 🚀
