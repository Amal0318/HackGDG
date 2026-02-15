# ICU Digital Twin System - Data Flow Architecture

## System Overview

The ICU Digital Twin System is a microservice-based real-time patient monitoring and deterioration prediction system. It processes vital signs data through multiple stages to provide actionable insights for healthcare providers.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ICU Digital Twin System                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│  │   Vital      │─────▶│    Kafka     │◀─────│   Pathway    │              │
│  │  Simulator   │      │   Broker     │      │    Engine    │              │
│  └──────────────┘      └──────────────┘      └──────────────┘              │
│                               │                       │                      │
│                               │                       │                      │
│                               ▼                       ▼                      │
│                        ┌─────────────┐        ┌──────────────┐             │
│                        │    Topic    │        │    Topic     │             │
│                        │   vitals    │        │   vitals_    │             │
│                        │             │        │   enriched   │             │
│                        └─────────────┘        └──────────────┘             │
│                                                       │                      │
│                                                       ▼                      │
│                                                ┌──────────────┐             │
│                                                │  ML Service  │             │
│                                                │   (LSTM)     │             │
│                                                └──────────────┘             │
│                                                       │                      │
│                                                       ▼                      │
│                                                ┌──────────────┐             │
│                                                │    Topic     │             │
│                                                │   vitals_    │             │
│                                                │   predictions│             │
│                                                └──────────────┘             │
│                                                       │                      │
│                                                       ▼                      │
│                                                ┌──────────────┐             │
│                                                │  Backend API │             │
│                                                │ (REST/WebSocket)          │
│                                                └──────────────┘             │
│                                                       │                      │
│                                                       ▼                      │
│                                               External Clients              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Stages

### Stage 1: Vital Signs Generation
**Service:** vital-simulator
**Container:** icu-vital-simulator
**Port:** Internal only

#### Function:
- Generates realistic ICU patient vital signs data for 8 patients
- Simulates three patient states:
  * STABLE: Normal physiological parameters
  * EARLY_DETERIORATION: Initial signs of patient decline
  * LATE_DETERIORATION: Critical deterioration

#### Output:
- **Kafka Topic:** `vitals`
- **Data Format:**
  ```json
  {
    "patient_id": "ICU-001",
    "timestamp": "2026-02-14T10:30:00Z",
    "heart_rate": 72,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "spo2": 98,
    "respiratory_rate": 16,
    "temperature": 37.0,
    "state": "STABLE"
  }
  ```

#### Key Features:
- Physiological correlations (e.g., HR/BP relationship)
- State transitions with realistic timing
- Acute medical event simulation
- Configurable generation rate

---

### Stage 2: Risk Analysis & Enrichment
**Service:** pathway-engine
**Container:** icu-pathway-engine
**Port:** Internal only

#### Function:
- Real-time streaming processing using Pathway framework
- Consumes raw vital signs from `vitals` topic
- Performs risk scoring and anomaly detection
- Enriches data with clinical context

#### Input:
- **Kafka Topic:** `vitals`

#### Output:
- **Kafka Topic:** `vitals_enriched`
- **Data Format:**
  ```json
  {
    "patient_id": "ICU-001",
    "timestamp": "2026-02-14T10:30:00Z",
    "heart_rate": 72,
    "systolic_bp": 120,
    "diastolic_bp": 80,
    "spo2": 98,
    "respiratory_rate": 16,
    "temperature": 37.0,
    "risk_score": 0.25,
    "risk_components": {
      "vital_deviation_score": 0.15,
      "critical_threshold_score": 0.05,
      "trend_score": 0.05
    },
    "anomaly_flags": {
      "hr_anomaly": false,
      "bp_anomaly": false,
      "spo2_anomaly": false,
      "rr_anomaly": false,
      "temp_anomaly": false
    },
    "processing_time_ms": 2.5
  }
  ```

#### Risk Scoring Algorithm:
1. **Vital Deviation Score**: Distance from normal ranges
2. **Critical Threshold Score**: Proximity to dangerous values
3. **Trend Analysis**: Rate of change detection
4. **Anomaly Detection**: Statistical outlier identification

---

### Stage 3: ML Prediction
**Service:** ml-service
**Container:** icu-ml-service
**Port:** 8001 (mapped to 8000 internally)

#### Function:
- Consumes enriched vital signs from `vitals_enriched` topic
- Applies LSTM deep learning model for deterioration prediction
- Provides REST API for on-demand predictions
- Publishes predictions back to Kafka

#### Input:
- **Kafka Topic:** `vitals_enriched`
- **REST Endpoint:** POST /predict

#### Output:
- **Kafka Topic:** `vitals_predictions`
- **Data Format:**
  ```json
  {
    "patient_id": "ICU-001",
    "timestamp": "2026-02-14T10:30:00Z",
    "deterioration_probability": 0.12,
    "risk_level": "LOW",
    "confidence": 0.94,
    "model_version": "lstm-v1.0",
    "features_used": 15,
    "processing_time_ms": 45
  }
  ```

#### Model Details:
- **Architecture:** LSTM with Attention mechanism
- **Input Features:** 15 vital sign and risk features
- **Sequence Length:** 12 time steps (1 hour of data)
- **Output:** Binary classification (deterioration probability)

---

### Stage 4: API Gateway
**Service:** backend-api
**Container:** icu-backend-api
**Port:** 8000

#### Function:
- REST API for external access
- WebSocket support for real-time updates
- Orchestrates communication between services
- Provides unified API interface

#### Endpoints:
- `GET /health` - Health check
- `GET /` - Service information
- (Additional endpoints can be added for patient data, predictions, etc.)

#### Integration:
- Connects to Kafka for data streaming
- Communicates with ML Service via HTTP
- Environment variables:
  * `KAFKA_BOOTSTRAP_SERVERS=kafka:9092`
  * `ML_SERVICE_URL=http://ml-service:8000`

---

## Infrastructure Services

### Zookeeper
**Container:** icu-zookeeper
**Port:** 2181

- Manages Kafka cluster metadata
- Handles leader election
- Required for Kafka operation

### Kafka Broker
**Container:** icu-kafka
**Ports:** 
- 9092 (internal)
- 29092 (external/host access)

- Message broker for decoupled communication
- Topics:
  * `vitals` - Raw vital signs data
  * `vitals_enriched` - Risk-analyzed data
  * `vitals_predictions` - ML predictions
- Auto-creates topics as needed
- Single broker configuration (suitable for development)

---

## Kafka Topics Overview

| Topic Name | Producer | Consumer | Data Type | Purpose |
|------------|----------|----------|-----------|---------|
| `vitals` | vital-simulator | pathway-engine | Raw vitals | Initial sensor data |
| `vitals_enriched` | pathway-engine | ml-service | Enriched vitals | Risk-analyzed data |
| `vitals_predictions` | ml-service | backend-api | Predictions | ML inference results |

---

## Network Configuration

All services communicate via the `icu-network` Docker bridge network:
- Enables secure inter-service communication
- DNS resolution by container name
- Isolated from external networks except exposed ports

**Exposed Ports:**
- `8000` - Backend API (REST/WebSocket)
- `8001` - ML Service (REST API)
- `29092` - Kafka (external access for testing)

---

## Data Processing Pipeline Summary

```
1. Vital Simulator generates patient data every second
   ↓
2. Data published to Kafka topic: vitals
   ↓
3. Pathway Engine consumes, enriches, and scores risk
   ↓
4. Enriched data published to Kafka topic: vitals_enriched
   ↓
5. ML Service consumes and applies LSTM model
   ↓
6. Predictions published to Kafka topic: vitals_predictions
   ↓
7. Backend API provides external access
   ↓
8. Healthcare providers receive real-time insights
```

---

## Performance Characteristics

- **End-to-End Latency:** ~50-100ms (vital generation to prediction)
- **Throughput:** 8 patients × 1 reading/second = 8 messages/sec
- **Processing Rate:** Real-time streaming (no batch delays)
- **Scalability:** Horizontally scalable via Kafka partitions

---

## Starting the System

### Build and Start All Services:
```bash
docker-compose up --build
```

### Start Without Rebuild:
```bash
docker-compose up
```

### Stop All Services:
```bash
docker-compose down
```

### View Logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f vital-simulator
docker-compose logs -f pathway-engine
docker-compose logs -f ml-service
docker-compose logs -f backend-api
```

### Check Service Health:
```bash
# Backend API
curl http://localhost:8000/health

# ML Service
curl http://localhost:8001/health
```

---

## Monitoring Kafka Topics

### Install Kafka tools (if not in containers):
```bash
# List topics
docker exec -it icu-kafka kafka-topics --bootstrap-server localhost:9092 --list

# View messages from vitals topic
docker exec -it icu-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals --from-beginning

# View enriched vitals
docker exec -it icu-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals_enriched --from-beginning

# View predictions
docker exec -it icu-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals_predictions --from-beginning
```

---

## Development Notes

### Service Dependencies:
1. **Zookeeper** → Must start first
2. **Kafka** → Depends on Zookeeper
3. **Vital Simulator** → Depends on Kafka
4. **Pathway Engine** → Depends on Kafka
5. **ML Service** → Can run independently but consumes from Kafka
6. **Backend API** → Depends on Kafka and ML Service

### Environment Variables:
All services use `KAFKA_BOOTSTRAP_SERVERS=kafka:9092` for internal communication.

### Restart Policies:
All services configured with `restart: unless-stopped` for resilience.

---

## Troubleshooting

### Service won't start:
1. Check Kafka is running: `docker ps | grep kafka`
2. Check logs: `docker-compose logs <service-name>`
3. Verify network: `docker network inspect icu-network`

### No data flowing:
1. Check vital-simulator is producing: `docker logs icu-vital-simulator`
2. Verify Kafka topics exist: `docker exec -it icu-kafka kafka-topics --list --bootstrap-server localhost:9092`
3. Monitor topic data: Use kafka-console-consumer (see above)

### High latency:
1. Check container resources: `docker stats`
2. Review pathway-engine logs for processing times
3. Verify ML model is loaded: Check ml-service startup logs

---

## Future Enhancements

1. **Database Integration:** PostgreSQL for historical data storage
2. **WebSocket Streaming:** Real-time dashboard updates
3. **Alert System:** Automated notifications for critical events
4. **Multi-Model Ensemble:** Combine multiple ML models
5. **Patient History:** Long-term trend analysis
6. **Clinical Decision Support:** Automated treatment recommendations

---

## Contact & Support

For issues, questions, or contributions, please refer to the main README.md file.
