# ICU Digital Twin System - Quick Start Guide

## Current System Status

### ✅ Running Services (5/6)

1. **✅ Zookeeper** (icu-zookeeper)
   - Status: Running
   - Port: 2181 (internal)
   - Purpose: Kafka cluster management

2. **✅ Kafka Broker** (icu-kafka)
   - Status: Running
   - Ports: 9092 (internal), 29092 (external)
   - Purpose: Message streaming broker
   - Topics: `vitals`, `vitals_enriched`, `vitals_predictions`

3. **✅ Vital Simulator** (icu-vital-simulator)
   - Status: Running
   - Purpose: Generates patient vital signs data
   - Output: Produces to Kafka topic `vitals`

4. **✅ ML Service** (icu-ml-service)
   - Status: Running
   - Port: 8001 → 8000
   - Purpose: LSTM-based deterioration prediction
   - Input: Consumes from `vitals_enriched`
   - Output: Produces to `vitals_predictions`

5. **✅ Backend API** (icu-backend-api)
   - Status: Running
   - Port: 8000
   - Purpose: REST/WebSocket API gateway
   - Access: http://localhost:8000

6. **⏳ Pathway Engine** (icu-pathway-engine)
   - Status: Building (in progress)
   - Purpose: Real-time risk scoring and enrichment
   - Input: Consumes from `vitals`
   - Output: Produces to `vitals_enriched`

---

## Current Data Flow (Partial)

```
┌──────────────┐      ┌────────┐      ┌──────────────┐
│    Vital     │─────▶│ Kafka  │      │   Pathway    │
│  Simulator   │      │ vitals │      │    Engine    │
│   (Running)  │      │(topic) │      │  (Building)  │
└──────────────┘      └────────┘      └──────────────┘
                           │
                           ▼
                      (waiting for
                       Pathway to
                        consume)
```

---

## Quick Commands

### Using the Helper Script:
```cmd
docker-helper.bat status    - Check running containers
docker-helper.bat logs      - View all logs
docker-helper.bat logs kafka - View Kafka logs
docker-helper.bat restart   - Restart services
docker-helper.bat down      - Stop everything
```

### Manual Docker Commands:
```cmd
REM Set PATH first
set "PATH=%PATH%;C:\Program Files\Docker\Docker\resources\bin"

REM Check running containers
docker ps

REM View logs
docker logs -f icu-kafka
docker logs -f icu-vital-simulator
docker logs -f icu-ml-service
docker logs -f icu-backend-api

REM View Kafka topics
docker exec -it icu-kafka kafka-topics --bootstrap-server localhost:9092 --list

REM Consume messages from vitals topic
docker exec -it icu-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals --from-beginning

REM Check health
curl http://localhost:8000/health
curl http://localhost:8001/health
```

---

## Next Steps

### 1. Wait for Pathway Engine to Build
The pathway-engine is currently building. This may take 5-10 minutes due to:
- Installing Rust compiler
- Compiling Pathway framework
- Installing dependencies

### 2. Start Pathway Engine
Once built, start it with:
```cmd
set "PATH=%PATH%;C:\Program Files\Docker\Docker\resources\bin"
docker compose up -d pathway-engine
```

### 3. Verify Complete Data Flow
Once pathway-engine is running:
```cmd
docker ps
```
You should see all 6 containers running.

### 4. Monitor Data Flow
```cmd
REM Terminal 1: Watch raw vitals
docker exec -it icu-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals

REM Terminal 2: Watch enriched vitals (after Pathway starts)
docker exec -it icu-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals_enriched

REM Terminal 3: Watch predictions
docker exec -it icu-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals_predictions
```

---

## Complete Data Flow (When All Services Running)

```
1. Vital Simulator → Generates patient data
                ↓
2. Kafka Topic: vitals → Raw vital signs
                ↓
3. Pathway Engine → Risk scoring & enrichment
                ↓
4. Kafka Topic: vitals_enriched → Risk-analyzed data
                ↓
5. ML Service → LSTM prediction
                ↓
6. Kafka Topic: vitals_predictions → Deterioration predictions
                ↓
7. Backend API → REST/WebSocket access
                ↓
8. External Clients
```

---

## Troubleshooting

### Pathway Engine Taking Too Long?
The Pathway engine requires Rust compilation. If it's taking more than 15 minutes:
1. Check build progress: `docker-helper.bat logs pathway-engine`
2. Or run without Pathway for now (other services work independently)

### Service Not Starting?
```cmd
docker logs icu-<service-name>
```

### Kafka Not Receiving Messages?
```cmd
docker logs icu-vital-simulator
docker exec -it icu-kafka kafka-topics --bootstrap-server localhost:9092 --describe
```

### Reset Everything
```cmd
docker-helper.bat down
docker-helper.bat build
docker-helper.bat up
```

---

## Testing the System

### Test 1: Backend API Health
```cmd
curl http://localhost:8000/health
```
Expected: `{"status":"healthy","service":"backend-api"}`

### Test 2: ML Service Health
```cmd
curl http://localhost:8001/health
```
Expected: Model status and configuration

### Test 3: Kafka Topics
```cmd
docker exec -it icu-kafka kafka-topics --bootstrap-server localhost:9092 --list
```
Expected: `vitals`, `vitals_enriched`, `vitals_predictions`

### Test 4: Live Data Flow
```cmd
docker exec -it icu-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic vitals --max-messages 5
```
Expected: 5 JSON messages with patient vital signs

---

## Service URLs

- **Backend API**: http://localhost:8000
- **Backend API Docs**: http://localhost:8000/docs
- **ML Service**: http://localhost:8001
- **ML Service Docs**: http://localhost:8001/docs
- **Kafka**: localhost:29092 (external access)

---

## System Architecture Reference

See [DATA_FLOW.md](DATA_FLOW.md) for detailed architecture documentation.

---

Last Updated: 2026-02-14
System Status: 5/6 Services Running (Pathway Engine building)
