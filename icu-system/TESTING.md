# VitalX - Quick Testing Commands

## Start System

```bash
cd icu-system
docker-compose up -d
```

## Verify Data Flow

### 1. Check Topics Created
```bash
docker exec icu-kafka kafka-topics --bootstrap-server kafka:9092 --list
```

Expected: vitals_raw, vitals_enriched, vitals_predictions, alerts_stream

### 2. Watch Raw Vitals (Simulator Output)
```bash
docker exec icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_raw \
  --from-beginning \
  --max-messages 3
```

### 3. Watch Enriched Vitals (Pathway Output)  
```bash
docker exec icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_enriched \
  --from-beginning \
  --max-messages 3
```

### 4. Watch Predictions (ML Service Output)
```bash
docker exec icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic vitals_predictions \
  --from-beginning \
  --max-messages 3
```

### 5. Watch Alerts (Alert Engine Output)
```bash
docker exec icu-kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 \
  --topic alerts_stream \
  --from-beginning
```

## API Testing

### Backend Health
```bash
curl http://localhost:8000/health
```

### List Patients
```bash
curl http://localhost:8000/patients | jq
```

### Get Patient Details
```bash
curl http://localhost:8000/patients/P001 | jq
```

### Get Patient History
```bash
curl http://localhost:8000/patients/P001/history?hours=1 | jq
```

### Chat Query (RAG)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "query": "What is the heart rate trend?"
  }' | jq
```

### Pathway RAG Health
```bash
curl http://localhost:8080/health | jq
```

### Pathway RAG Stats
```bash
curl http://localhost:8080/stats | jq
```

### Query RAG Directly
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "query_text": "blood pressure changes",
    "top_k": 5
  }' | jq
```

## View Logs

### All Services
```bash
docker-compose logs -f
```

### Specific Service
```bash
docker-compose logs -f vital-simulator
docker-compose logs -f pathway-engine  
docker-compose logs -f ml-service
docker-compose logs -f backend-api
docker-compose logs -f alert-engine
```

### Filter Logs
```bash
# Simulator status
docker-compose logs vital-simulator | grep "Iteration"

# Pathway features
docker-compose logs pathway-engine | grep "feature"

# ML predictions
docker-compose logs ml-service | grep "Predicted"

# Alerts triggered
docker-compose logs alert-engine | grep "ALERT TRIGGERED"
```

## Monitor Performance

### Container Stats
```bash
docker stats
```

### Kafka Consumer Groups
```bash
docker exec icu-kafka kafka-consumer-groups \
  --bootstrap-server kafka:9092 \
  --list

docker exec icu-kafka kafka-consumer-groups \
  --bootstrap-server kafka:9092 \
  --describe \
  --group pathway-engine
```

## WebSocket Testing

### Using wscat
```bash
npm install -g wscat
wscat -c ws://localhost:8000/ws
```

### Using Python
```python
import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(json.dumps(data, indent=2))

asyncio.run(test_websocket())
```

## Troubleshooting

### Restart Service
```bash
docker-compose restart vital-simulator
```

### Rebuild Service
```bash
docker-compose up -d --build vital-simulator
```

### Check Kafka Connection
```bash
docker exec icu-vital-simulator ping kafka
```

### Check Service Health
```bash
docker exec icu-vital-simulator ps aux
```

## Cleanup

### Stop Services
```bash
docker-compose down
```

### Full Cleanup (Delete Data)
```bash
docker-compose down -v
```

### Remove Specific Service
```bash
docker-compose stop vital-simulator
docker-compose rm vital-simulator
```

## Expected Data Flow Timeline

```
T=0s:    System starts, Kafka topics created
T=1s:    Simulator publishes first vitals to vitals_raw
T=2s:    Pathway consumes vitals_raw, starts feature computation
T=30s:   Pathway sliding window has enough data, publishes to vitals_enriched
T=60s:   ML Service buffer full (60 samples), starts predictions
T=61s:   First prediction published to vitals_predictions
T=62s:   Backend receives merged data, broadcasts via WebSocket
T=63s:   Alert engine monitors predictions, triggers alerts if risk_score > 0.75
```

## Validation Checklist

- [ ] All containers running (`docker-compose ps`)
- [ ] All 4 Kafka topics exist
- [ ] vitals_raw has messages (simulator working)
- [ ] vitals_enriched has messages with features (pathway working)
- [ ] vitals_predictions has risk_score (ML working)
- [ ] Backend /health returns 200
- [ ] Backend /patients returns patient list
- [ ] WebSocket connection established and receiving updates
- [ ] Pathway RAG /health returns statistics
- [ ] Chat endpoint returns contextual responses
- [ ] Alerts triggered when risk_score > threshold
