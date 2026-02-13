# ICU Digital Twin & Deterioration Prediction System

## Phase 0: Infrastructure Setup & Service Scaffolding

This is a clean, production-style microservice architecture skeleton for a Real-Time ICU Digital Twin & Deterioration Prediction System.

## Architecture Overview

```
icu-system/
├── vital-simulator    → ICU telemetry data generator
├── pathway-engine     → Real-time streaming processor  
├── ml-service         → ML inference engine
├── backend-api        → REST + WebSocket API gateway
├── frontend          → React dashboard (placeholder)
├── kafka             → Message broker
└── zookeeper         → Kafka coordination
```

## Quick Start

1. **Start the entire system:**
   ```bash
   cd icu-system
   docker-compose up --build
   ```

2. **Verify services are running:**
   - Backend API: http://localhost:8000/health
   - ML Service: http://localhost:8001/health
   - Kafka UI: http://localhost:29092 (if needed)

3. **View logs:**
   ```bash
   docker-compose logs -f [service-name]
   ```

4. **Stop services:**
   ```bash
   docker-compose down
   ```

## Service Ports

- **backend-api**: 8000 (External REST + WebSocket API)
- **ml-service**: 8001 (ML inference endpoints)
- **frontend**: 3000 (Reserved for React dashboard)
- **kafka**: 29092 (External Kafka access)

## Service Dependencies

- `zookeeper` → Base dependency
- `kafka` → Depends on Zookeeper
- `vital-simulator` → Depends on Kafka
- `pathway-engine` → Depends on Kafka  
- `ml-service` → Standalone FastAPI service
- `backend-api` → Depends on Kafka + ML Service

## Development Notes

- All services start successfully without errors
- No business logic implemented (Phase 0 only)
- Clean Docker networking with isolated services
- Hackathon-friendly Kafka configuration
- Professional production-style structure

## Next Phases

- **Phase 1**: Implement ICU data simulation
- **Phase 2**: Add ML model loading & inference
- **Phase 3**: Real-time streaming & digital twin logic  
- **Phase 4**: React dashboard & visualization