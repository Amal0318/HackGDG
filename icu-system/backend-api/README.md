# Backend API - ICU Monitoring System

## 🎯 Features

### ✅ Authentication System
- JWT-based authentication
- Role-based access control (Admin, Doctor, Nurse)
- Secure token management

### 🏥 Multi-Floor Management
- 3 ICU floors (1F, 2F, 3F)
- 8 patients per floor (24 total)
- Real-time patient monitoring

### 🔧 Requestly Integration (Sponsor Feature)
- API request monitoring and analytics
- Mock server fallback for resilience
- Request/Response interception
- Session recording capabilities

### 📊 Real-time Data Streaming
- Kafka integration for live patient data
- Per-patient vital signs and ML predictions
- Automatic floor assignment based on patient ID

---

## 🚀 Quick Start

### Demo Credentials

```
Admin:
  username: admin
  password: admin123

Doctor:
  username: doctor
  password: doctor123

Nurse:
  username: nurse
  password: nurse123
```

### API Endpoints

#### Authentication
```http
POST /api/auth/login
GET  /api/auth/me
```

#### Floors & Patients
```http
GET /api/floors
GET /api/floors/{floor_id}/patients
GET /api/patients/{patient_id}
```

#### System Stats
```http
GET /api/stats/overview
GET /health
```

#### Admin - Requestly Features
```http
GET  /api/admin/requestly/analytics
POST /api/admin/requestly/mock-mode
```

---

## 🔧 Environment Variables

```env
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
SECRET_KEY=your-secret-key-change-in-production
ML_SERVICE_URL=http://ml-service:8000
```

---

## 📖 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🎭 Requestly Mock Mode

Enable mock mode for testing without Kafka:

```bash
curl -X POST "http://localhost:8000/api/admin/requestly/mock-mode?enable=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 🏗️ Architecture

```
Frontend → Backend API → Kafka → Pathway Engine → ML Service
             ↓
        Requestly
       (Monitoring)
```

---

## 📦 Docker Deployment

```bash
# Build and start
docker compose up -d backend-api

# View logs
docker logs icu-backend-api -f

# Check health
curl http://localhost:8000/health
```

---

## 🎯 Sponsor Integration

This backend showcases **Requestly API integration** including:

1. **API Monitoring**: All requests logged and tracked
2. **Mock Server**: Fallback mode when Kafka is unavailable
3. **Analytics Dashboard**: Admin endpoint for request analytics
4. **Resilience**: Graceful degradation with mock data

---

## 📝 Patient ID Format

```
P{floor}-{number}
Examples:
  P1-001, P1-002, ... P1-008  → Floor 1F
  P2-001, P2-002, ... P2-008  → Floor 2F
  P3-001, P3-002, ... P3-008  → Floor 3F
```

---

## 🔐 Security Notes

- JWT tokens expire after 8 hours (configurable)
- CORS enabled for frontend access
- In production: Use environment secrets for SECRET_KEY
- In production: Hash passwords with bcrypt

---

## 🐛 Troubleshooting

**Kafka connection fails:**
- Backend automatically enables Requestly mock mode
- Check `requestly_mock_mode` in `/health` endpoint

**Authentication fails:**
- Verify credentials match demo users
- Check token expiration

**No patient data:**
- Ensure vital-simulator is running
- Check Pathway engine is processing
- Enable mock mode as fallback

---

## 📈 Next Steps

1. Connect frontend dashboard
2. Add WebSocket support for real-time updates
3. Implement database persistence
4. Add alert notification system
5. Deploy to Railway for cloud access
