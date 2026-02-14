# VitalX - Quick Start Guide

## 🚀 Running the Application

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Application will be available at: **http://localhost:3000**

## 📊 Phase Overview

### 🏥 Phase 1: Real-Time Vitals
- **View**: Single Patient Monitor
- **Features**: 
  - Live vital sign streaming
  - 6 vital cards (HR, BP, SpO2, RR, Temp, Shock Index)
  - Real-time trend charts
  - Patient selection from sidebar
- **Status**: Core monitoring functionality

### ⚠️ Phase 2: Risk Assessment
- **View**: Single Patient Monitor
- **Features**:
  - ML Risk Score Card (0-1 scale)
  - Risk level classification (LOW/MODERATE/HIGH/CRITICAL)
  - Risk trend visualization
  - Automatic alert banners for high risk
  - Animated pulse for critical patients
- **Triggers**: When `risk_score` field present in data

### 🔍 Phase 3: Anomaly Detection
- **View**: Single Patient Monitor
- **Features**:
  - Anomaly indicator badges
  - Event timeline with anomaly tracking
  - Click events for detailed modal
  - Multi-signal highlighting
- **Triggers**: When `anomaly_detected: true` in data

### ⚡ Phase 4: Clinical Workflows
- **View**: Single Patient Monitor
- **Features**:
  - Alert acknowledgment system
  - Event filtering (by type, state, date)
  - Data export (CSV/JSON)
  - Alert history tracking
- **Actions**: 
  - Click "Acknowledge Alert" for high-risk patients
  - Filter events using the filter panel
  - Export patient data using export component

### 👥 Phase 5: Multi-Patient & Monitoring
- **View**: Multi-Patient Dashboard
- **Features**:
  - Grid view of all active patients
  - Priority queue (risk-sorted)
  - Comparative vital charts
  - System metrics monitoring:
    - Kafka throughput
    - Stream latency
    - ML inference time
    - Active patient count
  - Service health grid
- **Toggle**: Use "👥 Dashboard" button in header

## 🎛️ View Controls

### Header Navigation
- **👥 Dashboard**: Multi-patient grid + system monitoring
- **👤 Single Patient**: Detailed patient view

### Phase Tabs (Single Patient View)
- **🎯 All Phases**: Complete feature set
- **📊 Phase 1**: Vitals only
- **⚠️ Phase 2**: Vitals + Risk
- **🔍 Phase 3**: Vitals + Risk + Anomalies
- **⚡ Phase 4**: All interactive features
- **👥 Phase 5**: Switches to multi-patient view

## 📡 Required Backend Endpoints

### REST API
- `GET /health` - System health status
- `GET /patients` - Patient list
- `GET /system/metrics` - Infrastructure metrics (Phase 5)
- `POST /alerts/acknowledge` - Acknowledge alerts (Phase 4)

### WebSocket
- `ws://localhost:8000/ws` - Real-time vital stream

## 🎨 Key UI Elements

### Color Coding
- **Green/Teal**: Normal/Stable
- **Yellow**: Warning/Moderate risk
- **Red**: Critical/High risk
- **Orange**: Anomaly detected
- **Purple**: Intervention state

### Patient States
- **STABLE**: Green indicator
- **EARLY_DETERIORATION**: Yellow indicator
- **CRITICAL**: Red indicator
- **INTERVENTION**: Purple indicator

### Alert Levels
- **INFO**: Blue (risk < 0.4)
- **WARNING**: Yellow (risk 0.7-0.85)
- **CRITICAL**: Red (risk > 0.85)

## 🔧 Configuration

### Environment Variables (.env)
```env
VITE_WS_URL=ws://localhost:8000/ws
```

### Vite Proxy (vite.config.ts)
```typescript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true
    }
  }
}
```

## 📱 Responsive Breakpoints

- **Mobile**: < 768px (limited support)
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px (optimal)

## 🐛 Common Issues

### 1. WebSocket Not Connecting
**Solution**: Ensure backend running on port 8000

### 2. No Patient Data
**Solution**: Check backend streaming via WebSocket

### 3. Charts Empty
**Solution**: Need at least 2 data points in history

### 4. Can't See Phase Features
**Solution**: Toggle phase tabs or check if backend sends required fields

## 💡 Tips

1. **Start with All Phases**: Toggle to "🎯 All Phases" for full experience
2. **Watch the Dashboard**: Use multi-patient view to monitor all patients
3. **Check System Health**: Footer shows service status
4. **Monitor Performance**: Phase 5 dashboard shows system metrics
5. **Acknowledge Alerts**: Clear alert banners by acknowledging them

## 📊 Sample Data Requirements

### Minimum (Phase 1)
```json
{
  "patient_id": "P1",
  "heart_rate": 82,
  "systolic_bp": 118,
  "diastolic_bp": 76,
  "spo2": 97,
  "respiratory_rate": 16,
  "temperature": 37.2,
  "shock_index": 0.69,
  "state": "STABLE",
  "event_type": "NONE",
  "timestamp": "2026-02-14T10:00:00Z"
}
```

### Full Features (All Phases)
```json
{
  ...vitals,
  "risk_score": 0.82,
  "risk_level": "HIGH",
  "anomaly_detected": true,
  "anomaly_type": "HEMODYNAMIC_INSTABILITY"
}
```

## 🚦 Startup Checklist

- [ ] Backend API running on port 8000
- [ ] WebSocket endpoint available
- [ ] Kafka streaming patient data
- [ ] ML service providing risk scores
- [ ] Pathway engine detecting anomalies
- [ ] Frontend dev server on port 3000
- [ ] Browser console shows no errors

## 📞 Support

Check the main README.md for detailed documentation on each phase and component architecture.
