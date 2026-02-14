# VitalX Frontend - ICU Digital Twin System

## Overview

VitalX is a comprehensive Real-Time ICU Digital Twin System built with React, TypeScript, and modern web technologies. The frontend is structured in 5 progressive phases, each adding critical features for ICU patient monitoring.

## Tech Stack

- **Framework**: React 18 with Vite
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Charts**: Recharts
- **Real-time Communication**: WebSocket API
- **HTTP Client**: Axios

## Phase Implementation

### Phase 1: Real-Time Vital Streaming ✅

**Objective**: Build clean, production-style real-time ICU monitoring dashboard

**Features**:
- Clean hospital-grade layout (Header, Sidebar, Main Panel, Footer)
- Live vital sign streaming via WebSocket
- Patient list with dynamic selection (P1-P8)
- Basic state visualization (STABLE, EARLY_DETERIORATION, CRITICAL, INTERVENTION)
- System health indicator
- Real-time vital cards for:
  - Heart Rate
  - Blood Pressure (Systolic/Diastolic)
  - SpO2
  - Respiratory Rate
  - Temperature
  - Shock Index
- Trend charts with rolling 50-value history

**Key Components**:
- `Header`: Logo, system status, timestamp
- `Sidebar`: Patient list with state badges
- `MainPanel`: Vital cards and trend charts
- `Footer`: WebSocket and backend service health
- `VitalCard`: Individual vital sign display
- `TrendChart`: Real-time vital trends

### Phase 2: Risk Score Visualization ✅

**Objective**: Integrate ML-driven deterioration prediction

**Features**:
- Risk Score Card (0-1 scale)
- Color-coded risk levels:
  - Green (0-0.4): LOW
  - Yellow (0.4-0.7): MODERATE
  - Red (0.7-0.85): HIGH
  - Red + Pulse (>0.85): CRITICAL
- Risk Trend Chart with threshold visualization
- Alert Banner for risk > 0.7
- Animated pulse for critical patients (>0.8)

**Key Components**:
- `RiskScoreCard`: ML risk score visualization
- `RiskTrendChart`: Historical risk trends
- `AlertBanner`: Warning/Critical alerts

### Phase 3: Anomaly Detection ✅

**Objective**: Real-time anomaly visualization from Pathway engine

**Features**:
- Anomaly Indicator Badge (flashing orange border)
- Event Timeline Panel (scrollable history)
- Multi-signal highlighting for anomalies
- Event filtering by type and time
- Hospital-grade subtle animations
- Anomaly type classification

**Key Components**:
- `AnomalyIndicator`: Visual anomaly alert
- `EventTimeline`: Chronological event display
- `AnomalyDetailsModal`: Detailed event information

### Phase 4: Interactive Clinical Workflows ✅

**Objective**: ICU-grade alert escalation and acknowledgment

**Features**:
- Three-tier alert system (INFO, WARNING, CRITICAL)
- Alert acknowledgment system with notes
- Alert Modal with full patient snapshot
- Alert history tracking
- Timer display (time since alert)
- Event filtering and export
- Data export functionality (CSV/JSON)

**Key Components**:
- `AlertAcknowledgment`: Alert management UI
- `EventFilter`: Advanced event filtering
- `DataExport`: Patient data export
- Alert state management in Zustand store

### Phase 5: Multi-Patient Dashboard & System Monitoring ✅

**Objective**: Enterprise-grade system monitoring and multi-patient view

**Features**:
- Multi-Patient Grid Dashboard
- Priority Queue (risk-based sorting)
- Comparative patient charts
- Dashboard statistics
- System Monitoring Panel:
  - Kafka Throughput Gauge (msgs/sec)
  - Stream Latency Indicator (ms)
  - ML Inference Time Card (ms)
  - Active Patient Load Meter
  - Service Health Grid
  - Performance Summary
- Dark theme monitoring panel
- Real-time metrics polling

**Key Components**:
- `MultiPatientDashboard`: Grid view of all patients
- `PatientGridCard`: Compact patient summary
- `PriorityQueue`: Risk-sorted patient list
- `ComparativeChart`: Multi-patient comparison
- `DashboardStats`: Overall statistics
- `SystemMonitoringPanel`: Infrastructure metrics

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── alerts/
│   │   │   └── AlertAcknowledgment.tsx
│   │   ├── anomaly/
│   │   │   ├── AnomalyIndicator.tsx
│   │   │   ├── EventTimeline.tsx
│   │   │   └── AnomalyDetailsModal.tsx
│   │   ├── dashboard/
│   │   │   ├── MultiPatientDashboard.tsx
│   │   │   ├── PatientGridCard.tsx
│   │   │   ├── PriorityQueue.tsx
│   │   │   ├── ComparativeChart.tsx
│   │   │   └── DashboardStats.tsx
│   │   ├── export/
│   │   │   └── DataExport.tsx
│   │   ├── filters/
│   │   │   └── EventFilter.tsx
│   │   ├── info/
│   │   │   └── PhaseInfo.tsx
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── MainPanel.tsx
│   │   │   └── Footer.tsx
│   │   ├── monitoring/
│   │   │   └── SystemMonitoringPanel.tsx
│   │   ├── risk/
│   │   │   ├── RiskScoreCard.tsx
│   │   │   ├── RiskTrendChart.tsx
│   │   │   └── AlertBanner.tsx
│   │   └── vitals/
│   │       ├── VitalCard.tsx
│   │       ├── TrendChart.tsx
│   │       └── Badges.tsx
│   ├── hooks/
│   │   ├── useHealthStatus.ts
│   │   └── useVitalStream.ts
│   ├── services/
│   │   ├── api.ts
│   │   └── websocket.ts
│   ├── store/
│   │   └── index.ts
│   ├── types/
│   │   └── index.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
└── README.md
```

## Backend API Contract

### REST Endpoints

**GET /health**
```json
{
  "status": "healthy",
  "kafka": "connected",
  "pathway": "initializing",
  "ml_service": "offline"
}
```

**GET /patients**
```json
{
  "patients": [
    { "patient_id": "P1", "state": "STABLE" },
    { "patient_id": "P2", "state": "CRITICAL" }
  ]
}
```

**GET /system/metrics** (Phase 5)
```json
{
  "kafka_throughput": 1200,
  "stream_latency_ms": 85,
  "ml_inference_time_ms": 42,
  "active_patients": 8
}
```

**POST /alerts/acknowledge** (Phase 4)
```json
{
  "patient_id": "P1",
  "note": "Administered medication"
}
```

### WebSocket Messages

**Connection**: `ws://localhost:8000/ws`

**Message Format**:
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
  "timestamp": "2026-02-14T10:00:00Z",
  "risk_score": 0.35,
  "risk_level": "LOW",
  "anomaly_detected": false,
  "anomaly_type": null
}
```

## Installation & Setup

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Install Dependencies

```bash
npm install
```

### Development Server

```bash
npm run dev
```

Application runs on `http://localhost:3000`

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## State Management

The application uses Zustand for global state management:

- **Patient Data**: Stores latest vitals and rolling history (50 values)
- **Risk History**: Tracks risk score trends
- **Alert Acknowledgments**: Manages acknowledged alerts
- **View Mode**: Toggles between single-patient and multi-patient views
- **Phase Selection**: Controls which features are visible
- **WebSocket Connection**: Tracks connection status
- **Health Status**: Backend service health

## Key Features

### Auto-Reconnection
WebSocket automatically reconnects on disconnection with 3-second delay

### Data Retention
Rolling window of last 50 vital readings per patient

### Responsive Design
Adapts to desktop and tablet screens

### Real-time Updates
- Vital signs: Instant via WebSocket
- Health status: Polled every 10 seconds
- System metrics: Polled every 5 seconds
- Patient list: Updated every 30 seconds

### Professional Styling
- Hospital-grade clean interface
- Minimal animations
- High contrast for readability
- Color-coded severity levels
- Accessibility compliant

## View Modes

### Single Patient View
- Select a patient from sidebar
- See detailed vital cards and trends
- Risk assessment and alerts
- Anomaly detection
- Event timeline
- Data export

### Multi-Patient Dashboard (Phase 5)
- Grid view of all active patients
- Priority queue (risk-sorted)
- Comparative vital trends
- System statistics
- Infrastructure monitoring

## Development Notes

### Type Safety
Strict TypeScript typing throughout
All API responses and WebSocket messages are typed

### Component Reusability
Components are modular and reusable
Props interfaces clearly defined

### Performance
- Memoized components where appropriate
- Efficient WebSocket message handling
- Optimized re-renders with Zustand

### Code Organization
- Domain-based folder structure
- Separation of concerns
- Custom hooks for complex logic

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Environment Variables

Create `.env` file:

```env
VITE_WS_URL=ws://localhost:8000/ws
VITE_API_URL=/api
```

## Troubleshooting

### WebSocket Won't Connect
- Check backend is running on port 8000
- Verify WebSocket URL in environment

### Vital Data Not Showing
- Confirm patient data is streaming via WebSocket
- Check browser console for errors

### Charts Not Rendering
- Ensure at least 2 data points exist in history
- Check Recharts is installed

## License

MIT

## Authors

VitalX Development Team - 2026
