# VitalX Architecture Diagrams

**Visual Reference for Streaming Refactor**

---

## 🏗️ SYSTEM ARCHITECTURE (Target State)

```mermaid
graph TD
    subgraph "Data Generation"
        VS[Vital Simulator<br/>Drift Model]
    end
    
    subgraph "Event Streaming"
        K1[Kafka: vitals_raw]
        K2[Kafka: vitals_enriched]
        K3[Kafka: vitals_predictions]
        K4[Kafka: alerts_stream]
    end
    
    subgraph "Pathway Engine"
        PE1[Feature Engineering<br/>Sliding Window]
        PE2[Streaming RAG Index<br/>Live Embeddings]
        PE3[Query API<br/>HTTP Endpoint]
    end
    
    subgraph "ML Pipeline"
        ML[ML Service<br/>Risk Authority]
    end
    
    subgraph "Backend Layer"
        BE1[Stream Merger<br/>Join Enriched + Predictions]
        BE2[WebSocket Handler]
        BE3[REST Endpoints]
        BE4[Chat Handler]
    end
    
    subgraph "Alert System"
        AL[Alert Engine<br/>Threshold Based]
    end
    
    subgraph "Frontend"
        FE1[Patient Dashboard]
        FE2[Risk Trend Charts]
        FE3[RAG Chat Panel]
    end
    
    VS -->|1 msg/sec/patient| K1
    K1 --> PE1
    PE1 --> K2
    PE1 -.->|event-driven| PE2
    K2 --> ML
    ML --> K3
    K3 --> BE1
    K2 --> BE1
    K3 --> AL
    AL --> K4
    BE1 --> BE2
    BE1 --> BE3
    BE4 --> PE3
    BE2 --> FE1
    BE3 --> FE1
    BE4 --> FE3
    FE1 --> FE2
```

---

## 🔄 DATA FLOW SEQUENCE

```mermaid
sequenceDiagram
    participant VS as Vital Simulator
    participant K1 as Kafka: vitals_raw
    participant PE as Pathway Engine
    participant K2 as Kafka: vitals_enriched
    participant ML as ML Service
    participant K3 as Kafka: vitals_predictions
    participant BE as Backend API
    participant FE as Frontend
    
    VS->>K1: Publish raw vitals<br/>{hr, sbp, spo2, shock_index}
    K1->>PE: Consume
    PE->>PE: Feature Engineering<br/>(rolling stats, deltas)
    PE->>PE: Update RAG Index<br/>(embed & store)
    PE->>K2: Publish enriched<br/>{+ rolling_mean_hr, + hr_delta}
    K2->>ML: Consume
    ML->>ML: Buffer sequence<br/>(60 timesteps)
    ML->>ML: predict(sequence)<br/>→ risk_score
    ML->>K3: Publish prediction<br/>{risk_score}
    K3->>BE: Consume
    K2->>BE: Consume
    BE->>BE: Merge streams<br/>(join by patient_id)
    BE->>FE: WebSocket push<br/>{vitals + risk_score}
    FE->>FE: Render dashboard<br/>& charts
```

---

## 🧩 SERVICE RESPONSIBILITY BREAKDOWN

```mermaid
graph LR
    subgraph "Vital Simulator"
        VS1[Physiological<br/>Baselines]
        VS2[Drift Model<br/>Brownian Motion]
        VS3[Probabilistic<br/>Deterioration]
    end
    
    subgraph "Pathway Engine"
        PE1[Feature<br/>Engineering]
        PE2[Streaming<br/>RAG Index]
        PE3[Query<br/>Interface]
    end
    
    subgraph "ML Service"
        ML1[Sequence<br/>Buffer]
        ML2[Risk<br/>Prediction]
        ML3[Sole<br/>Authority]
    end
    
    subgraph "Backend API"
        BE1[Stream<br/>Merger]
        BE2[WebSocket<br/>Handler]
        BE3[REST<br/>Endpoints]
        BE4[Chat<br/>Handler]
    end
    
    VS1 --> VS2 --> VS3
    PE1 --> PE2
    PE2 --> PE3
    ML1 --> ML2 --> ML3
    BE1 --> BE2
    BE1 --> BE3
    BE4 --> PE3
```

---

## 🔀 BEFORE vs AFTER DATA FLOW

### BEFORE (Problematic)

```mermaid
graph TD
    VS1[Vital Simulator<br/>STATE MACHINE]
    VS1 -->|"States: STABLE→CRITICAL"| K1[Kafka: vitals]
    K1 --> PE1[Pathway Engine<br/>CALCULATES RISK ❌]
    PE1 --> K2[Kafka: vitals_enriched<br/>with risk_score ❌]
    K2 --> ML1[ML Service<br/>DISCONNECTED ❌]
    K2 --> BE1[Backend API<br/>ALSO CALCULATES RISK ❌]
    BE1 --> FE1[Frontend<br/>HARDCODED STATES ❌]
    
    RAG1[RAG Service<br/>BATCH INDEXING ❌]
    K2 -.-> RAG1
    
    style PE1 fill:#ff9999
    style ML1 fill:#ff9999
    style BE1 fill:#ff9999
    style RAG1 fill:#ff9999
```

### AFTER (Streaming-First)

```mermaid
graph TD
    VS2[Vital Simulator<br/>DRIFT MODEL ✅]
    VS2 -->|"Gradual Changes"| K1[Kafka: vitals_raw]
    K1 --> PE2[Pathway Engine<br/>FEATURES ONLY ✅]
    PE2 --> K2[Kafka: vitals_enriched<br/>NO risk_score ✅]
    PE2 -.->|"Real-time"| RAG2[Streaming RAG Index<br/>INSIDE PATHWAY ✅]
    K2 --> ML2[ML Service<br/>SOLE RISK AUTHORITY ✅]
    ML2 --> K3[Kafka: vitals_predictions]
    K3 --> BE2[Backend API<br/>ORCHESTRATOR ONLY ✅]
    K2 --> BE2
    BE2 --> FE2[Frontend<br/>ML-DRIVEN ✅]
    BE2 <-.-> RAG2
    
    style PE2 fill:#99ff99
    style ML2 fill:#99ff99
    style BE2 fill:#99ff99
    style RAG2 fill:#99ff99
```

---

## 🎯 KAFKA TOPIC TOPOLOGY

```mermaid
graph LR
    subgraph "Topics"
        T1[vitals_raw<br/>8 partitions]
        T2[vitals_enriched<br/>8 partitions]
        T3[vitals_predictions<br/>8 partitions]
        T4[alerts_stream<br/>1 partition]
    end
    
    subgraph "Producers"
        P1[Vital Simulator]
        P2[Pathway Engine]
        P3[ML Service]
        P4[Alert Engine]
    end
    
    subgraph "Consumers"
        C1[Pathway Engine<br/>group: pathway]
        C2[ML Service<br/>group: ml-service]
        C3[Backend API<br/>group: backend-vitals]
        C4[Backend API<br/>group: backend-predictions]
        C5[Alert Engine<br/>group: alerts]
    end
    
    P1 --> T1
    P2 --> T2
    P3 --> T3
    P4 --> T4
    
    T1 --> C1
    T2 --> C2
    T2 --> C3
    T3 --> C4
    T3 --> C5
```

---

## 🔍 PATHWAY ENGINE INTERNALS

```mermaid
graph TD
    subgraph "Pathway Engine"
        IN[Kafka Input<br/>vitals_raw]
        
        subgraph "Feature Engineering Pipeline"
            W1[Sliding Window<br/>30-60 min]
            W2[Group by<br/>patient_id]
            W3[Compute Aggregates<br/>rolling_mean, deltas]
            W4[Anomaly Detection<br/>z-score]
        end
        
        subgraph "Streaming RAG"
            R1[Event to Text<br/>Structured Chunks]
            R2[Embedding Model<br/>sentence-transformers]
            R3[Vector Index<br/>Per-patient]
            R4[Expiry Manager<br/>3-hour window]
        end
        
        subgraph "Query Interface"
            Q1[HTTP Endpoint<br/>POST /query]
            Q2[Embed Query]
            Q3[Similarity Search]
            Q4[Return Context]
        end
        
        OUT[Kafka Output<br/>vitals_enriched]
        
        IN --> W1 --> W2 --> W3 --> W4 --> OUT
        OUT -.-> R1 --> R2 --> R3
        R4 -.-> R3
        Q1 --> Q2 --> Q3
        R3 --> Q3 --> Q4
    end
```

---

## 🤖 ML SERVICE PIPELINE

```mermaid
graph TD
    IN[Kafka Consumer<br/>vitals_enriched]
    
    subgraph "Sequence Management"
        B1[Extract Features<br/>hr, sbp, shock_index, etc.]
        B2[Patient Buffer<br/>deque maxlen=60]
        B3[Check Buffer Ready<br/>len >= 60?]
    end
    
    subgraph "Inference"
        I1[Get Sequence<br/>numpy array]
        I2[Placeholder Model<br/>predict function]
        I3[Risk Score<br/>float 0.0-1.0]
    end
    
    OUT[Kafka Producer<br/>vitals_predictions]
    
    IN --> B1 --> B2 --> B3
    B3 -->|Yes| I1 --> I2 --> I3 --> OUT
    B3 -->|No| B1
```

---

## 🌐 BACKEND API ARCHITECTURE

```mermaid
graph TD
    subgraph "Stream Merger Thread"
        SM1[Consume<br/>vitals_enriched]
        SM2[Consume<br/>vitals_predictions]
        SM3[In-Memory State<br/>per patient]
        SM4[Join by patient_id<br/>& timestamp]
    end
    
    subgraph "API Endpoints"
        WS[WebSocket /ws<br/>Real-time stream]
        REST1[GET /patients<br/>List all]
        REST2[GET /patients/:id/latest<br/>Current state]
        REST3[GET /patients/:id/history<br/>Time series]
        CHAT[POST /chat<br/>RAG-powered]
    end
    
    subgraph "External Calls"
        PE[Pathway Engine<br/>Query API]
        LLM[LLM Service<br/>OpenAI/Anthropic]
    end
    
    SM1 --> SM3
    SM2 --> SM3
    SM3 --> SM4
    SM4 --> WS
    SM4 --> REST1
    SM4 --> REST2
    SM4 --> REST3
    CHAT --> PE --> LLM --> CHAT
```

---

## 🎨 FRONTEND COMPONENT HIERARCHY

```mermaid
graph TD
    APP[App.tsx]
    
    subgraph "Dashboard Layout"
        DL[DashboardLayout]
        
        subgraph "Main View"
            PG[PatientGrid]
            PC1[PatientCard - P001]
            PC2[PatientCard - P002]
            PC3[PatientCard - ...]
        end
        
        subgraph "Patient Card Components"
            VS[VitalSigns<br/>HR, BP, SpO2]
            RS[RiskScore Display<br/>Large Number]
            RB[RiskBadge<br/>Color-coded]
            AB[AlertBanner<br/>If risk > threshold]
        end
        
        subgraph "Detail View"
            DD[PatientDetailDrawer]
            VTC[VitalsTrendChart<br/>Time series]
            RTC[RiskTrendChart<br/>ML scores]
            RAG[RAGChatPanel<br/>Ask questions]
        end
    end
    
    APP --> DL
    DL --> PG
    PG --> PC1 & PC2 & PC3
    PC1 --> VS & RS & RB & AB
    PC1 -.-> DD
    DD --> VTC & RTC & RAG
```

---

## 🔐 SECURITY & ISOLATION

```mermaid
graph TD
    subgraph "Multi-Tenancy (Patient Isolation)"
        P1[Patient P001<br/>Data Stream]
        P2[Patient P002<br/>Data Stream]
        P3[Patient P003<br/>Data Stream]
    end
    
    subgraph "Kafka Partitioning"
        K1[Partition 0<br/>P001 data]
        K2[Partition 1<br/>P002 data]
        K3[Partition 2<br/>P003 data]
    end
    
    subgraph "Pathway Processing"
        PE1[Window P001]
        PE2[Window P002]
        PE3[Window P003]
    end
    
    subgraph "RAG Indices"
        R1[Index P001<br/>Isolated]
        R2[Index P002<br/>Isolated]
        R3[Index P003<br/>Isolated]
    end
    
    P1 --> K1 --> PE1 --> R1
    P2 --> K2 --> PE2 --> R2
    P3 --> K3 --> PE3 --> R3
    
    style R1 fill:#e1f5ff
    style R2 fill:#e1f5ff
    style R3 fill:#e1f5ff
```

---

## 📈 SCALING ARCHITECTURE

```mermaid
graph TD
    subgraph "Development (Single Node)"
        D1[All Services<br/>on localhost]
        D2[8 patients<br/>1 msg/sec each]
        D3[Single Kafka broker]
    end
    
    subgraph "Production (Distributed)"
        P1[Load Balancer]
        
        subgraph "Backend Cluster"
            BE1[Backend Instance 1]
            BE2[Backend Instance 2]
            BE3[Backend Instance N]
        end
        
        subgraph "Kafka Cluster"
            K1[Broker 1]
            K2[Broker 2]
            K3[Broker 3]
        end
        
        subgraph "ML Service Cluster"
            ML1[ML Instance 1<br/>Patients 1-100]
            ML2[ML Instance 2<br/>Patients 101-200]
            ML3[ML Instance N<br/>Patients ...]
        end
        
        P1 --> BE1 & BE2 & BE3
        BE1 & BE2 & BE3 --> K1 & K2 & K3
        K1 & K2 & K3 --> ML1 & ML2 & ML3
    end
```

---

## 🔧 DEPLOYMENT ARCHITECTURE

```mermaid
graph TD
    subgraph "Docker Compose Stack"
        ZK[Zookeeper<br/>Port 2181]
        K[Kafka<br/>Port 29092]
        VS[Vital Simulator<br/>Internal]
        PE[Pathway Engine<br/>Port 8080]
        ML[ML Service<br/>Port 8001]
        BE[Backend API<br/>Port 8000]
        AL[Alert Engine<br/>Internal]
    end
    
    subgraph "External Access"
        LB[Nginx<br/>Port 80/443]
        FE[Frontend<br/>Static Files]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus<br/>Port 9090]
        GRAF[Grafana<br/>Port 3000]
    end
    
    ZK --> K
    K --> VS & PE & ML & AL
    PE --> BE
    ML --> BE
    LB --> BE
    LB --> FE
    BE --> PROM
    PROM --> GRAF
```

---

## 📊 MONITORING DASHBOARD LAYOUT

```mermaid
graph TD
    subgraph "Grafana Dashboards"
        subgraph "System Health"
            M1[Kafka Lag<br/>Per Consumer]
            M2[Service Uptime<br/>All Containers]
            M3[Error Rate<br/>Per Service]
        end
        
        subgraph "Performance"
            M4[Message Throughput<br/>msgs/sec]
            M5[Latency<br/>p50, p95, p99]
            M6[ML Inference Time<br/>per prediction]
        end
        
        subgraph "Business Metrics"
            M7[Active Patients<br/>Count]
            M8[Risk Score Distribution<br/>Histogram]
            M9[Alert Frequency<br/>Count over time]
        end
        
        subgraph "RAG Metrics"
            M10[Query Response Time<br/>milliseconds]
            M11[Index Size<br/>embeddings per patient]
            M12[Retrieval Quality<br/>relevance scores]
        end
    end
```

---

## 🎯 IMPLEMENTATION PHASES TIMELINE

```mermaid
gantt
    title VitalX Refactor Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1
    Vital Simulator Refactor      :p1, 2026-02-24, 3d
    section Phase 2
    Kafka Cleanup                  :p2, after p1, 1d
    section Phase 3
    Pathway Feature Engineering    :p3, after p2, 3d
    section Phase 4
    Pathway Streaming RAG          :p4, after p3, 4d
    section Phase 5
    Pathway Query API              :p5, after p4, 2d
    section Phase 6
    ML Service Refactor            :p6, after p3, 3d
    section Phase 7
    Backend API Refactor           :p7, after p5 p6, 4d
    section Phase 8
    Alert Engine                   :p8, after p6, 2d
    section Phase 9
    Frontend Cleanup               :p9, after p7, 3d
    section Phase 10
    Production Hardening           :p10, after p8 p9, 3d
```

---

**Note:** These diagrams are written in Mermaid syntax and will render in:
- GitHub markdown files
- VS Code with Mermaid extension
- Documentation sites (GitBook, Docusaurus, etc.)
- Confluence with Mermaid plugin

**Rendering Instructions:**
1. Install Mermaid extension in VS Code
2. Preview this markdown file
3. Diagrams will render interactively

**Export Options:**
- PNG: Use Mermaid CLI or online editor
- SVG: For high-quality documentation
- PDF: For presentations
