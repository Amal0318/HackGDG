export type PatientState = 'STABLE' | 'EARLY_DETERIORATION' | 'CRITICAL' | 'INTERVENTION';
export type EventType = 'NONE' | 'HYPOTENSION' | 'TACHYCARDIA' | 'HYPOXIA' | 'SEPSIS_ALERT';

export interface VitalMessage {
  patient_id: string;
  heart_rate: number;
  systolic_bp: number;
  diastolic_bp: number;
  spo2: number;
  respiratory_rate: number;
  temperature: number;
  shock_index: number;
  state: PatientState;
  event_type: EventType;
  timestamp: string;
  // Future fields (Phase 2+)
  risk_score?: number;
  risk_level?: string;
  anomaly_detected?: boolean;
  anomaly_type?: string;
}

export interface Patient {
  patient_id: string;
  state: PatientState;
}

export interface PatientData {
  latest: VitalMessage | null;
  history: VitalMessage[];
  riskHistory: Array<{ timestamp: string; risk_score: number }>;
}

export interface HealthStatus {
  status: string;
  kafka: string;
  pathway: string;
  ml_service: string;
  timestamp?: string;
}

export interface PatientsResponse {
  patients: Patient[];
}

export interface SystemMetrics {
  kafka_throughput: number;
  stream_latency_ms: number;
  ml_inference_time_ms: number;
  active_patients: number;
}
