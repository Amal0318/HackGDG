// API service wrapper for VitalX monitoring
// Connects to backend API through Vite proxy (avoids CORS)

const API_BASE_URL = import.meta.env.VITE_API_URL || '';  // Empty = use Vite proxy

interface PatientData {
  patient_id: string;
  floor_id: string;
  timestamp: string;
  heart_rate: number;
  systolic_bp: number;
  diastolic_bp: number;
  spo2: number;
  shock_index: number;
  rolling_hr?: number;
  rolling_spo2?: number;
  rolling_sbp?: number;
  computed_risk?: number;
  anomaly_flag?: boolean;
  state?: string;
}

interface FloorData {
  id: string;
  name: string;
  capacity: number;
  current_patients: number;
  available_beds: number;
}

interface StatsOverview {
  total_patients: number;
  high_risk_count: number;
  anomaly_count: number;
  stable_patients: number;
  unstable_patients: number;
  floors_active: number;
  data_source: string;
}

// Generic fetch wrapper
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  
  // Merge with any provided headers
  if (options.headers) {
    Object.assign(headers, options.headers);
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
  }

  return response.json();
}

// Floors API
export const floorsAPI = {
  async getAll(): Promise<FloorData[]> {
    const response = await apiFetch<{floors: FloorData[]}>('/api/floors');
    return response.floors;
  },

  async getPatients(floorId: string): Promise<PatientData[]> {
    const response = await apiFetch<{patients: PatientData[]}>(`/api/floors/${floorId}/patients`);
    return response.patients;
  },
};

// Patients API
export const patientsAPI = {
  async getById(patientId: string): Promise<PatientData> {
    return apiFetch<PatientData>(`/api/patients/${patientId}`);
  },
  
  async getVitalsHistory(patientId: string, hours: number = 1): Promise<any> {
    const response = await apiFetch<any>(`/api/patients/${patientId}/history?hours=${hours}`);
    // Extract the history array from the response
    return response.history || [];
  },

  async getRiskHistory(patientId: string, hours: number = 1): Promise<any> {
    const response = await apiFetch<any>(`/api/patients/${patientId}/risk-history?hours=${hours}`);
    // Extract risk history from response
    return response.history || response.risk_history || [];
  },

  async generateReport(
    patientId: string, 
    options: { timeRangeHours?: number; includeAiSummary?: boolean } = {}
  ): Promise<Blob> {
    const { timeRangeHours = 3, includeAiSummary = true } = options;
    
    const response = await fetch(`${API_BASE_URL}/api/patients/${patientId}/reports/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        time_range_hours: timeRangeHours,
        include_ai_summary: includeAiSummary,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Report generation failed:', response.status, errorText);
      throw new Error(`Failed to generate report: ${response.statusText}`);
    }

    // Ensure we're getting a PDF
    const contentType = response.headers.get('content-type');
    if (!contentType || !contentType.includes('application/pdf')) {
      console.error('Unexpected content type:', contentType);
      const text = await response.text();
      console.error('Response body:', text);
      throw new Error('Server did not return a PDF');
    }

    return await response.blob();
  },
};

// Stats API
export const statsAPI = {
  async getOverview(): Promise<StatsOverview> {
    return apiFetch<StatsOverview>('/api/stats/overview');
  },
};

// Admin API (Requestly integration)
export const adminAPI = {
  async getRequestlyAnalytics() {
    return apiFetch('/api/admin/requestly/analytics');
  },

  async toggleMockMode(enabled: boolean) {
    return apiFetch('/api/admin/requestly/mock-mode', {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    });
  },
};

// Helper: Transform backend patient data to frontend format
export function transformPatientData(backendData: any) {
  // Calculate derived values - Use ?? instead of || to preserve 0 as valid value
  const rawRiskScore = backendData.risk_score ?? backendData.computed_risk ?? backendData.latest_risk_score ?? 0;
  const riskScore = rawRiskScore * 100; // Convert 0-1 to 0-100
  
  // Extract anomaly flags from features
  const features = backendData.features || {};
  const vitals = backendData.vitals || backendData;
  
  // Build abnormal_vitals array from anomaly flags
  const abnormalVitals: Array<{ vital: string; value: number; unit: string }> = [];
  
  if (features.hr_anomaly) {
    abnormalVitals.push({
      vital: 'Heart Rate',
      value: Math.round(vitals.heart_rate ?? backendData.heart_rate ?? 0),
      unit: 'bpm'
    });
  }
  
  if (features.sbp_anomaly) {
    abnormalVitals.push({
      vital: 'Systolic BP',
      value: Math.round(vitals.systolic_bp ?? backendData.systolic_bp ?? 0),
      unit: 'mmHg'
    });
  }
  
  if (features.spo2_anomaly) {
    abnormalVitals.push({
      vital: 'SpO2',
      value: Math.round(vitals.spo2 ?? backendData.spo2 ?? 0),
      unit: '%'
    });
  }
  
  if (features.shock_index_anomaly) {
    abnormalVitals.push({
      vital: 'Shock Index',
      value: Number((vitals.shock_index ?? backendData.shock_index ?? 0).toFixed(2)),
      unit: ''
    });
  }
  
  if (features.lactate_anomaly) {
    abnormalVitals.push({
      vital: 'Lactate',
      value: Number((vitals.lactate ?? backendData.lactate ?? 0).toFixed(2)),
      unit: 'mmol/L'
    });
  }
  
  return {
    patient_id: backendData.patient_id,
    id: backendData.patient_id, // Add camelCase alias
    name: backendData.name || `Patient ${backendData.patient_id}`,
    bed_number: backendData.bed_number || backendData.patient_id,
    bed: backendData.bed_number || backendData.patient_id, // Add short alias
    floor: typeof backendData.floor === 'number' ? backendData.floor : Number.parseInt((backendData.floor_id || 'ICU-1').replace(/\D/g, '') || '1', 10),
    latest_risk_score: riskScore,
    riskScore: riskScore, // Add camelCase alias for consistency
    risk_history: backendData.risk_history || [],
    abnormal_vitals: abnormalVitals.length > 0 ? abnormalVitals : (backendData.abnormal_vitals || []),
    anomaly_flag: features.anomaly_flag || false,
    age: backendData.age || 45, // Default age if not provided
    alerts: backendData.alerts || [], // Add alerts array
    vitals: {
      heart_rate: Math.round(vitals.heart_rate ?? backendData.heart_rate ?? backendData.rolling_hr ?? 75),
      systolic_bp: Math.round(vitals.systolic_bp ?? backendData.systolic_bp ?? backendData.rolling_sbp ?? 120),
      diastolic_bp: Math.round(vitals.diastolic_bp ?? backendData.diastolic_bp ?? (backendData.systolic_bp || 120) * 0.6),
      spo2: Math.round(vitals.spo2 ?? backendData.spo2 ?? backendData.rolling_spo2 ?? 98),
      respiratory_rate: Math.round(vitals.respiratory_rate ?? backendData.respiratory_rate ?? 16),
      temperature: Number((vitals.temperature ?? backendData.temperature ?? 37).toFixed(1)),
      lactate: Number((vitals.lactate ?? backendData.lactate ?? 1.0).toFixed(2)),
      shock_index: Number((vitals.shock_index ?? backendData.shock_index ?? 0.5).toFixed(2)),
    },
  };
}

async function apiRequest(endpoint: string, options: RequestInit = {}): Promise<any> {
  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
  
  const token = localStorage.getItem('token');
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };
  
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  const response = await fetch(url, {
    ...options,
    headers,
  });
  
  if (!response.ok) {
    throw new Error(`API request failed: ${response.statusText}`);
  }
  
  return response.json();
}

export const authAPI = {
  async login(credentials: { username: string; password: string }) {
    const response = await apiRequest('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    });
    
    if (response.access_token) {
      localStorage.setItem('token', response.access_token);
      localStorage.setItem('user', JSON.stringify(response.user));
    }
    
    return response.user;
  },
  
  logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },
  
  getCurrentUser() {
    const userStr = localStorage.getItem('user');
    return userStr ? JSON.parse(userStr) : null;
  }
};

// Alert acknowledgment API
export const alertAPI = {
  async acknowledgeAlert(patientId: string, acknowledgedBy: string = "Doctor"): Promise<any> {
    return apiFetch(`/api/patients/${patientId}/acknowledge-alert?acknowledged_by=${encodeURIComponent(acknowledgedBy)}`, {
      method: 'POST'
    });
  }
};

// RAG API - AI-powered patient handoff assistant
// Uses integrated RAG inside pathway-engine (via backend API)
const BACKEND_API_URL = 'http://localhost:8000';

export const ragAPI = {
  async query(question: string, patientId?: string): Promise<any> {
    // Use default patient ID if none provided (for general queries)
    const actualPatientId = patientId || 'P1';
    
    const response = await fetch(`${BACKEND_API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: actualPatientId,
        question: question
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend connection failed (${response.status}): ${errorText}`);
    }
    
    const data = await response.json();
    // Transform backend response to match expected format
    return {
      answer: data.answer,
      sources: data.sources || [],
      patient_id: data.patient_id
    };
  },

  async getSummary(patientId: string, hours: number = 4): Promise<any> {
    // Generate summary using chat endpoint
    const response = await fetch(`${BACKEND_API_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: patientId,
        question: `Provide a comprehensive clinical summary for the last ${hours} hours including current status, vital trends, and any concerns.`
      })
    });
    
    if (!response.ok) {
      throw new Error(`RAG summary failed: ${response.statusText}`);
    }
    
    const data = await response.json();
    return {
      summary: data.answer,
      patient_id: patientId,
      sources: data.sources || []
    };
  },

  async checkHealth(): Promise<any> {
    // Check backend API health (which includes pathway-engine RAG)
    const response = await fetch(`${BACKEND_API_URL}/health`);
    
    if (!response.ok) {
      throw new Error('RAG service unavailable');
    }
    
    const data = await response.json();
    return {
      status: data.status || 'healthy',
      rag_enabled: true,
      backend: 'pathway-engine-integrated'
    };
  }
};

export default {
  floors: floorsAPI,
  patients: patientsAPI,
  stats: statsAPI,
  admin: adminAPI,
  alerts: alertAPI,
  rag: ragAPI,
};
