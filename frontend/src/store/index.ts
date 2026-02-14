import { create } from 'zustand';
import { VitalMessage, PatientData, Patient, HealthStatus } from '../types';

interface AlertAcknowledgment {
  patientId: string;
  timestamp: string;
  note: string;
}

interface AppState {
  patients: Map<string, PatientData>;
  patientList: Patient[];
  selectedPatientId: string | null;
  healthStatus: HealthStatus | null;
  wsConnected: boolean;
  
  // Phase 4 - Interactivity
  acknowledgedAlerts: Map<string, AlertAcknowledgment>;
  selectedEvent: VitalMessage | null;
  
  // Phase 5 - Multi-Patient Dashboard
  viewMode: 'single-patient' | 'multi-patient';
  selectedPhase: 'all' | 'phase1' | 'phase2' | 'phase3' | 'phase4' | 'phase5';
  
  setPatients: (patients: Patient[]) => void;
  setSelectedPatient: (patientId: string) => void;
  addVitalMessage: (message: VitalMessage) => void;
  setHealthStatus: (status: HealthStatus) => void;
  setWsConnected: (connected: boolean) => void;
  
  // Phase 4 actions
  acknowledgeAlert: (patientId: string, note: string) => void;
  clearAcknowledgment: (patientId: string) => void;
  setSelectedEvent: (event: VitalMessage | null) => void;
  
  // Phase 5 actions
  setViewMode: (mode: 'single-patient' | 'multi-patient') => void;
  setSelectedPhase: (phase: 'all' | 'phase1' | 'phase2' | 'phase3' | 'phase4' | 'phase5') => void;
}

const MAX_HISTORY = 50;

export const useStore = create<AppState>((set) => ({
  patients: new Map(),
  patientList: [],
  selectedPatientId: null,
  healthStatus: null,
  wsConnected: false,
  
  // Phase 4 initial state
  acknowledgedAlerts: new Map(),
  selectedEvent: null,
  
  // Phase 5 initial state
  viewMode: 'multi-patient',
  selectedPhase: 'all',

  setPatients: (patients: Patient[]) => set((state) => {
    const newPatientsMap = new Map(state.patients);
    
    patients.forEach(patient => {
      if (!newPatientsMap.has(patient.patient_id)) {
        newPatientsMap.set(patient.patient_id, {
          latest: null,
          history: [],
          riskHistory: [],
        });
      }
    });

    return {
      patientList: patients,
      patients: newPatientsMap,
      selectedPatientId: state.selectedPatientId || patients[0]?.patient_id || null,
    };
  }),

  setSelectedPatient: (patientId: string) => set({ selectedPatientId: patientId }),

  addVitalMessage: (message: VitalMessage) => set((state) => {
    const newPatientsMap = new Map(state.patients);
    const patientData = newPatientsMap.get(message.patient_id) || {
      latest: null,
      history: [],
      riskHistory: [],
    };

    const newHistory = [...patientData.history, message];
    if (newHistory.length > MAX_HISTORY) {
      newHistory.shift();
    }

    // Track risk score history if available
    let newRiskHistory = [...patientData.riskHistory];
    if (message.risk_score !== undefined) {
      newRiskHistory.push({
        timestamp: message.timestamp,
        risk_score: message.risk_score,
      });
      if (newRiskHistory.length > MAX_HISTORY) {
        newRiskHistory.shift();
      }
    }

    newPatientsMap.set(message.patient_id, {
      latest: message,
      history: newHistory,
      riskHistory: newRiskHistory,
    });

    return { patients: newPatientsMap };
  }),
  
  // Phase 4 actions
  acknowledgeAlert: (patientId: string, note: string) => set((state) => {
    const newAcknowledgments = new Map(state.acknowledgedAlerts);
    newAcknowledgments.set(patientId, {
      patientId,
      timestamp: new Date().toISOString(),
      note,
    });
    return { acknowledgedAlerts: newAcknowledgments };
  }),
  
  clearAcknowledgment: (patientId: string) => set((state) => {
    const newAcknowledgments = new Map(state.acknowledgedAlerts);
    newAcknowledgments.delete(patientId);
    return { acknowledgedAlerts: newAcknowledgments };
  }),
  
  setSelectedEvent: (event: VitalMessage | null) => set({ selectedEvent: event }),
  
  // Phase 5 actions
  setViewMode: (mode: 'single-patient' | 'multi-patient') => set({ viewMode: mode }),
  setSelectedPhase: (phase: 'all' | 'phase1' | 'phase2' | 'phase3' | 'phase4' | 'phase5') => set({ selectedPhase: phase }),

  setHealthStatus: (status: HealthStatus) => set({ healthStatus: status }),
  
  setWsConnected: (connected: boolean) => set({ wsConnected: connected }),
}));
