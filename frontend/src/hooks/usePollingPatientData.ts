// HTTP Polling hook - Fetches patient data continuously without WebSocket
import { useState, useEffect, useCallback, useRef } from 'react';
import { floorsAPI } from '../services/api';

interface RiskDataPoint {
  timestamp: string;
  risk_score: number;
}

interface VitalsDataPoint {
  timestamp: string;
  heart_rate: number;
  systolic_bp: number;
  diastolic_bp: number;
  spo2: number;
  respiratory_rate: number;
  temperature: number;
}

const MAX_HISTORY_POINTS = 60;

export function usePollingPatientData(_patientId?: string, pollInterval: number = 2000) {
  const [riskHistory, setRiskHistory] = useState<{ [key: string]: RiskDataPoint[] }>({});
  const [vitalsHistory, setVitalsHistory] = useState<{ [key: string]: VitalsDataPoint[] }>({});
  const [isPolling, setIsPolling] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  const fetchPatientData = useCallback(async () => {
    try {
      // Fetch all patients from all floors
      const floors = await floorsAPI.getAll();
      const allPatientsPromises = floors.map(floor => floorsAPI.getPatients(floor.id));
      const allPatientsArrays = await Promise.all(allPatientsPromises);
      const allPatients = allPatientsArrays.flat();

      console.log(`📊 HTTP Polling: Fetched ${allPatients.length} patients`);

      // Process each patient
      allPatients.forEach((patient: any) => {
        const pid = patient.patient_id;
        const timestamp = patient.timestamp || new Date().toISOString();

        // Add risk data point
        if (patient.computed_risk !== undefined || patient.risk_score !== undefined) {
          const riskValue = patient.computed_risk || patient.risk_score || 0;
          console.log(`📈 Patient ${pid} risk: ${riskValue}`);

          const riskPoint: RiskDataPoint = {
            timestamp,
            risk_score: riskValue
          };

          setRiskHistory(prev => {
            const currentHistory = prev[pid] || [];
            const updated = [...currentHistory, riskPoint];
            const trimmed = updated.slice(-MAX_HISTORY_POINTS);
            return { ...prev, [pid]: trimmed };
          });
        }

        // Add vitals data point
        if (patient.heart_rate !== undefined || patient.rolling_hr !== undefined) {
          const vitalsPoint: VitalsDataPoint = {
            timestamp,
            heart_rate: patient.heart_rate || patient.rolling_hr || 0,
            systolic_bp: patient.systolic_bp || patient.rolling_sbp || 0,
            diastolic_bp: patient.diastolic_bp || 80,
            spo2: patient.spo2 || patient.rolling_spo2 || 0,
            respiratory_rate: patient.respiratory_rate || 16,
            temperature: patient.temperature || 37.0
          };

          setVitalsHistory(prev => {
            const currentHistory = prev[pid] || [];
            const updated = [...currentHistory, vitalsPoint];
            const trimmed = updated.slice(-MAX_HISTORY_POINTS);
            return { ...prev, [pid]: trimmed };
          });
        }
      });

      setIsPolling(true);
    } catch (error) {
      console.error('❌ Polling error:', error);
    }
  }, []);

  useEffect(() => {
    // Initial fetch
    fetchPatientData();

    // Start polling
    intervalRef.current = setInterval(fetchPatientData, pollInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [fetchPatientData, pollInterval]);

  // Get data for specific patient
  const getPatientRiskHistory = useCallback((pid: string) => {
    return riskHistory[pid] || [];
  }, [riskHistory]);

  const getPatientVitalsHistory = useCallback((pid: string) => {
    return vitalsHistory[pid] || [];
  }, [vitalsHistory]);

  return {
    isPolling,
    riskHistory,
    vitalsHistory,
    getPatientRiskHistory,
    getPatientVitalsHistory
  };
}
