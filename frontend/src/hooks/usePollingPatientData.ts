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

const MAX_HISTORY_POINTS = 450; // Keep last 450 points (15 minutes at 2s intervals)

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
        const timestamp = patient.timestamp || patient.last_updated || new Date().toISOString();

        // Add risk data point
        if (patient.computed_risk !== undefined || patient.risk_score !== undefined) {
          const riskValue = patient.computed_risk || patient.risk_score || 0;
          const riskPoint: RiskDataPoint = {
            timestamp,
            risk_score: riskValue
          };

          setRiskHistory(prev => {
            const currentHistory = prev[pid] || [];
            // Only add if value or timestamp changed (prevent exact duplicates)
            const lastEntry = currentHistory[currentHistory.length - 1];
            const isDifferent = !lastEntry || 
                              lastEntry.timestamp !== timestamp || 
                              lastEntry.risk_score !== riskValue;
            
            if (isDifferent) {
              console.log(`✅ Adding risk for ${pid}: ${riskValue.toFixed(6)}`);
              const updated = [...currentHistory, riskPoint];
              const trimmed = updated.slice(-MAX_HISTORY_POINTS);
              return { ...prev, [pid]: trimmed };
            }
            return prev;
          });
        }

        // Add vitals data point - check both nested and flat structures
        const vitals = patient.vitals || patient;
        if (vitals.heart_rate !== undefined || patient.rolling_hr !== undefined) {
          const vitalsPoint: VitalsDataPoint = {
            timestamp,
            heart_rate: vitals.heart_rate || patient.rolling_hr || 0,
            systolic_bp: vitals.systolic_bp || patient.rolling_sbp || 0,
            diastolic_bp: vitals.diastolic_bp || 80,
            spo2: vitals.spo2 || patient.rolling_spo2 || 0,
            respiratory_rate: vitals.respiratory_rate || patient.respiratory_rate || 16,
            temperature: vitals.temperature || patient.temperature || 37.0
          };

          setVitalsHistory(prev => {
            const currentHistory = prev[pid] || [];
            // Only add if any value or timestamp changed (prevent exact duplicates)
            const lastEntry = currentHistory[currentHistory.length - 1];
            const isDifferent = !lastEntry || 
                              lastEntry.timestamp !== timestamp ||
                              lastEntry.heart_rate !== vitalsPoint.heart_rate ||
                              lastEntry.systolic_bp !== vitalsPoint.systolic_bp ||
                              lastEntry.spo2 !== vitalsPoint.spo2;
            
            if (isDifferent) {
              console.log(`✅ Adding vitals for ${pid}: HR=${vitalsPoint.heart_rate}, BP=${vitalsPoint.systolic_bp}/${vitalsPoint.diastolic_bp}`);
              const updated = [...currentHistory, vitalsPoint];
              const trimmed = updated.slice(-MAX_HISTORY_POINTS);
              return { ...prev, [pid]: trimmed };
            }
            return prev;
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
