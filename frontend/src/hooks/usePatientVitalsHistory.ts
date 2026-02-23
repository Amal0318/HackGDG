// Hook for managing patient vitals history with WebSocket updates
import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';
import { patientsAPI } from '../services/api';

interface VitalsDataPoint {
  timestamp: string;
  heart_rate: number;
  systolic_bp: number;
  diastolic_bp: number;
  spo2: number;
  respiratory_rate: number;
  temperature: number;
}

interface PatientVitalsHistory {
  [patientId: string]: VitalsDataPoint[];
}

const MAX_VITALS_HISTORY_POINTS = 450; // Keep last 450 data points (15 minutes at 2-second intervals)

export function usePatientVitalsHistory(patientId?: string) {
  const [vitalsHistory, setVitalsHistory] = useState<PatientVitalsHistory>({});
  const [isLive, setIsLive] = useState(false);

  const handleWebSocketMessage = useCallback((message: any) => {
    if (message.type === 'patient_update' || message.type === 'floor_update') {
      const pid = message.patient_id;
      const data = message.data;

      if (pid && data && (data.heart_rate !== undefined || data.rolling_hr !== undefined)) {
        console.log(`💓 REAL Vitals update for ${pid}:`, {
          hr: data.heart_rate || data.rolling_hr,
          sbp: data.systolic_bp || data.rolling_sbp,
          spo2: data.spo2 || data.rolling_spo2
        });
        
        const newPoint: VitalsDataPoint = {
          timestamp: data.timestamp || data.prediction_time || new Date().toISOString(),
          heart_rate: data.heart_rate || data.rolling_hr || 0,
          systolic_bp: data.systolic_bp || data.rolling_sbp || 0,
          diastolic_bp: data.diastolic_bp || 80, // fallback
          spo2: data.spo2 || data.rolling_spo2 || 0,
          respiratory_rate: data.respiratory_rate || 16, // fallback
          temperature: data.temperature || 37.0, // fallback
        };

        setVitalsHistory(prev => {
          const patientHistory = prev[pid] || [];
          const updated = [...patientHistory, newPoint];
          
          // Keep only last MAX_VITALS_HISTORY_POINTS
          const trimmed = updated.slice(-MAX_VITALS_HISTORY_POINTS);
          
          return {
            ...prev,
            [pid]: trimmed,
          };
        });
      }
    }
  }, []);

  const { isConnected, subscribeToPatient, subscribeToFloor } = useWebSocket({
    onMessage: handleWebSocketMessage,
    onConnect: () => {
      console.log('Vitals history WebSocket connected');
      setIsLive(true);
      
      // Resubscribe to patient if specified
      if (patientId) {
        subscribeToPatient(patientId);
      }
    },
    onDisconnect: () => {
      setIsLive(false);
    },
  });

  // Subscribe to patient when patientId changes and fetch initial history
  useEffect(() => {
    if (isConnected && patientId) {
      subscribeToPatient(patientId);
      
      // Fetch initial vitals history from API
      patientsAPI.getVitalsHistory(patientId)
        .then((history) => {
          if (history && history.length > 0) {
            console.log(`📊 Fetched ${history.length} historical vitals points for ${patientId}`);
            console.log('Sample data point:', JSON.stringify(history[0]));
            
            const formattedHistory: VitalsDataPoint[] = history.map((point: any) => {
              // Backend returns vitals nested inside a 'vitals' object
              const vitals = point.vitals || point;
              return {
                timestamp: point.timestamp || new Date().toISOString(),
                heart_rate: vitals.heart_rate || vitals.rolling_hr || 0,
                systolic_bp: vitals.systolic_bp || vitals.rolling_sbp || 0,
                diastolic_bp: vitals.diastolic_bp || 80,
                spo2: vitals.spo2 || vitals.rolling_spo2 || 0,
                respiratory_rate: vitals.respiratory_rate || 16,
                temperature: vitals.temperature || 37.0,
              };
            });
            
            console.log(`✅ Formatted ${formattedHistory.length} vitals points for chart`);
            console.log('Sample formatted point:', JSON.stringify(formattedHistory[0]));
            
            setVitalsHistory(prev => ({
              ...prev,
              [patientId]: formattedHistory,
            }));
          } else {
            console.warn(`No vitals history returned for ${patientId}`);
          }
        })
        .catch((error) => {
          console.error(`Error fetching vitals history for ${patientId}:`, error);
        });
    }
  }, [isConnected, patientId, subscribeToPatient]);

  const getPatientVitalsHistory = useCallback((pid: string): VitalsDataPoint[] => {
    return vitalsHistory[pid] || [];
  }, [vitalsHistory]);

  const subscribeToFloorVitals = useCallback((floorId: string) => {
    if (isConnected) {
      subscribeToFloor(floorId);
    }
  }, [isConnected, subscribeToFloor]);

  return {
    vitalsHistory,
    getPatientVitalsHistory,
    isLive,
    isConnected,
    subscribeToPatient,
    subscribeToFloorVitals,
  };
}