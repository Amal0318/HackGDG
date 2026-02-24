// Hook for managing patient risk history with WebSocket updates
import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';

interface RiskDataPoint {
  timestamp: string;
  risk_score: number;
}

interface PatientRiskHistory {
  [patientId: string]: RiskDataPoint[];
}

const MAX_HISTORY_POINTS = 450; // Keep last 450 data points (15 minutes at 2-second intervals)

export function usePatientRiskHistory(patientId?: string) {
  const [riskHistory, setRiskHistory] = useState<PatientRiskHistory>({});
  const [isLive, setIsLive] = useState(false);

  const handleWebSocketMessage = useCallback((message: any) => {
    if (message.type === 'patient_update' || message.type === 'floor_update') {
      const pid = message.patient_id;
      const data = message.data;

      if (pid && data && (data.computed_risk !== undefined || data.risk_score !== undefined)) {
        const riskValue = data.computed_risk || data.risk_score || 0;
        console.log(`📊 Risk update for ${pid}: ${(riskValue * 100).toFixed(2)}%`);
        
        const newPoint: RiskDataPoint = {
          timestamp: data.last_updated || data.timestamp || data.prediction_time || new Date().toISOString(),
          risk_score: riskValue
        };

        setRiskHistory(prev => {
          const currentHistory = prev[pid] || [];
          const updatedHistory = [...currentHistory, newPoint].slice(-MAX_HISTORY_POINTS);
          
          return {
            ...prev,
            [pid]: updatedHistory
          };
        });
        
        setIsLive(true);
      }
    } else if (message.type === 'initial_data' && message.patients) {
      console.log('📊 Initial patient data received:', message.patients.length, 'patients');
      // Process initial data for all patients
      message.patients.forEach((patient: any) => {
        if (patient.computed_risk !== undefined || patient.risk_score !== undefined) {
          const riskValue = patient.computed_risk || patient.risk_score || 0;
          const newPoint: RiskDataPoint = {
            timestamp: patient.last_updated || patient.timestamp || patient.prediction_time || new Date().toISOString(),
            risk_score: riskValue
          };
          
          setRiskHistory(prev => ({
            ...prev,
            [patient.patient_id]: [newPoint]
          }));
        }
      });
      setIsLive(true);
    }
  }, []);

  const { isConnected, subscribeToPatient, subscribeToFloor } = useWebSocket({
    onMessage: handleWebSocketMessage,
    onConnect: () => {
      console.log('Risk history WebSocket connected');
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

  // Subscribe to patient when patientId changes
  useEffect(() => {
    if (isConnected && patientId) {
      subscribeToPatient(patientId);
    }
  }, [isConnected, patientId, subscribeToPatient]);

  const getPatientHistory = useCallback((pid: string): RiskDataPoint[] => {
    return riskHistory[pid] || [];
  }, [riskHistory]);

  const subscribeToFloorHistory = useCallback((floorId: string) => {
    if (isConnected) {
      subscribeToFloor(floorId);
    }
  }, [isConnected, subscribeToFloor]);

  return {
    riskHistory,
    getPatientHistory,
    isLive,
    isConnected,
    subscribeToPatient,
    subscribeToFloorHistory,
  };
}
