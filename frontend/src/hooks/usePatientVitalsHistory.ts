// Hook for managing patient vitals history with WebSocket updates
import { useState, useEffect, useCallback, useRef } from 'react';
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

const MAX_VITALS_HISTORY_POINTS = 30; // Keep last 30 data points (enough for ~1 minute)
const UPDATE_THROTTLE_MS = 1000; // Update graph only once per second
const MAX_POINTS_PER_FLUSH = 5; // Limit points added per flush
const MAX_AGE_MS = 120000; // Remove points older than 2 minutes

export function usePatientVitalsHistory(patientId?: string) {
  const [vitalsHistory, setVitalsHistory] = useState<PatientVitalsHistory>({});
  const [isLive, setIsLive] = useState(false);
  
  // Throttling: buffer to collect incoming data points (array to accumulate all points)
  const pendingUpdatesRef = useRef<{ [patientId: string]: VitalsDataPoint[] }>({});
  const throttleTimerRef = useRef<number | null>(null);

  // Flush pending updates to state (called once per second)
  const flushPendingUpdates = useCallback(() => {
    const updates = { ...pendingUpdatesRef.current };
    if (Object.keys(updates).length === 0) return;

    setVitalsHistory(prev => {
      const next = { ...prev };
      const now = Date.now();
      
      Object.entries(updates).forEach(([pid, pointsBuffer]) => {
        let newPoints = pointsBuffer;
        
        // Limit flush to prevent overwhelming the graph
        if (newPoints.length > MAX_POINTS_PER_FLUSH) {
          newPoints = newPoints.slice(-MAX_POINTS_PER_FLUSH);
        }
        
        const currentHistory = next[pid] || [];
        
        // Remove old points (older than MAX_AGE_MS)
        const recentHistory = currentHistory.filter(p => 
          (now - new Date(p.timestamp).getTime()) < MAX_AGE_MS
        );
        
        // Deduplicate: remove points with timestamps within 1000ms of each other
        const dedupedPoints: VitalsDataPoint[] = [];
        newPoints.forEach(point => {
          const isDuplicate = recentHistory.some(existing => 
            Math.abs(new Date(existing.timestamp).getTime() - new Date(point.timestamp).getTime()) < 1000
          );
          if (!isDuplicate) {
            dedupedPoints.push(point);
          }
        });
        
        const updated = [...recentHistory, ...dedupedPoints];
        next[pid] = updated.slice(-MAX_VITALS_HISTORY_POINTS);
      });
      
      return next;
    });

    // Clear the buffer
    pendingUpdatesRef.current = {};
  }, []);

  // Start throttle timer on mount
  useEffect(() => {
    throttleTimerRef.current = setInterval(flushPendingUpdates, UPDATE_THROTTLE_MS);
    
    return () => {
      if (throttleTimerRef.current) {
        clearInterval(throttleTimerRef.current);
      }
    };
  }, [flushPendingUpdates]);

  const handleWebSocketMessage = useCallback((message: any) => {
    if (message.type === 'patient_update' || message.type === 'floor_update') {
      const pid = message.patient_id;
      const data = message.data;

      if (pid && data) {
        // Extract vitals from nested structure OR flat structure
        const vitals = data.vitals || data;
        
        // Check if we have any vital signs data
        if (vitals.heart_rate !== undefined || vitals.rolling_hr !== undefined) {
          const newPoint: VitalsDataPoint = {
            timestamp: data.last_updated || data.timestamp || data.prediction_time || new Date().toISOString(),
            heart_rate: vitals.heart_rate || vitals.rolling_hr || 0,
            systolic_bp: vitals.systolic_bp || vitals.rolling_sbp || 0,
            diastolic_bp: vitals.diastolic_bp || 80, // fallback
            spo2: vitals.spo2 || vitals.rolling_spo2 || 0,
            respiratory_rate: vitals.respiratory_rate || 16, // fallback
            temperature: vitals.temperature || 37.0, // fallback
          };

          // Add to buffer array instead of directly updating state (throttled to 1 update/second)
          if (!pendingUpdatesRef.current[pid]) {
            pendingUpdatesRef.current[pid] = [];
          }
          pendingUpdatesRef.current[pid].push(newPoint);
        }
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