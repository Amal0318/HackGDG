// Hook for managing patient risk history with WebSocket updates
import { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import api from '../services/api';

interface RiskDataPoint {
  timestamp: string;
  risk_score: number;
}

interface PatientRiskHistory {
  [patientId: string]: RiskDataPoint[];
}

const MAX_HISTORY_POINTS = 30; // Keep last 30 data points (~1 minute at 2s intervals)
const UPDATE_THROTTLE_MS = 1000; // Update graph only once per second
const MAX_POINTS_PER_FLUSH = 3; // Max new points added per second per patient
const MAX_AGE_MS = 60000; // Remove points older than 60 seconds

export function usePatientRiskHistory(patientId?: string) {
  const [riskHistory, setRiskHistory] = useState<PatientRiskHistory>({});
  const [isLive, setIsLive] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Throttling: buffer to collect incoming data points
  const pendingUpdatesRef = useRef<{ [patientId: string]: RiskDataPoint[] }>({});
  const throttleTimerRef = useRef<number | null>(null);
  // Track last added timestamp per patient to prevent initial_data spam
  const lastAddedTimestampRef = useRef<{ [patientId: string]: number }>({});
  // Track if we've fetched initial history for each patient
  const hasInitialDataRef = useRef<{ [patientId: string]: boolean }>({});

  // Flush pending updates to state (called once per second) - mirrors vitals hook exactly
  const flushPendingUpdates = useCallback(() => {
    const updates = { ...pendingUpdatesRef.current };
    if (Object.keys(updates).length === 0) return;

    setRiskHistory(prev => {
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

        // Deduplicate: skip points within 1500ms of the latest existing point
        const latestExisting = recentHistory.length > 0
          ? new Date(recentHistory[recentHistory.length - 1].timestamp).getTime()
          : 0;

        const dedupedPoints = newPoints.filter(point =>
          new Date(point.timestamp).getTime() > latestExisting + 1500
        );

        const updated = [...recentHistory, ...dedupedPoints];
        next[pid] = updated.slice(-MAX_HISTORY_POINTS);
      });

      return next;
    });

    // Clear the buffer
    pendingUpdatesRef.current = {};
  }, []);

  // Start throttle timer on mount - same pattern as vitals hook
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

      if (pid && data && (data.computed_risk !== undefined || data.risk_score !== undefined)) {
        const riskValue = data.computed_risk || data.risk_score || 0;
        const timestamp = data.last_updated || data.timestamp || data.prediction_time || new Date().toISOString();
        const tsMs = new Date(timestamp).getTime();

        // Only buffer if this timestamp is newer than last added for this patient
        const lastTs = lastAddedTimestampRef.current[pid] || 0;
        if (tsMs > lastTs + 1500) {
          const newPoint: RiskDataPoint = { timestamp, risk_score: riskValue };
          if (!pendingUpdatesRef.current[pid]) {
            pendingUpdatesRef.current[pid] = [];
          }
          // Replace buffer for this patient (only keep latest in buffer)
          pendingUpdatesRef.current[pid] = [newPoint];
          lastAddedTimestampRef.current[pid] = tsMs;
        }

        setIsLive(true);
      }
    } else if (message.type === 'initial_data' && message.patients) {
      // Only seed patients that have NO history yet - prevents accumulation on each poll
      message.patients.forEach((patient: any) => {
        if (patient.computed_risk !== undefined || patient.risk_score !== undefined) {
          const pid = patient.patient_id;
          const riskValue = patient.computed_risk || patient.risk_score || 0;
          const timestamp = patient.last_updated || patient.timestamp || new Date().toISOString();
          const tsMs = new Date(timestamp).getTime();

          // Only add if we haven't added a recent point for this patient
          const lastTs = lastAddedTimestampRef.current[pid] || 0;
          if (tsMs > lastTs + 1500) {
            setRiskHistory(prev => {
              // Skip if patient already has history
              if (prev[pid] && prev[pid].length > 0) return prev;
              return { ...prev, [pid]: [{ timestamp, risk_score: riskValue }] };
            });
            lastAddedTimestampRef.current[pid] = tsMs;
          }
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

  // Fetch initial history from API when patientId changes
  useEffect(() => {
    if (!patientId || hasInitialDataRef.current[patientId]) return;

    const fetchHistory = async () => {
      setIsLoading(true);
      try {
        const history = await api.patients.getRiskHistory(patientId, 1);
        if (history && Array.isArray(history) && history.length > 0) {
          // Transform history into RiskDataPoint format
          const riskPoints: RiskDataPoint[] = history
            .map((item: any) => {
              const riskValue = item.computed_risk !== undefined
                ? item.computed_risk
                : item.risk_score !== undefined
                ? item.risk_score
                : item.prediction !== undefined
                ? item.prediction
                : undefined;

              if (riskValue === undefined) return null;

              return {
                timestamp: item.timestamp || item.prediction_time || item.last_updated || new Date().toISOString(),
                risk_score: riskValue,
              };
            })
            .filter((p): p is RiskDataPoint => p !== null)
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

          // Only seed if we got valid data
          if (riskPoints.length > 0) {
            setRiskHistory(prev => ({
              ...prev,
              [patientId]: riskPoints.slice(-MAX_HISTORY_POINTS),
            }));
            hasInitialDataRef.current[patientId] = true;
            // Update last timestamp to prevent WebSocket duplicates
            const lastPoint = riskPoints[riskPoints.length - 1];
            lastAddedTimestampRef.current[patientId] = new Date(lastPoint.timestamp).getTime();
          }
        }
      } catch (error) {
        console.error(`Failed to fetch risk history for ${patientId}:`, error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, [patientId]);

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
    isLoading,
    isConnected,
    subscribeToPatient,
    subscribeToFloorHistory,
  };
}
