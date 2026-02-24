// HTTPS polling hook that emulates the previous WebSocket API
import { useEffect, useRef, useState, useCallback } from 'react';
import { floorsAPI } from '../services/api';

interface PatientUpdate {
  type: 'patient_update' | 'floor_update' | 'subscribed' | 'initial_data';
  patient_id?: string;
  floor_id?: string;
  data?: any;
  target?: string;
  patients?: any[];
}

interface UseWebSocketOptions {
  onMessage?: (data: PatientUpdate) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  autoReconnect?: boolean; // When true keep polling on an interval
  reconnectInterval?: number; // Reused as HTTP polling interval (ms)
}

function normalizeFloorId(value?: string | number): string {
  if (value === undefined || value === null) {
    return 'F1';
  }
  if (typeof value === 'number') {
    return `F${value}`;
  }
  const trimmed = String(value).trim().toUpperCase();
  if (/^\d+F$/.test(trimmed)) {
    const [digits, suffix] = [trimmed.slice(0, -1), trimmed.slice(-1)];
    return `${suffix}${digits}`;
  }
  if (/^F\d+$/.test(trimmed)) {
    return trimmed;
  }
  return trimmed;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    onMessage,
    onConnect,
    onDisconnect,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<PatientUpdate | null>(null);

  const pollTimerRef = useRef<ReturnType<typeof setInterval>>();
  const isFetchingRef = useRef(false);
  const hasSentInitialRef = useRef(false);
  const patientSubscriptionsRef = useRef<Set<string>>(new Set());
  const floorSubscriptionsRef = useRef<Set<string>>(new Set());
  const isConnectedRef = useRef(false);

  const updateConnectionState = useCallback((next: boolean) => {
    if (isConnectedRef.current === next) {
      return;
    }
    isConnectedRef.current = next;
    setIsConnected(next);
    if (next) {
      onConnect?.();
    } else {
      onDisconnect?.();
    }
  }, [onConnect, onDisconnect]);

  const emitMessage = useCallback((message: PatientUpdate) => {
    setLastMessage(message);
    onMessage?.(message);
  }, [onMessage]);

  const emitPatientPayload = useCallback((patient: any, floorId: string, viaFloor: boolean) => {
    emitMessage({
      type: viaFloor ? 'floor_update' : 'patient_update',
      patient_id: patient.patient_id,
      floor_id: floorId,
      target: viaFloor ? floorId : patient.patient_id,
      data: patient,
    });
  }, [emitMessage]);

  const emitInitialPayload = useCallback((patients: any[]) => {
    emitMessage({
      type: 'initial_data',
      patients,
    });
  }, [emitMessage]);

  const poll = useCallback(async () => {
    if (isFetchingRef.current) {
      console.log('[useWebSocket] Skipping poll - already fetching');
      return;
    }
    isFetchingRef.current = true;
    console.log('[useWebSocket] Starting poll...');

    try {
      const floors = await floorsAPI.getAll();
      console.log('[useWebSocket] Fetched floors:', floors.length);
      const floorBatches = await Promise.all(
        floors.map(async (floor) => {
          try {
            const patients = await floorsAPI.getPatients(floor.id);
            console.log(`[useWebSocket] Fetched ${patients.length} patients from floor ${floor.id}`);
            return patients.map((patient) => ({
              ...patient,
              floor_id: patient.floor_id || floor.id,
            }));
          } catch (error) {
            console.error(`Failed to fetch patients for floor ${floor.id}`, error);
            return [];
          }
        })
      );

      const allPatients = floorBatches.flat();
      console.log('[useWebSocket] Total patients fetched:', allPatients.length);

      if (!hasSentInitialRef.current) {
        console.log('[useWebSocket] Emitting initial payload');
        emitInitialPayload(allPatients);
        hasSentInitialRef.current = true;
      }

      const subscribedPatients = patientSubscriptionsRef.current;
      const subscribedFloors = floorSubscriptionsRef.current;
      const hasSubscriptions = subscribedPatients.size > 0 || subscribedFloors.size > 0;

      console.log('[useWebSocket] Subscriptions:', {
        patients: Array.from(subscribedPatients),
        floors: Array.from(subscribedFloors),
      });

      allPatients.forEach((patient) => {
        const fallbackFloor = (patient as any).floor ?? (patient as any).floorId;
        const normalizedFloor = normalizeFloorId(patient.floor_id ?? fallbackFloor);
        const matchesPatient = subscribedPatients.has(patient.patient_id);
        const matchesFloor = subscribedFloors.has(normalizedFloor);

        if (!hasSubscriptions || matchesPatient || matchesFloor) {
          emitPatientPayload(patient, normalizedFloor, matchesFloor && !matchesPatient);
        }
      });

      updateConnectionState(true);
      console.log('[useWebSocket] Poll complete - emitted updates for', allPatients.length, 'patients');
    } catch (error) {
      console.error('HTTPS polling failed:', error);
      updateConnectionState(false);
    } finally {
      isFetchingRef.current = false;
    }
  }, [emitInitialPayload, emitPatientPayload, updateConnectionState]);

  const subscribeToPatient = useCallback((patientId: string) => {
    if (!patientId) {
      return;
    }
    patientSubscriptionsRef.current.add(patientId);
    poll();
  }, [poll]);

  const subscribeToFloor = useCallback((floorId: string) => {
    if (!floorId) {
      return;
    }
    floorSubscriptionsRef.current.add(normalizeFloorId(floorId));
    poll();
  }, [poll]);

  const connect = useCallback(() => {
    if (pollTimerRef.current) {
      return;
    }
    hasSentInitialRef.current = false;
    console.log('[useWebSocket] Starting polling with interval:', reconnectInterval);
    poll();

    if (autoReconnect) {
      pollTimerRef.current = setInterval(() => {
        console.log('[useWebSocket] Polling tick');
        poll();
      }, reconnectInterval);
    }
  }, [autoReconnect, poll, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = undefined;
    }
    patientSubscriptionsRef.current.clear();
    floorSubscriptionsRef.current.clear();
    hasSentInitialRef.current = false;
    updateConnectionState(false);
  }, [updateConnectionState]);

  const send = useCallback((data: any) => {
    console.warn('send() is not supported with HTTPS polling transport:', data);
  }, []);

  useEffect(() => {
    connect();
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    send,
    subscribeToPatient,
    subscribeToFloor,
    connect,
    disconnect,
  };
}
