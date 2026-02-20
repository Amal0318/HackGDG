// React hooks for fetching patient data from backend
import { useState, useEffect, useCallback } from 'react';
import { floorsAPI, patientsAPI, statsAPI, transformPatientData } from '../services/api';
import { useWebSocket } from './useWebSocket';

export interface Patient {
  patient_id: string;
  name: string;
  bed_number: string;
  floor: number;
  latest_risk_score: number;
  risk_history: Array<{ timestamp: string; risk_score: number }>;
  abnormal_vitals: Array<{ vital: string; value: number; unit: string }>;
  vitals: {
    heart_rate: number;
    systolic_bp: number;
    diastolic_bp: number;
    spo2: number;
    respiratory_rate: number;
    temperature: number;
  };
  alert_acknowledged?: boolean;
  acknowledged_by?: string;
  acknowledged_at?: string;
  // Keep other potential fields
  [key: string]: any;
}

interface UsePatientOptions {
  floorId?: string;
  refreshInterval?: number; // milliseconds
}

export function usePatients(options: UsePatientOptions = {}) {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [availableFloors, setAvailableFloors] = useState<string[]>([]);

  const upsertPatient = useCallback((rawPatient: any) => {
    if (!rawPatient) {
      return;
    }
    const transformed = transformPatientData(rawPatient);
    setPatients((prev) => {
      const existingIndex = prev.findIndex((p) => p.patient_id === transformed.patient_id);
      let next: Patient[];
      if (existingIndex === -1) {
        next = [...prev, transformed];
      } else {
        next = [...prev];
        next[existingIndex] = {
          ...next[existingIndex],
          ...transformed,
          vitals: transformed.vitals,
          latest_risk_score: transformed.latest_risk_score,
          risk_history: transformed.risk_history?.length ? transformed.risk_history : next[existingIndex].risk_history,
        };
      }
      return next;
    });
  }, []);

  const handleRealtimeMessage = useCallback((message: any) => {
    if (!message) {
      return;
    }

    if (message.type === 'initial_data' && Array.isArray(message.patients)) {
      message.patients.forEach((patient: any) => {
        if (!options.floorId || patient.floor_id === options.floorId || patient.floor === options.floorId) {
          upsertPatient(patient);
        }
      });
      setLoading(false);
      return;
    }

    if ((message.type === 'patient_update' || message.type === 'floor_update') && message.data) {
      if (!options.floorId || message.floor_id === options.floorId || message.data.floor_id === options.floorId) {
        upsertPatient(message.data);
      }
    }
  }, [options.floorId, upsertPatient]);

  const { isConnected, subscribeToFloor } = useWebSocket({
    onMessage: handleRealtimeMessage,
  });

  const { floorId, refreshInterval = 5000 } = options;

  const refetchPatients = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      let allPatients: Patient[] = [];

      if (floorId) {
        const backendData = await floorsAPI.getPatients(floorId);
        allPatients = backendData.map(transformPatientData);
        setAvailableFloors([floorId]);
      } else {
        const floors = await floorsAPI.getAll();
        const floorIds = floors.map((floor) => floor.id);
        setAvailableFloors(floorIds);
        const patientsPromises = floors.map(floor => 
          floorsAPI.getPatients(floor.id)
        );
        const allFloorsData = await Promise.all(patientsPromises);
        allPatients = allFloorsData.flat().map(transformPatientData);
      }

      setPatients(allPatients);
      setLoading(false);
    } catch (err) {
      console.error('Failed to fetch patients:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch patients');
      setLoading(false);
    }
  }, [floorId]);

  useEffect(() => {
    let mounted = true;
    let intervalId: ReturnType<typeof setInterval> | undefined;

    async function fetchPatients() {
      try {
        setLoading(true);
        setError(null);

        let allPatients: Patient[] = [];

        if (floorId) {
          // Fetch specific floor
          const backendData = await floorsAPI.getPatients(floorId);
          allPatients = backendData.map(transformPatientData);
          setAvailableFloors([floorId]);
        } else {
          // Fetch all floors
          const floors = await floorsAPI.getAll();
          const floorIds = floors.map((floor) => floor.id);
          setAvailableFloors(floorIds);
          const patientsPromises = floors.map(floor => 
            floorsAPI.getPatients(floor.id)
          );
          const allFloorsData = await Promise.all(patientsPromises);
          allPatients = allFloorsData.flat().map(transformPatientData);
        }

        if (mounted) {
          setPatients(allPatients);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to fetch patients:', err);
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to fetch patients');
          setLoading(false);
        }
      }
    }

    fetchPatients();

    // Set up auto-refresh
    if (refreshInterval > 0) {
      intervalId = setInterval(fetchPatients, refreshInterval);
    }

    return () => {
      mounted = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [floorId, refreshInterval]);

  useEffect(() => {
    if (!isConnected) {
      return;
    }

    const floorsToSubscribe = floorId ? [floorId] : availableFloors;
    floorsToSubscribe
      .filter(Boolean)
      .forEach((floor) => {
        subscribeToFloor(floor);
      });
  }, [isConnected, floorId, availableFloors, subscribeToFloor]);

  return { patients, loading, error, refetchPatients };
}

export function usePatient(patientId: string, refreshInterval = 5000) {
  const [patient, setPatient] = useState<Patient | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    let intervalId: ReturnType<typeof setInterval> | undefined;

    async function fetchPatient() {
      try {
        setLoading(true);
        setError(null);

        const backendData = await patientsAPI.getById(patientId);
        const transformed = transformPatientData(backendData);

        if (mounted) {
          setPatient(transformed);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to fetch patient:', err);
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to fetch patient');
          setLoading(false);
        }
      }
    }

    if (patientId) {
      fetchPatient();

      if (refreshInterval > 0) {
        intervalId = setInterval(fetchPatient, refreshInterval);
      }
    }

    return () => {
      mounted = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [patientId, refreshInterval]);

  return { patient, loading, error };
}

export function useFloors() {
  const [floors, setFloors] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchFloors() {
      try {
        setLoading(true);
        setError(null);
        const data = await floorsAPI.getAll();
        setFloors(data);
        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch floors:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch floors');
        setLoading(false);
      }
    }

    fetchFloors();
  }, []);

  return { floors, loading, error };
}

export function useStats(refreshInterval = 10000) {
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    let intervalId: ReturnType<typeof setInterval> | undefined;

    async function fetchStats() {
      try {
        setLoading(true);
        setError(null);
        const data = await statsAPI.getOverview();
        if (mounted) {
          setStats(data);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to fetch stats:', err);
        if (mounted) {
          setError(err instanceof Error ? err.message : 'Failed to fetch stats');
          setLoading(false);
        }
      }
    }

    fetchStats();

    if (refreshInterval > 0) {
      intervalId = setInterval(fetchStats, refreshInterval);
    }

    return () => {
      mounted = false;
      if (intervalId) clearInterval(intervalId);
    };
  }, [refreshInterval]);

  return { stats, loading, error };
}
