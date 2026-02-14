import { useEffect } from 'react';
import { getHealth, getPatients } from '../services/api';
import { useStore } from '../store';

export const useHealthStatus = () => {
  const setHealthStatus = useStore(state => state.setHealthStatus);
  const setPatients = useStore(state => state.setPatients);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const health = await getHealth();
        setHealthStatus(health);
      } catch (error) {
        console.error('Failed to fetch health status:', error);
      }
    };

    const fetchPatients = async () => {
      try {
        const response = await getPatients();
        setPatients(response.patients);
      } catch (error) {
        console.error('Failed to fetch patients:', error);
      }
    };

    // Initial fetch
    fetchHealth();
    fetchPatients();

    // Poll health status every 10 seconds
    const healthInterval = setInterval(fetchHealth, 10000);

    // Poll patients every 30 seconds
    const patientsInterval = setInterval(fetchPatients, 30000);

    return () => {
      clearInterval(healthInterval);
      clearInterval(patientsInterval);
    };
  }, [setHealthStatus, setPatients]);
};
