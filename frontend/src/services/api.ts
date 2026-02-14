import axios from 'axios';
import { HealthStatus, PatientsResponse } from '../types';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 5000,
});

export const getHealth = async (): Promise<HealthStatus> => {
  const response = await api.get<HealthStatus>('/health');
  return response.data;
};

export const getPatients = async (): Promise<PatientsResponse> => {
  const response = await api.get<PatientsResponse>('/patients');
  return response.data;
};
