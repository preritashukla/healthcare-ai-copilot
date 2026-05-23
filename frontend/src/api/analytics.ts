/**
 * Analytics API service.
 * Wraps GET /api/analytics/vitals and GET / (health check).
 */

import { apiClient } from './client';
import type { VitalsResponse, HealthResponse } from '../types/api';

/** Mock vitals used when backend is unreachable. */
const MOCK_VITALS: VitalsResponse = {
  status: 'mock',
  data: {
    patient_id: 'Dr. Smith — Demo Patient',
    heart_rate: [85, 90, 95, 100, 92, 88, 84],
    temperature: [98.4, 98.7, 99.1, 99.5, 98.9, 98.6, 98.3],
    days: [1, 2, 3, 4, 5, 6, 7],
    avg_heart_rate: 90.6,
    avg_temperature: 98.8,
    max_heart_rate: 100,
    min_heart_rate: 84,
  },
};

/**
 * Fetch patient vitals telemetry.
 * Returns mock data if backend is unreachable.
 */
export async function fetchVitals(patientId = 'Marcus Vance'): Promise<VitalsResponse> {
  try {
    return await apiClient.get<VitalsResponse>(
      `/api/analytics/vitals?patient_id=${encodeURIComponent(patientId)}`
    );
  } catch (err) {
    console.warn('[Analytics] Backend unreachable, using mock vitals.', err);
    return MOCK_VITALS;
  }
}

/**
 * Ping backend health endpoint.
 * Returns null if unreachable.
 */
export async function checkHealth(): Promise<HealthResponse | null> {
  try {
    return await apiClient.get<HealthResponse>('/');
  } catch {
    return null;
  }
}
