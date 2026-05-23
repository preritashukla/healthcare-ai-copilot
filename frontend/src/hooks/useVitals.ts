/**
 * useVitals — React hook for fetching patient vitals telemetry.
 *
 * Usage:
 *   const { vitals, isLoading, error, isMock } = useVitals('Marcus Vance');
 */

import { useState, useEffect } from 'react';
import { fetchVitals } from '../api/analytics';
import type { VitalsResponse } from '../types/api';

interface UseVitalsResult {
  vitals: VitalsResponse | null;
  isLoading: boolean;
  error: string | null;
  isMock: boolean;
}

export function useVitals(patientId = 'Marcus Vance'): UseVitalsResult {
  const [vitals, setVitals] = useState<VitalsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isMock, setIsMock] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchVitals(patientId);
        if (!cancelled) {
          setVitals(data);
          setIsMock(data.status === 'mock');
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load vitals');
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    load();
    return () => { cancelled = true; };
  }, [patientId]);

  return { vitals, isLoading, error, isMock };
}
