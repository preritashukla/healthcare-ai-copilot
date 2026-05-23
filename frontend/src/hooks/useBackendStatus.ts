/**
 * useBackendStatus — Checks backend health on mount.
 * Returns isOnline: true if /  endpoint responds with status: "healthy".
 */

import { useState, useEffect } from 'react';
import { checkHealth } from '../api/analytics';

interface UseBackendStatusResult {
  isOnline: boolean | null; // null = still checking
}

export function useBackendStatus(): UseBackendStatusResult {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth().then((health) => {
      setIsOnline(health?.status === 'healthy');
    });
  }, []);

  return { isOnline };
}
