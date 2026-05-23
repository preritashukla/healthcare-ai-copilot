/**
 * useRagQuery — React hook for submitting clinical queries to the RAG pipeline.
 *
 * Usage:
 *   const { query, setQuery, submit, response, isLoading, error, isMock } = useRagQuery();
 */

import { useState, useCallback } from 'react';
import { queryRag } from '../api/rag';
import type { QueryResponse } from '../types/api';

interface UseRagQueryResult {
  query: string;
  setQuery: (q: string) => void;
  submit: () => Promise<void>;
  response: QueryResponse | null;
  isLoading: boolean;
  error: string | null;
  isMock: boolean;
}

export function useRagQuery(): UseRagQueryResult {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isMock, setIsMock] = useState(false);

  const submit = useCallback(async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await queryRag(query);
      setResponse(result);
      setIsMock(result.status === 'mock');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unexpected query error');
    } finally {
      setIsLoading(false);
    }
  }, [query]);

  return { query, setQuery, submit, response, isLoading, error, isMock };
}
