/**
 * RAG Query API service.
 * Wraps POST /api/rag/query with typed request/response.
 */

import { apiClient } from './client';
import type { QueryRequest, QueryResponse } from '../types/api';

/** Mock fallback response used when backend is unreachable. */
const MOCK_RAG_RESPONSE: QueryResponse = {
  status: 'mock',
  query: '',
  response:
    'Clinical decision support is currently operating in offline mode. Backend connectivity could not be established. Please verify the backend service is running.',
  sources: [],
};

/**
 * Submit a clinical query to the RAG pipeline.
 * Returns mock data if the backend is unreachable.
 */
export async function queryRag(query: string, k = 3): Promise<QueryResponse> {
  const request: QueryRequest = { query, k };
  try {
    return await apiClient.post<QueryResponse>('/api/rag/query', request);
  } catch (err) {
    console.warn('[RAG] Backend unreachable, using mock fallback.', err);
    return { ...MOCK_RAG_RESPONSE, query };
  }
}
