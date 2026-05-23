/**
 * Central Axios HTTP client.
 * Base URL is read from VITE_API_BASE_URL environment variable.
 * Falls back to http://127.0.0.1:8000 for local development.
 */

const BASE_URL = (import.meta as { env: Record<string, string> }).env?.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

interface RequestOptions {
  headers?: Record<string, string>;
  signal?: AbortSignal;
}

async function request<T>(
  method: 'GET' | 'POST',
  path: string,
  body?: unknown,
  options?: RequestOptions
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      ...(options?.headers ?? {}),
    },
    body: body ? JSON.stringify(body) : undefined,
    signal: options?.signal,
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`API ${method} ${path} → ${res.status}: ${detail}`);
  }

  return res.json() as Promise<T>;
}

export const apiClient = {
  get: <T>(path: string, options?: RequestOptions) =>
    request<T>('GET', path, undefined, options),
  post: <T>(path: string, body: unknown, options?: RequestOptions) =>
    request<T>('POST', path, body, options),
};
