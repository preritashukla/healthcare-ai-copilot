/**
 * Shared API type definitions mirroring the FastAPI Pydantic backend models.
 * Do not modify without also updating app/routes/*.py
 */

// ─── RAG Query ────────────────────────────────────────────────────────────────

export interface QueryRequest {
  query: string;
  k?: number;
}

export interface SourceResponse {
  source: string;
  score: number;
  text: string;
}

export interface QueryResponse {
  status: string;
  query: string;
  response: string;
  sources: SourceResponse[];
}

// ─── Analytics / Vitals ───────────────────────────────────────────────────────

export interface VitalDataPoint {
  day: number;
  heartRate: number;
  temperature: number;
}

export interface VitalsResponse {
  status: string;
  data: {
    patient_id: string;
    heart_rate: number[];
    temperature: number[];
    days: number[];
    avg_heart_rate: number;
    avg_temperature: number;
    max_heart_rate: number;
    min_heart_rate: number;
  };
}

// ─── Health Check ─────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  configuration: {
    embedding_model: string;
    llm_model: string;
    vector_store: string;
    api_key_configured: boolean;
  };
}
