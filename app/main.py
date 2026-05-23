from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routes import ingestion, rag, analytics

app = FastAPI(
    title="Hospital Copilot AI - Backend Services",
    description="RAG-powered clinical decision support API utilizing SentenceTransformers, FAISS, and Groq LLM.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware for Frontend React App compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(ingestion.router)
app.include_router(rag.router)
app.include_router(analytics.router)

@app.get("/")
async def root():
    """Service status and system health details."""
    return {
        "status": "healthy",
        "service": "Hospital Copilot AI Backend",
        "version": "1.0.0",
        "configuration": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.GROQ_MODEL,
            "vector_store": "FAISS IndexFlatL2",
            "api_key_configured": bool(settings.GROQ_API_KEY)
        }
    }
