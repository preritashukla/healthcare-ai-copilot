from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service

router = APIRouter(prefix="/api/rag", tags=["RAG Query"])

class QueryRequest(BaseModel):
    query: str = Field(..., example="Does the patient have a penicillin allergy or medication issues?")
    k: int = Field(3, ge=1, le=10, description="Number of source chunks to retrieve from FAISS")

class SourceResponse(BaseModel):
    source: str
    score: float
    text: str

class QueryResponse(BaseModel):
    status: str
    query: str
    response: str
    sources: list[SourceResponse]

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Retrieves relevant contexts from FAISS database and utilizes Groq LLM to answer clinical inquiries."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    try:
        # Retrieve context from FAISS
        sources = rag_service.retrieve_context(request.query, k=request.k)
        
        # Format retrieval context for the LLM
        combined_context = "\n\n".join([
            f"Source: {src['source']} (Distance: {src['score']:.4f})\n{src['text']}"
            for src in sources
        ])
        
        if not combined_context:
            combined_context = "No relevant context found in database. Rely on general medical knowledge warnings."
            
        # Get LLM safety insights
        insights = llm_service.generate_insights(request.query, combined_context)
        
        # Build response sources
        sources_response = [
            SourceResponse(source=src["source"], score=src["score"], text=src["text"])
            for src in sources
        ]
        
        return QueryResponse(
            status="success",
            query=request.query,
            response=insights,
            sources=sources_response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query pipeline error: {str(e)}")
