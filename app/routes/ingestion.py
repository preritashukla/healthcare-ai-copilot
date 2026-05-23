import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import settings
from app.services.pdf_service import pdf_service
from app.services.rag_service import rag_service

router = APIRouter(prefix="/api/ingestion", tags=["Ingestion"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF document, extracts text, and indexes it in the FAISS vector database."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
        
    try:
        content = await file.read()
        
        # Save file to upload directory
        if not os.path.exists(settings.UPLOAD_DIR):
            os.makedirs(settings.UPLOAD_DIR)
        path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            f.write(content)
            
        # Parse PDF text
        text = pdf_service.extract_text_from_bytes(content)
        if not text.strip():
            raise HTTPException(status_code=400, detail="The uploaded PDF contains no extractable text.")
            
        # Add to vector database
        words_count = rag_service.add_document(file.filename, text)
        
        return {
            "status": "success",
            "message": f"Successfully parsed and indexed document: {file.filename}",
            "data": {
                "filename": file.filename,
                "word_count": words_count,
                "vectors_total": rag_service.index.ntotal if rag_service.index else 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")

@router.get("/documents")
async def get_indexed_documents():
    """Retrieves a list of all currently indexed patient documents."""
    try:
        docs = rag_service.list_documents()
        return {
            "status": "success",
            "data": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
