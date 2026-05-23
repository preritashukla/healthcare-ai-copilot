import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.services.pdf_service import pdf_service

class RAGService:
    def __init__(self):
        print(f"Initializing SentenceTransformer model: {settings.EMBEDDING_MODEL}...")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index = None
        self.texts = []
        self.sources = []
        
        # Build initial vector store if there are PDFs in the upload folder
        self.initialize_index()

    def initialize_index(self):
        """Scans the upload directory and indexes existing PDFs."""
        if not os.path.exists(settings.UPLOAD_DIR):
            os.makedirs(settings.UPLOAD_DIR)
            
        pdf_files = [f for f in os.listdir(settings.UPLOAD_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            print("No existing PDFs found in upload directory. Vector store is currently empty.")
            return

        documents = {}
        for fname in pdf_files:
            path = os.path.join(settings.UPLOAD_DIR, fname)
            try:
                text = pdf_service.extract_text_from_path(path)
                if text.strip():
                    documents[fname] = text
            except Exception as e:
                print(f"Error reading {fname}: {e}")

        if documents:
            texts_list = list(documents.values())
            files_list = list(documents.keys())
            
            print(f"Generating embeddings for {len(texts_list)} documents...")
            embeddings = self.embedding_model.encode(texts_list)
            embeddings_np = np.array(embeddings)
            
            dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_np)
            
            self.texts = texts_list
            self.sources = files_list
            print(f"FAISS index built successfully with {self.index.ntotal} vectors.")

    def add_document(self, filename: str, content: str) -> int:
        """Adds a new document text dynamically to the FAISS index."""
        if not content.strip():
            return 0
            
        # Generate embedding
        embedding = self.embedding_model.encode([content])
        embedding_np = np.array(embedding)
        
        # Initialize index if it doesn't exist
        if self.index is None:
            dimension = embedding_np.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            
        self.index.add(embedding_np)
        self.texts.append(content)
        self.sources.append(filename)
        
        print(f"Dynamically added '{filename}' to FAISS index. Total vectors: {self.index.ntotal}")
        return len(content.split())

    def retrieve_context(self, query: str, k: int = 3) -> list[dict]:
        """Queries FAISS index and returns matching texts with source metadata and scores."""
        if self.index is None or not self.texts:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        query_np = np.array(query_embedding)
        
        # Ensure k is not larger than indexed documents
        actual_k = min(k, self.index.ntotal)
        if actual_k <= 0:
            return []
            
        distances, indices = self.index.search(query_np, actual_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "source": self.sources[idx],
                "score": float(dist),
                "text": self.texts[idx]
            })
            
        return results

    def list_documents(self) -> list[dict]:
        """Lists metadata of all currently indexed documents."""
        docs = []
        for i, src in enumerate(self.sources):
            docs.append({
                "name": src,
                "word_count": len(self.texts[i].split()),
                "index_position": i
            })
        return docs

# Instantiate as a singleton to share state across routes
rag_service = RAGService()
