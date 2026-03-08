# Healthcare AI Copilot

AI-powered clinical decision support prototype using Retrieval Augmented Generation (RAG).

## Overview

This project builds an AI assistant that analyzes patient records and clinical handover notes to extract safety insights and assist doctors in decision making.

The system:

- Ingests patient PDFs containing clinical history
- Extracts safety signals from medical text
- Uses Retrieval-Augmented Generation to reference historical cases
- Visualizes patient stability trends
- Supports multiple clinical roles

This system is **decision support only** and does not provide diagnoses.

---

## Architecture

Patient PDFs  
↓  
Text Extraction  
↓  
Embedding Model (SentenceTransformers)  
↓  
Vector Database (FAISS)  
↓  
LLM Reasoning (Groq API)  
↓  
Clinical Insights + Visualization

---

## Tech Stack

Python  
SentenceTransformers  
FAISS Vector Search  
Groq LLM API  
Matplotlib / Seaborn

---

## Features

• Clinical history analysis  
• Medication–allergy conflict detection  
• Patient stability trend visualization  
• RAG-based reasoning on past cases

---

## Future Possible Improvements

- Small Language Model fine-tuning
- Clinical dataset expansion
- Interactive dashboard
