# Healthcare AI Copilot

AI-powered clinical decision support system using Retrieval-Augmented Generation.

## Problem
Clinical handover notes and patient histories are often unstructured and difficult to analyze quickly.

## Solution
This project builds an LLM-powered assistant that:

- retrieves relevant medical knowledge
- analyzes patient notes
- generates safety-focused suggestions

## Architecture

PDF → Embedding → Vector Database → LLM → Response

## Tech Stack

Python  
LangChain / RAG  
LLM APIs  
FastAPI
