"""
Healthcare AI Copilot
---------------------

Prototype clinical decision support system using
Retrieval-Augmented Generation (RAG).

This script:
1. Loads patient PDF documents
2. Extracts medical text
3. Creates embeddings using SentenceTransformers
4. Stores embeddings in FAISS vector database
5. Retrieves relevant context
6. Sends context to LLM for insights

NOTE: This system provides decision support only.
It does NOT provide medical diagnosis.
"""

import os
import PyPDF2
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Configuration
# -----------------------------

PDF_FOLDER = "sample_data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GROQ_MODEL = "llama3-8b-8192"

API_KEY = os.getenv("GROQ_API_KEY")


# -----------------------------
# Initialize models
# -----------------------------

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

groq_client = Groq(api_key=API_KEY)


# -----------------------------
# PDF Text Extraction
# -----------------------------

def extract_text_from_pdf(path):
    text = ""

    with open(path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            text += page.extract_text()

    return text


# -----------------------------
# Load Patient Documents
# -----------------------------

def load_patient_documents(folder):

    documents = {}

    for fname in os.listdir(folder):

        if fname.endswith(".pdf"):

            path = os.path.join(folder, fname)

            documents[fname] = extract_text_from_pdf(path)

    return documents


# -----------------------------
# Create Embeddings
# -----------------------------

def create_embeddings(documents):

    texts = list(documents.values())

    embeddings = embedding_model.encode(texts)

    return np.array(embeddings), texts


# -----------------------------
# Build FAISS Vector Store
# -----------------------------

def build_vector_store(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index


# -----------------------------
# Retrieve Similar Documents
# -----------------------------

def retrieve_context(query, index, texts, k=3):

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(query_embedding, k)

    results = []

    for idx in indices[0]:
        results.append(texts[idx])

    return results


# -----------------------------
# Query LLM
# -----------------------------

def ask_llm(context, question):

    prompt = f"""
You are an AI clinical decision support assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{question}

Provide safety-focused insights.
Do not provide medical diagnosis.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# -----------------------------
# Patient Trend Visualization
# -----------------------------

def plot_patient_trends():

    import matplotlib.pyplot as plt

    data = pd.DataFrame({
        "Day": [1, 2, 3, 4, 5],
        "HeartRate": [85, 90, 95, 100, 92],
        "Temperature": [98.4, 98.7, 99.1, 99.5, 98.9]
    })

    plt.figure(figsize=(8,4))

    plt.plot(data["Day"], data["HeartRate"], label="Heart Rate")
    plt.plot(data["Day"], data["Temperature"], label="Temperature")

    plt.xlabel("Day")
    plt.ylabel("Vitals")
    plt.title("Patient Stability Trends")

    plt.legend()

    plt.show()


# -----------------------------
# Main Pipeline
# -----------------------------

def main():

    print("Loading patient documents...")

    documents = load_patient_documents(PDF_FOLDER)

    print("Creating embeddings...")

    embeddings, texts = create_embeddings(documents)

    print("Building vector database...")

    index = build_vector_store(embeddings)

    while True:

        query = input("\nEnter clinical question (or 'exit'): ")

        if query == "exit":
            break

        context = retrieve_context(query, index, texts)

        combined_context = "\n".join(context)

        answer = ask_llm(combined_context, query)

        print("\nAI Copilot Response:\n")

        print(answer)


# -----------------------------
# Run
# -----------------------------

if __name__ == "__main__":

    main()