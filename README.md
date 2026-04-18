# Multi-Modal Document Intelligence (RAG-Based QA System)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for question answering over complex PDF documents.

The system processes real-world reports (e.g., IMF country reports) that contain rich textual and structured information. It extracts text, splits it into chunks, converts them into embeddings, and retrieves relevant content to answer user queries.

---

## Objectives
- Build a document ingestion pipeline for PDF parsing  
- Implement a smart chunking strategy  
- Generate embeddings for semantic search  
- Develop a retrieval system using similarity metrics  
- Implement a QA system grounded in retrieved context  
- Provide a user interface  

---

## System Architecture
The system follows a modular pipeline:

1. Document Ingestion  
2. Text Chunking  
3. Embedding Generation  
4. Semantic Retrieval  
5. Answer Generation  
6. User Interface  

---

## Project Structure
│
├── ingestion/
├── chunking/
├── embedding/
├── retrieval/
├── qa/
├── app/
├── data/
├── outputs/
├── requirements.txt
└── README.md

---

## Technologies Used
- PyMuPDF (fitz) – PDF text extraction  
- SentenceTransformers – embeddings  
- NumPy – similarity computation  
- HuggingFace Transformers – text generation  
- Streamlit – UI  

---

## Methodology

### 1. Document Ingestion
Extract text from PDFs using PyMuPDF.

### 2. Chunking
Split text into smaller segments to improve retrieval quality.

### 3. Embedding
Convert each chunk into vector representations using `all-MiniLM-L6-v2`.

### 4. Retrieval
Use cosine similarity to find the most relevant chunks.

### 5. Answer Generation
Generate answers using a language model based on retrieved context.

---

## Installation
```bash
pip install -r requirements.txt

python ingestion/extract_pdf.py
python chunking/chunk_text.py
python embedding/embed_chunks.py
python qa/generate_answer.py

streamlit run app/streamlit_app.py
```

### Example Queries
What is the inflation rate in Egypt?
What are the main economic challenges?
What is GDP growth?


### Output
Retrieved context
Generated answer
Source pages

Outputs are saved in .json and .txt files.

### Limitations
No image/table understanding yet
Basic language model
Text-only retrieval
### Future Work
Add image embeddings (CLIP / ColPali)
Improve table extraction
Use stronger LLMs
