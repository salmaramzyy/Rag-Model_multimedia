Multi-Modal Document Intelligence (RAG-Based QA System)
Overview

This project implements a Retrieval-Augmented Generation (RAG) system for question answering over complex PDF documents. The system is designed to process real-world reports (e.g., IMF country reports) that contain rich textual information and structured content.

The pipeline extracts text from PDF documents, segments it into meaningful chunks, converts them into embeddings, and retrieves relevant information to answer user queries. The system includes a command-line interface and a Streamlit-based interactive application.

Objectives
Build a document ingestion pipeline for PDF parsing
Implement a smart chunking strategy for efficient retrieval
Generate embeddings for semantic search
Develop a retrieval system using similarity metrics
Implement a QA system grounded in retrieved context
Provide a user interface for interaction
System Architecture

The system follows a modular pipeline:

Document Ingestion
Text Chunking
Embedding Generation
Semantic Retrieval
Answer Generation
User Interface
Project Structure
Assignment1_Multimedia/
│
├── ingestion/
│   └── extract_pdf.py
│
├── chunking/
│   └── chunk_text.py
│
├── embedding/
│   └── embed_chunks.py
│
├── retrieval/
│   └── search.py
│
├── qa/
│   └── generate_answer.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── pdfs/            (not included in repository)
│
├── outputs/             (generated files, ignored)
│
├── requirements.txt
├── README.md
└── .gitignore
Technologies Used
PyMuPDF (fitz): PDF text extraction
SentenceTransformers: Text embeddings
NumPy: Vector operations and similarity computation
Transformers (Hugging Face): Local text generation model
Streamlit: Interactive user interface
Methodology
1. Document Ingestion

PDF files are parsed using PyMuPDF. Text is extracted page by page and stored in structured format.

2. Chunking Strategy

Each page is divided into smaller chunks (approximately 300 words) to improve retrieval performance and maintain semantic coherence.

3. Embedding Generation

Each chunk is converted into a dense vector using the SentenceTransformer model (all-MiniLM-L6-v2).

4. Retrieval

User queries are embedded and compared with stored embeddings using cosine similarity. The top-k most relevant chunks are retrieved.

5. Answer Generation

A local language model (DistilGPT2) generates answers based on retrieved context.

6. User Interface

A Streamlit application allows users to interact with the system, view retrieved content, and see generated answers with source references.

Installation

Clone the repository and install dependencies:

pip install -r requirements.txt
Usage
Step 1: Extract text from PDF
python ingestion/extract_pdf.py
Step 2: Chunk the text
python chunking/chunk_text.py
Step 3: Generate embeddings
python embedding/embed_chunks.py
Step 4: Run QA (CLI)
python qa/generate_answer.py
Step 5: Run the web interface
streamlit run app/streamlit_app.py
Example Queries
What is the inflation rate in Egypt?
What are the main economic challenges?
What is the GDP growth outlook?
Output

The system provides:

Retrieved relevant text chunks
Generated answer
Source attribution (page numbers)

Outputs are also saved in structured formats (.json and .txt) for evaluation and reproducibility.

Limitations
Image and table content are not fully processed
Local language model has limited reasoning capability
Retrieval is based only on text embeddings
Future Work
Integrate image embeddings (e.g., CLIP)
Add table extraction and structured querying
Use more advanced language models for answer generation
Incorporate layout-aware models such as ColPali
Author

Salma Wael
