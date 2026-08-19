# DocuMind RAG 🧠📚

A multi-document **Retrieval-Augmented Generation (RAG)** research assistant that lets users upload PDFs and ask grounded questions with **page-level citations**.

Instead of sending entire documents to an LLM, DocuMind:
1. extracts PDF text page-by-page,
2. chunks the content,
3. creates local semantic embeddings,
4. retrieves the most relevant chunks for each question,
5. sends only that evidence to the LLM,
6. returns an answer with source references.

## Why this project is useful

LLMs can hallucinate or answer from general knowledge. This project constrains generation to retrieved document evidence and explicitly tells the model to say when the answer is not supported.

## Features

- Multi-PDF ingestion
- Page-aware text extraction
- Overlapping text chunking
- Local embeddings with Sentence Transformers
- Persistent vector index using NumPy
- Cosine-similarity semantic retrieval
- FastAPI backend
- Streamlit chat interface
- Page-level citations and source previews
- Retrieval confidence scores
- Grounded-answer prompt guardrail
- Index reset endpoint
- Unit tests
- GitHub Actions CI
- Docker support

## Architecture

```text
                 ┌───────────────┐
                 │   Streamlit   │
                 │      UI       │
                 └───────┬───────┘
                         │ HTTP
                         ▼
                 ┌───────────────┐
                 │    FastAPI    │
                 │    Backend    │
                 └───────┬───────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   PDF Ingestion                   User Question
          │                             │
          ▼                             ▼
   Page Extraction                Query Embedding
          │                             │
          ▼                             ▼
      Chunking                 Similarity Retrieval
          │                             │
          ▼                             │
 Document Embeddings                    │
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  Retrieved Context
                         │
                         ▼
                    OpenAI LLM
                         │
                         ▼
                 Answer + Citations
```

## Tech Stack

`Python` `FastAPI` `Streamlit` `Sentence Transformers` `NumPy` `PyMuPDF` `OpenAI API` `Docker` `Pytest`

## Project Structure

```text
documind-rag/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   └── services/
│       ├── chunker.py
│       ├── pdf_service.py
│       ├── rag_service.py
│       └── vector_store.py
├── frontend/
│   └── app.py
├── tests/
│   ├── test_chunker.py
│   └── test_vector_store.py
├── .github/workflows/ci.yml
├── .env.example
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── requirements.txt
```

## How RAG Works Here

### 1. Ingestion
Each PDF is read page-by-page using PyMuPDF. Page numbers are preserved as metadata.

### 2. Chunking
Long pages are split into overlapping chunks. Overlap reduces the chance of losing context at chunk boundaries.

### 3. Embeddings
Chunks are converted into normalized dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.

### 4. Retrieval
The user's question is embedded using the same model. The vector index computes semantic similarity and returns the top matching chunks.

### 5. Grounded Generation
Retrieved chunks are formatted as evidence and sent to the LLM. The system prompt instructs the model to answer only from that evidence and cite sources.

## Quick Start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/YOUR_USERNAME/documind-rag.git
cd documind-rag

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and add your OpenAI API key.

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-mini
```

Never commit your real `.env` file.

### 4. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

API docs:

```text
http://localhost:8000/docs
```

### 5. Start the frontend

Open another terminal:

```bash
streamlit run frontend/app.py
```

Then upload PDFs and ask questions.

## Docker

```bash
docker compose up --build
```

Frontend:

```text
http://localhost:8501
```

Backend:

```text
http://localhost:8000
```

## Example Questions

After uploading technical documents:

- "Summarize the main findings across these documents."
- "What limitations do the authors mention?"
- "Compare the approaches described in the uploaded papers."
- "What evidence supports the main conclusion?"
- "Which page discusses deployment constraints?"

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/documents/upload` | Ingest one or more PDFs |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents` | Clear the vector index |
| `POST` | `/ask` | Ask a grounded question |

## Testing

```bash
pytest -q
```

The CI workflow automatically runs tests on pushes and pull requests.

## Design Decisions

### Why local embeddings?
Using a local embedding model reduces API cost and makes the retrieval layer independently reproducible.

### Why not LangChain?
The core RAG pipeline is implemented directly so the retrieval, chunking, ranking, metadata, and prompting logic are visible and easy to explain.

### Why normalized vectors?
Normalized embeddings let dot-product similarity act like cosine similarity, keeping retrieval simple and efficient.

## Limitations

- Scanned/image-only PDFs need OCR before indexing.
- The lightweight NumPy index is designed for small/medium document collections, not millions of chunks.
- Retrieval quality depends on document quality and chunking.
- This version uses dense retrieval only; a future version could combine BM25 + dense retrieval.

## Future Improvements

- Hybrid BM25 + vector search
- Cross-encoder reranking
- Conversation memory
- DOCX/TXT ingestion
- OCR for scanned PDFs
- Evaluation dataset with Recall@K / MRR
- PostgreSQL/pgvector persistence
- User authentication and per-user indexes

## What I Learned

This project demonstrates the full RAG pipeline: document ingestion, chunking, embeddings, semantic retrieval, metadata-aware citations, grounded prompting, API design, frontend integration, testing, and containerization.
