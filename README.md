# DocuMind RAG 🧠📚

### Local Multi-Document Retrieval-Augmented Generation Assistant

[![CI](https://github.com/Yadavrishi9500/documind-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Yadavrishi9500/documind-rag/actions)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit\&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black)
![RAG](https://img.shields.io/badge/Architecture-RAG-purple)

**DocuMind RAG** is a privacy-focused AI research assistant that allows users to upload multiple PDF documents and ask questions about them using **Retrieval-Augmented Generation (RAG)**.

Instead of relying only on an LLM's internal knowledge, DocuMind retrieves the most relevant sections from the uploaded documents and supplies them as context to a locally running **Llama 3.2 model through Ollama**.

The result is a grounded answer with **page-level citations, source previews, and semantic similarity scores**.

> 🔒 Fully local generation — no paid LLM API key required.

---
## 🎥 Demo

### Watch DocuMind RAG in Action

[![Watch DocuMind RAG Demo](docs/screenshots/home.png)](https://lnkd.in/p/gQq-AJts)

▶️ **Click the image above to watch the demo video**

### Grounded Answer with Retrieved Sources

![DocuMind RAG Answer](docs/screenshots/answer.png)

---

## ✨ Key Features

* 📄 **Multi-PDF Upload** — index multiple documents into one knowledge base
* 🧩 **Page-Aware Chunking** — preserves document and page metadata
* 🧠 **Local Embeddings** — generated using Sentence Transformers
* 🔍 **Semantic Retrieval** — finds chunks based on meaning rather than keyword matching
* 📚 **Persistent Vector Index** — lightweight NumPy-based vector storage
* 🤖 **Local LLM Generation** — powered by Llama 3.2 through Ollama
* 🔗 **Page-Level Citations** — answers reference the retrieved document pages
* 👀 **Source Previews** — inspect the context used to generate each answer
* 📊 **Similarity Scores** — view retrieval confidence for matched chunks
* 🛡️ **Hallucination Guardrail** — refuses unsupported questions when evidence is missing
* ⚡ **FastAPI Backend** — clean REST API for ingestion and querying
* 💬 **Streamlit Interface** — simple interactive chat experience
* 🧪 **Pytest Tests** — unit tests for core components
* ✅ **GitHub Actions CI** — automated testing on pushes and pull requests

---

## 🏗️ System Architecture

```text
                         User
                          │
                          ▼
                ┌──────────────────┐
                │    Streamlit     │
                │     Frontend     │
                └────────┬─────────┘
                         │ HTTP
                         ▼
                ┌──────────────────┐
                │     FastAPI      │
                │      Backend     │
                └────────┬─────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       PDF Ingestion            User Query
              │                     │
              ▼                     ▼
       Text Extraction        Query Embedding
              │                     │
              ▼                     ▼
          Chunking          Semantic Similarity
              │                   Search
              ▼                     │
      Document Embeddings           │
              │                     │
              └──────────┬──────────┘
                         ▼
                  Top-K Retrieval
                         │
                         ▼
                  Retrieved Context
                         │
                         ▼
                 Ollama + Llama 3.2
                         │
                         ▼
                 Grounded Response
                  + Page Citations
```

---

## 🧰 Tech Stack

### AI / RAG

`Llama 3.2` `Ollama` `Sentence Transformers` `Semantic Search` `RAG`

### Backend

`Python` `FastAPI` `Pydantic` `Requests`

### Document Processing

`PyMuPDF`

### Vector Search

`NumPy` `Cosine Similarity`

### Frontend

`Streamlit`

### Testing & Automation

`Pytest` `GitHub Actions`

---

## 🧠 How the RAG Pipeline Works

### 1. Document Ingestion

Users upload one or more PDF files.

DocuMind uses **PyMuPDF** to extract text page-by-page while preserving:

* document filename
* page number
* extracted page content

This metadata is later used for citations.

---

### 2. Text Chunking

Large sections of text are divided into smaller overlapping chunks.

```text
Document Page
      ↓
Chunk 1
      ↓ overlap
Chunk 2
      ↓ overlap
Chunk 3
```

The overlap reduces context loss when important information appears near chunk boundaries.

---

### 3. Embedding Generation

Each text chunk is converted into a dense numerical vector using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

These vectors capture the **semantic meaning** of the text.

For example:

```text
"How does the system detect vehicles?"
```

can retrieve text discussing:

```text
"Vehicle detection is performed using a YOLO-based model..."
```

even when the wording is different.

---

### 4. Semantic Retrieval

The user's question is converted into an embedding using the same model.

DocuMind compares the query embedding against stored document embeddings and retrieves the most similar chunks.

```text
User Question
      ↓
Query Embedding
      ↓
Vector Similarity
      ↓
Top-K Relevant Chunks
```

The embeddings are normalized, allowing dot-product similarity to behave like cosine similarity.

---

### 5. Context Construction

The retrieved chunks are combined with metadata such as:

```text
[Source 1]
File: research-paper.pdf
Page: 4
<retrieved text>

[Source 2]
File: report.pdf
Page: 12
<retrieved text>
```

Only this relevant context is supplied to the LLM.

---

### 6. Local LLM Generation

DocuMind sends the retrieved context and user question to:

```text
Ollama
└── llama3.2:1b
```

The model is instructed to:

* answer only using the supplied evidence
* cite retrieved sources
* avoid inventing facts or page numbers
* explicitly say when enough evidence cannot be found

Example fallback:

```text
I couldn't find enough evidence in the uploaded documents to answer that.
```

---

## 📂 Project Structure

```text
documind-rag/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── chunker.py
│       ├── pdf_service.py
│       ├── rag_service.py
│       └── vector_store.py
│
├── frontend/
│   └── app.py
│
├── tests/
│   ├── test_chunker.py
│   └── test_vector_store.py
│
├── docs/
│   └── screenshots/
│       ├── home.png
│       └── answer.png
│
├── data/
│   └── .gitkeep
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .env.example
├── .gitignore
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

Make sure you have:

* **Python 3.11+**
* **Ollama**
* Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/Yadavrishi9500/documind-rag.git
cd documind-rag
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

You can use the environment's Python directly:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Install Ollama

Install Ollama and download the local model:

```bash
ollama pull llama3.2:1b
```

Verify that the model works:

```bash
ollama run llama3.2:1b
```

Then type:

```text
hello
```

Exit with:

```text
/bye
```

---

## 4. Start the FastAPI Backend

### Windows

```bash
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

The backend will run at:

```text
http://localhost:8000
```

FastAPI Swagger documentation:

```text
http://localhost:8000/docs
```

---

## 5. Start the Streamlit Frontend

Open another terminal.

### Windows

```bash
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

---

## 6. Use DocuMind

1. Upload one or more PDF files
2. Click **Index documents**
3. Wait for the embeddings to be generated
4. Ask questions about the uploaded documents
5. Expand **View retrieved sources** to inspect citations and similarity scores

---

## 💡 Example Questions

```text
Summarize the main findings of this document.
```

```text
What limitations are discussed?
```

```text
Compare the approaches described in these documents.
```

```text
What evidence supports the main conclusion?
```

```text
Which page discusses deployment constraints?
```

You can also deliberately ask an unrelated question to test the grounding guardrail.

For example:

```text
Who won the FIFA World Cup in 2018?
```

If the uploaded documents contain no relevant information, DocuMind should avoid answering from general model knowledge.

---

## 🔌 API Endpoints

| Method   | Endpoint            | Description                      |
| -------- | ------------------- | -------------------------------- |
| `GET`    | `/health`           | Check API and index status       |
| `POST`   | `/documents/upload` | Upload and index PDF documents   |
| `GET`    | `/documents`        | List indexed documents           |
| `DELETE` | `/documents`        | Clear the current knowledge base |
| `POST`   | `/ask`              | Ask a grounded question          |

---

## 🧪 Testing

Run the test suite:

### Windows

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

The repository also includes a **GitHub Actions CI workflow** that automatically runs tests whenever code is pushed or a pull request is opened.

---

## 🛡️ Grounding & Hallucination Control

One of the goals of DocuMind is to reduce unsupported LLM answers.

Before generation:

```text
Question
   ↓
Semantic Retrieval
   ↓
Similarity Threshold
   ↓
Relevant Evidence?
   ├── No → Refuse unsupported answer
   │
   └── Yes
        ↓
     Local LLM
        ↓
 Answer + Sources
```

The system combines:

* semantic retrieval
* similarity filtering
* explicit system instructions
* source metadata
* page references

to encourage evidence-based responses.

---

## 🎯 Why I Built This

I built DocuMind as a rapid hands-on project to understand **how Retrieval-Augmented Generation actually works internally**.

Rather than hiding the complete pipeline behind a high-level RAG framework, the major components are implemented directly:

```text
PDF Extraction
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Storage
      ↓
Semantic Retrieval
      ↓
Context Construction
      ↓
Local LLM
      ↓
Grounded Answer + Citations
```

Building the working prototype helped me understand how factors such as **chunk size, chunk overlap, embeddings, similarity thresholds, retrieval quality, metadata, and prompting** affect RAG performance.

The first working version was built during an approximately **2-hour development sprint**, then refined with testing, documentation, and repository automation.

---

## ⚠️ Current Limitations

* Scanned/image-only PDFs require OCR
* Current vector storage is optimized for small-to-medium document collections
* Retrieval uses dense semantic search only
* The lightweight Llama 3.2 1B model prioritizes local execution over maximum reasoning quality
* Conversation history is not yet incorporated into retrieval
* Documents are currently stored in a shared local index

---

## 🔮 Future Improvements

* 🔎 Hybrid **BM25 + dense vector retrieval**
* 🎯 Cross-encoder reranking
* 🗄️ PostgreSQL + pgvector
* 🧠 Conversation memory
* 📄 DOCX and TXT support
* 👁️ OCR for scanned PDFs
* 📊 RAG evaluation using Recall@K, MRR and answer-groundedness metrics
* 🔐 User authentication
* 👤 Per-user knowledge bases
* 🧩 Larger local LLM support
* 🌐 Production deployment

---

## 📚 What This Project Demonstrates

DocuMind demonstrates practical experience with:

* Retrieval-Augmented Generation
* Local LLM integration
* Embedding models
* Semantic search
* Vector similarity
* Document processing
* Prompt grounding
* REST API development
* Frontend/backend integration
* Testing
* CI automation

---

## 👨‍💻 Author

### Rishi Yadav

Computer Science student and software developer interested in:

**Artificial Intelligence • Computer Vision • Backend Engineering • Full-Stack Development • Generative AI**

GitHub: [Yadavrishi9500](https://github.com/Yadavrishi9500)

---

⭐ **If you found DocuMind interesting, consider starring the repository.**
