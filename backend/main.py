from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.schemas import AskRequest, AskResponse, Source, UploadResponse, UploadSummary
from backend.services.pdf_service import extract_pdf_chunks
from backend.services.rag_service import RAGService
from backend.services.vector_store import VectorStore


app = FastAPI(
    title="DocuMind RAG API",
    description="Multi-document RAG backend with grounded answers and page citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore(settings.data_dir, settings.embedding_model)
rag = RAGService(store)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": store.size,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.openai_model,
    }


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)):
    summaries: list[UploadSummary] = []

    for file in files:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        data = await file.read()

        try:
            chunks, page_count = extract_pdf_chunks(data, file.filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=f"No extractable text found in {file.filename}. "
                       "It may be a scanned/image-only PDF.",
            )

        added = store.add_chunks(chunks)
        summaries.append(
            UploadSummary(
                filename=file.filename,
                pages=page_count,
                chunks=added,
            )
        )

    return UploadResponse(
        documents=summaries,
        total_chunks_in_index=store.size,
    )


@app.get("/documents")
def list_documents():
    return {"documents": store.list_documents(), "total_chunks": store.size}


@app.delete("/documents")
def clear_documents():
    store.clear()
    return {"message": "Document index cleared."}


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    if store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents are indexed. Upload PDFs first.",
        )

    try:
        answer, retrieved = rag.ask(payload.question, payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM request failed: {exc}") from exc

    sources = [
        Source(
            filename=item["filename"],
            page=item["page"],
            score=round(item["score"], 4),
            preview=item["text"][:360] + ("..." if len(item["text"]) > 360 else ""),
        )
        for item in retrieved
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        retrieval_count=len(sources),
    )
