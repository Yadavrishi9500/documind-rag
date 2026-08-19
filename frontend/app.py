import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="DocuMind RAG",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 DocuMind RAG")
st.caption("Ask questions across multiple PDFs with semantic retrieval and page-level citations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📚 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDF files.",
    )

    if st.button("Index documents", use_container_width=True, disabled=not uploaded_files):
        files = [
            ("files", (file.name, file.getvalue(), "application/pdf"))
            for file in uploaded_files
        ]

        with st.spinner("Extracting, chunking and embedding documents..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/documents/upload",
                    files=files,
                    timeout=600,
                )
                response.raise_for_status()
                payload = response.json()

                st.success(
                    f"Indexed {len(payload['documents'])} document(s) "
                    f"({payload['total_chunks_in_index']} total chunks)."
                )
            except requests.RequestException as exc:
                detail = ""
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    pass
                st.error(detail or f"Upload failed: {exc}")

    st.divider()

    try:
        docs_response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        docs_response.raise_for_status()
        docs = docs_response.json()["documents"]

        if docs:
            st.subheader("Indexed")
            for doc in docs:
                st.write(f"**{doc['filename']}**")
                st.caption(f"{doc['pages_indexed']} pages • {doc['chunks']} chunks")
        else:
            st.info("No documents indexed yet.")
    except requests.RequestException:
        st.warning("Backend is not reachable.")

    st.divider()

    if st.button("Clear knowledge base", use_container_width=True):
        try:
            requests.delete(f"{BACKEND_URL}/documents", timeout=10).raise_for_status()
            st.session_state.messages = []
            st.success("Knowledge base cleared.")
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"Could not clear index: {exc}")

with st.expander("How it works"):
    st.markdown(
        """
        **PDF → text extraction → overlapping chunks → local embeddings → semantic search → LLM answer**

        The LLM receives only the most relevant retrieved chunks and is instructed
        to ground its answer in those sources.
        """
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved sources"):
                for idx, source in enumerate(message["sources"], start=1):
                    st.markdown(
                        f"**Source {idx} — {source['filename']}, page {source['page']}**  \n"
                        f"Similarity: `{source['score']:.3f}`"
                    )
                    st.caption(source["preview"])
                    st.divider()

question = st.chat_input("Ask something about your uploaded documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question},
                    timeout=180,
                )
                response.raise_for_status()
                payload = response.json()

                st.markdown(payload["answer"])

                if payload["sources"]:
                    with st.expander("View retrieved sources"):
                        for idx, source in enumerate(payload["sources"], start=1):
                            st.markdown(
                                f"**Source {idx} — {source['filename']}, page {source['page']}**  \n"
                                f"Similarity: `{source['score']:.3f}`"
                            )
                            st.caption(source["preview"])
                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": payload["answer"],
                        "sources": payload["sources"],
                    }
                )

            except requests.RequestException as exc:
                detail = ""
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    pass
                st.error(detail or f"Request failed: {exc}")
