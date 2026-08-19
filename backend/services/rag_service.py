import requests

from backend.config import settings
from backend.services.vector_store import VectorStore


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:1b"

SYSTEM_PROMPT = """You are DocuMind, a grounded document research assistant.

Rules:
1. Answer ONLY using the supplied document context.
2. If the context does not support an answer, say:
   "I couldn't find enough evidence in the uploaded documents to answer that."
3. Cite factual claims using source labels like [Source 1].
4. Never invent page numbers, facts, names, statistics, or quotations.
5. Keep answers clear and concise.
"""


class RAGService:
    def __init__(self, store: VectorStore):
        self.store = store

    def ask(self, question: str, top_k: int | None = None):
        k = top_k or settings.top_k

        retrieved = self.store.search(question, top_k=k)

        relevant = [
            item
            for item in retrieved
            if item["score"] >= settings.min_similarity
        ]

        if not relevant:
            return (
                "I couldn't find enough evidence in the uploaded documents to answer that.",
                [],
            )

        context_blocks = []

        for i, item in enumerate(relevant, start=1):
            context_blocks.append(
                f"[Source {i}] "
                f"File: {item['filename']} | "
                f"Page: {item['page']}\n"
                f"{item['text']}"
            )

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer using only the document context above.
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
            },
            timeout=180,
        )

        response.raise_for_status()

        data = response.json()

        answer = data["message"]["content"].strip()

        return answer, relevant