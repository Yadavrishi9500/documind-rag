from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.services.chunker import Chunk


class VectorStore:
    def __init__(self, data_dir: Path, model_name: str):
        self.data_dir = data_dir
        self.index_path = data_dir / "index.npy"
        self.metadata_path = data_dir / "metadata.json"
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.metadata: list[dict] = []
        self._load()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            self.embeddings = np.load(self.index_path).astype(np.float32)
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

            if len(self.metadata) != len(self.embeddings):
                raise RuntimeError("Vector index and metadata are out of sync.")

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_path, self.embeddings)
        self.metadata_path.write_text(
            json.dumps(self.metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        chunks = list(chunks)
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        new_embeddings = self.model.encode_document(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            if self.embeddings.shape[1] != new_embeddings.shape[1]:
                raise RuntimeError("Embedding dimensions do not match.")
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.metadata.extend(
            {
                "text": chunk.text,
                "filename": chunk.filename,
                "page": chunk.page,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunks
        )

        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.embeddings.size == 0 or not self.metadata:
            return []

        query_vector = self.model.encode_query(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        scores = self.embeddings @ query_vector
        top_k = min(top_k, len(scores))
        indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []
        for idx in indices:
            item = dict(self.metadata[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)
        return results

    def list_documents(self) -> list[dict]:
        by_name: dict[str, dict] = {}
        for item in self.metadata:
            name = item["filename"]
            if name not in by_name:
                by_name[name] = {"filename": name, "pages": set(), "chunks": 0}
            by_name[name]["pages"].add(item["page"])
            by_name[name]["chunks"] += 1

        return [
            {
                "filename": value["filename"],
                "pages_indexed": len(value["pages"]),
                "chunks": value["chunks"],
            }
            for value in sorted(by_name.values(), key=lambda x: x["filename"].lower())
        ]

    def clear(self) -> None:
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.metadata = []

        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

    @property
    def size(self) -> int:
        return len(self.metadata)
