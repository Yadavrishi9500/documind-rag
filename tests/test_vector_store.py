from pathlib import Path

import numpy as np

from backend.services.vector_store import VectorStore


def test_empty_store_returns_no_results(tmp_path: Path):
    store = VectorStore(tmp_path, "unused-model")
    assert store.search("anything") == []
    assert store.size == 0


def test_clear_removes_persisted_index(tmp_path: Path):
    store = VectorStore(tmp_path, "unused-model")

    store.embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    store.metadata = [
        {
            "text": "Example chunk",
            "filename": "a.pdf",
            "page": 1,
            "chunk_id": "a.pdf:p1:c0",
        }
    ]
    store._save()

    assert store.index_path.exists()
    assert store.metadata_path.exists()

    store.clear()

    assert store.size == 0
    assert not store.index_path.exists()
    assert not store.metadata_path.exists()
