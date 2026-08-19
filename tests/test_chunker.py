import pytest

from backend.services.chunker import split_text


def test_short_text_returns_single_chunk():
    chunks = split_text(
        "RAG combines retrieval with generation.",
        filename="notes.pdf",
        page=1,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0].filename == "notes.pdf"
    assert chunks[0].page == 1
    assert "retrieval" in chunks[0].text


def test_long_text_creates_overlapping_chunks():
    text = " ".join(f"word{i}" for i in range(200))

    chunks = split_text(
        text,
        filename="long.pdf",
        page=2,
        chunk_size=160,
        overlap=40,
    )

    assert len(chunks) > 1
    assert all(chunk.page == 2 for chunk in chunks)
    assert all(chunk.filename == "long.pdf" for chunk in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text(
            "hello world",
            filename="x.pdf",
            page=1,
            chunk_size=100,
            overlap=100,
        )
