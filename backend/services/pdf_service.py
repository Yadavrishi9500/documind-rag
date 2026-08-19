import fitz

from backend.services.chunker import Chunk, split_text


def extract_pdf_chunks(
    data: bytes,
    filename: str,
    chunk_size: int = 900,
    overlap: int = 180,
) -> tuple[list[Chunk], int]:
    if not data:
        raise ValueError("Empty PDF")

    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    chunks: list[Chunk] = []

    try:
        page_count = document.page_count
        for page_index in range(page_count):
            page = document.load_page(page_index)
            text = page.get_text("text")
            chunks.extend(
                split_text(
                    text=text,
                    filename=filename,
                    page=page_index + 1,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
    finally:
        document.close()

    return chunks, page_count
