from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    filename: str
    page: int
    chunk_id: str


def split_text(
    text: str,
    filename: str,
    page: int,
    chunk_size: int = 900,
    overlap: int = 180,
) -> list[Chunk]:
    """Split one page into overlapping character chunks while preserving metadata."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))

        # Try to end on a sentence/word boundary when possible.
        if end < len(cleaned):
            boundary = max(
                cleaned.rfind(". ", start, end),
                cleaned.rfind("? ", start, end),
                cleaned.rfind("! ", start, end),
                cleaned.rfind(" ", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk_text = cleaned[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    text=chunk_text,
                    filename=filename,
                    page=page,
                    chunk_id=f"{filename}:p{page}:c{index}",
                )
            )
            index += 1

        if end >= len(cleaned):
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks
