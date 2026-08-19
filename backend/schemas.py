from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    filename: str
    page: int
    score: float
    preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    retrieval_count: int


class UploadSummary(BaseModel):
    filename: str
    pages: int
    chunks: int


class UploadResponse(BaseModel):
    documents: list[UploadSummary]
    total_chunks_in_index: int
