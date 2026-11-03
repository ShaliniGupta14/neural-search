from pydantic import BaseModel, Field, field_validator, model_validator

from app.core import MAX_CANDIDATES, MAX_TOP_K


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, examples=["Does vitamin D reduce fracture risk?"])
    top_k: int = Field(5, ge=1, le=MAX_TOP_K)
    rerank: bool = True
    # 20 had the best MRR@10 of the pool sizes evaluated, at roughly half the latency of 50 (see results/).
    candidates: int = Field(20, ge=1, le=MAX_CANDIDATES, description="Dense candidate pool size for reranking.")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def pool_covers_top_k(self) -> "SearchRequest":
        if self.rerank and self.candidates < self.top_k:
            raise ValueError("candidates must be >= top_k when rerank is true")
        return self


class SearchResultItem(BaseModel):
    rank: int
    document_id: str
    title: str
    text: str
    score: float
    dense_score: float
    dense_rank: int
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    reranked: bool
    results: list[SearchResultItem]
    latency_ms: float
    timings_ms: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    num_documents: int
    embedding_model: str
    reranker_model: str | None
    devices: dict[str, str]
