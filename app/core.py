"""Types and validation shared by every backend. Deliberately free of torch/faiss imports so the
lightweight ONNX deployment can use it."""

from dataclasses import dataclass, field

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_TOP_K = 100
MAX_CANDIDATES = 200


class InvalidQueryError(ValueError):
    pass


@dataclass(frozen=True)
class SearchResult:
    document_id: str
    title: str
    text: str
    dense_score: float
    dense_rank: int
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.dense_score


@dataclass
class SearchOutput:
    results: list[SearchResult]
    reranked: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


def validate_query(query: object) -> str:
    if not isinstance(query, str):
        raise InvalidQueryError("Query must be a string.")
    query = query.strip()
    if not query:
        raise InvalidQueryError("Query must not be empty.")
    return query


def validate_top_k(top_k: object, maximum: int = MAX_TOP_K) -> int:
    # bool is a subclass of int; reject it explicitly.
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise InvalidQueryError("top_k must be an integer.")
    if not 1 <= top_k <= maximum:
        raise InvalidQueryError(f"top_k must be between 1 and {maximum}.")
    return top_k


def validate_search_args(top_k: int, rerank: bool, candidates: int, has_reranker: bool) -> tuple[int, int]:
    top_k = validate_top_k(top_k, MAX_TOP_K)
    if rerank and not has_reranker:
        raise InvalidQueryError("Reranking requested but no reranker is loaded.")
    if rerank:
        candidates = validate_top_k(candidates, MAX_CANDIDATES)
        if candidates < top_k:
            raise InvalidQueryError("candidates must be >= top_k when reranking.")
    return top_k, candidates
