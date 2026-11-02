import time
from dataclasses import dataclass, field

from app.config import MAX_CANDIDATES, MAX_TOP_K
from app.reranker import CrossEncoderReranker
from app.search import DenseSearcher, InvalidQueryError, SearchResult, validate_query, validate_top_k


@dataclass
class SearchOutput:
    results: list[SearchResult]
    reranked: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


class SearchEngine:
    """Dense retrieval followed by optional cross-encoder reranking, with per-stage timings."""

    def __init__(self, searcher: DenseSearcher, reranker: CrossEncoderReranker | None):
        self.searcher = searcher
        self.reranker = reranker

    @classmethod
    def load(
        cls, embed_device: str = "cpu", rerank_device: str = "auto", with_reranker: bool = True
    ) -> "SearchEngine":
        # Defaults reflect measurements on Apple Silicon: encoding a single short query is fastest on
        # CPU, while scoring dozens of long (query, abstract) pairs benefits from MPS.
        searcher = DenseSearcher.load(device=embed_device)
        reranker = CrossEncoderReranker(device=rerank_device) if with_reranker else None
        return cls(searcher, reranker)

    @property
    def devices(self) -> dict[str, str]:
        devices = {"embedder": str(self.searcher.model.device)}
        if self.reranker is not None:
            devices["reranker"] = str(self.reranker.model.device)
        return devices

    def search(self, query: str, top_k: int = 10, rerank: bool = True, candidates: int = 50) -> SearchOutput:
        query = validate_query(query)
        top_k = validate_top_k(top_k, MAX_TOP_K)
        if rerank and self.reranker is None:
            raise InvalidQueryError("Reranking requested but no reranker is loaded.")
        if rerank:
            candidates = validate_top_k(candidates, MAX_CANDIDATES)
            if candidates < top_k:
                raise InvalidQueryError("candidates must be >= top_k when reranking.")

        start = time.perf_counter()
        vector = self.searcher.encode([query])
        encoded = time.perf_counter()
        retrieved = self.searcher.search_vectors(vector, candidates if rerank else top_k)[0]
        searched = time.perf_counter()

        timings = {
            "encode_ms": (encoded - start) * 1000,
            "retrieval_ms": (searched - encoded) * 1000,
        }
        if rerank:
            results = self.reranker.rerank(query, retrieved, top_k)
            timings["rerank_ms"] = (time.perf_counter() - searched) * 1000
        else:
            results = retrieved
        timings["total_ms"] = (time.perf_counter() - start) * 1000

        return SearchOutput(results=results, reranked=rerank, timings_ms=timings)
