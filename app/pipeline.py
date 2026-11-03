import time

from app.core import SearchOutput, validate_query, validate_search_args
from app.reranker import CrossEncoderReranker
from app.search import DenseSearcher


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
        top_k, candidates = validate_search_args(top_k, rerank, candidates, self.reranker is not None)

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
