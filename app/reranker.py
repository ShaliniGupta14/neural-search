from dataclasses import replace

from sentence_transformers import CrossEncoder

from app.config import RERANKER_MODEL_NAME, resolve_device
from app.core import SearchResult


class CrossEncoderReranker:
    """Scores (query, document) pairs jointly; slower than the bi-encoder but more precise."""

    def __init__(self, model_name: str = RERANKER_MODEL_NAME, device: str = "cpu", batch_size: int = 32):
        try:
            self.model = CrossEncoder(model_name, device=resolve_device(device))
        except OSError as exc:
            raise RuntimeError(f"Could not load reranker model '{model_name}'.") from exc
        self.model_name = model_name
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, f"{c.title}. {c.text}" if c.title else c.text) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        rescored = [replace(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
        # Stable sort keeps dense order as the tie-breaker.
        rescored.sort(key=lambda r: r.rerank_score, reverse=True)
        return rescored[:top_k]
