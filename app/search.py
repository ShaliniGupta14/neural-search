import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    DOCUMENTS_PATH,
    INDEX_INFO_PATH,
    INDEX_PATH,
    check_openmp_runtime,
    resolve_device,
)
from app.core import InvalidQueryError, SearchResult, validate_query, validate_top_k  # noqa: F401


class IndexNotFoundError(FileNotFoundError):
    pass


class IndexCorruptedError(RuntimeError):
    pass


def _load_documents(path: Path, expected_count: int) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8") as f:
            documents = json.load(f)
    except json.JSONDecodeError as exc:
        raise IndexCorruptedError(f"Document metadata at {path} is not valid JSON.") from exc

    if not isinstance(documents, list) or len(documents) != expected_count:
        raise IndexCorruptedError(
            f"Document metadata has {len(documents) if isinstance(documents, list) else 'invalid'} "
            f"entries but the FAISS index has {expected_count} vectors. Rebuild the index."
        )
    required = {"id", "title", "text"}
    for position, doc in enumerate(documents):
        if not isinstance(doc, dict) or not required <= doc.keys():
            raise IndexCorruptedError(f"Document metadata entry {position} is missing {required}.")
    return documents


class DenseSearcher:
    """Bi-encoder retrieval: embed the query, then exact inner-product search over normalized vectors."""

    def __init__(self, model: SentenceTransformer, index: faiss.Index, documents: list[dict[str, str]]):
        self.model = model
        self.index = index
        self.documents = documents

    @classmethod
    def load(
        cls,
        index_path: Path = INDEX_PATH,
        documents_path: Path = DOCUMENTS_PATH,
        info_path: Path = INDEX_INFO_PATH,
        device: str = "cpu",
    ) -> "DenseSearcher":
        check_openmp_runtime()
        for path in (index_path, documents_path, info_path):
            if not path.exists():
                raise IndexNotFoundError(f"{path} not found. Build it with: python -m scripts.build_index")

        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            model_name = info["embedding_model"]
            dimension = int(info["dimension"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise IndexCorruptedError(f"Index info at {info_path} is unreadable. Rebuild the index.") from exc

        try:
            index = faiss.read_index(str(index_path))
        except RuntimeError as exc:
            raise IndexCorruptedError(f"FAISS index at {index_path} could not be read.") from exc
        if index.d != dimension:
            raise IndexCorruptedError(f"Index dimension {index.d} does not match index info ({dimension}).")

        documents = _load_documents(documents_path, index.ntotal)

        try:
            model = SentenceTransformer(model_name, device=resolve_device(device))
        except OSError as exc:
            raise RuntimeError(f"Could not load embedding model '{model_name}'.") from exc
        if model.get_embedding_dimension() != dimension:
            raise IndexCorruptedError(f"Model '{model_name}' dimension does not match the index.")

        return cls(model, index, documents)

    @property
    def num_documents(self) -> int:
        return self.index.ntotal

    def encode(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def search_vectors(self, vectors: np.ndarray, top_k: int) -> list[list[SearchResult]]:
        top_k = min(top_k, self.num_documents)
        scores, indices = self.index.search(vectors, top_k)
        batches = []
        for row_scores, row_indices in zip(scores, indices):
            results = []
            for rank, (score, idx) in enumerate(zip(row_scores, row_indices), start=1):
                if idx < 0:  # FAISS pads with -1 when fewer than top_k vectors exist
                    continue
                doc = self.documents[idx]
                results.append(
                    SearchResult(
                        document_id=doc["id"],
                        title=doc["title"],
                        text=doc["text"],
                        dense_score=float(score),
                        dense_rank=rank,
                    )
                )
            batches.append(results)
        return batches

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        query = validate_query(query)
        top_k = validate_top_k(top_k, maximum=self.num_documents)
        return self.search_vectors(self.encode([query]), top_k)[0]
