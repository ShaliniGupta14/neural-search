"""Torch-free search engine on ONNX Runtime, for serverless hosts with tight bundle limits.

Uses the same models, document vectors, and ranking logic as the torch pipeline:
  - query embedding: MiniLM mean pooling over token states, then L2 normalization
  - retrieval: exact cosine (inner product on unit vectors) over the vectors from the FAISS index
  - reranking: cross-encoder logit per (query, "title. abstract") pair

Build the bundle with: python -m scripts.export_onnx
"""

import gzip
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from app.core import SearchOutput, SearchResult, validate_query, validate_search_args

DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parent.parent / "data" / "onnx"
EMBED_MAX_LENGTH = 256  # sentence-transformers max_seq_length for all-MiniLM-L6-v2
# Truncating abstracts to 256 tokens roughly halves reranking cost for a small quality loss on SciFact
# (MRR@10 0.658 -> 0.654); 128 tokens drops below dense-only. The serverless deployment sets 256.
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))


class BundleNotFoundError(FileNotFoundError):
    pass


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Serverless hosts can report more cores than the CPU quota allows; oversubscribed threads thrash.
    if threads := int(os.getenv("ORT_INTRA_OP_THREADS", "0")):
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _tokenizer(path: Path, max_length: int) -> Tokenizer:
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    return tokenizer


def _feeds(session: ort.InferenceSession, encodings) -> dict[str, np.ndarray]:
    arrays = {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
    }
    return {i.name: arrays[i.name] for i in session.get_inputs()}


class OnnxSearcher:
    def __init__(self, bundle: Path, documents: list[dict[str, str]], embeddings: np.ndarray):
        self.session = _session(bundle / "embedder" / "model.onnx")
        self.tokenizer = _tokenizer(bundle / "embedder" / "tokenizer.json", EMBED_MAX_LENGTH)
        self.documents = documents
        self.embeddings = embeddings

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    def encode(self, queries: list[str]) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(queries)
        feeds = _feeds(self.session, encodings)
        token_states = self.session.run(None, feeds)[0]
        mask = feeds["attention_mask"][..., None].astype(np.float32)
        pooled = (token_states * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        pooled /= np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled.astype(np.float32)

    def search_vectors(self, vectors: np.ndarray, top_k: int) -> list[list[SearchResult]]:
        top_k = min(top_k, self.num_documents)
        scores = vectors @ self.embeddings.T
        batches = []
        for row in scores:
            top = np.argpartition(-row, top_k - 1)[:top_k]
            top = top[np.argsort(-row[top], kind="stable")]
            batches.append(
                [
                    SearchResult(
                        document_id=self.documents[i]["id"],
                        title=self.documents[i]["title"],
                        text=self.documents[i]["text"],
                        dense_score=float(row[i]),
                        dense_rank=rank,
                    )
                    for rank, i in enumerate(top, start=1)
                ]
            )
        return batches


class OnnxReranker:
    def __init__(self, bundle: Path, batch_size: int = 32):
        self.session = _session(bundle / "reranker" / "model.onnx")
        self.tokenizer = _tokenizer(bundle / "reranker" / "tokenizer.json", RERANK_MAX_LENGTH)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if not candidates:
            return []
        pairs = [(query, f"{c.title}. {c.text}" if c.title else c.text) for c in candidates]
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            encodings = self.tokenizer.encode_batch(pairs[start : start + self.batch_size])
            logits = self.session.run(None, _feeds(self.session, encodings))[0]
            scores.extend(float(s) for s in logits.reshape(len(encodings), -1)[:, 0])
        rescored = [replace(c, rerank_score=s) for c, s in zip(candidates, scores)]
        rescored.sort(key=lambda r: r.rerank_score, reverse=True)
        return rescored[:top_k]


class OnnxSearchEngine:
    def __init__(self, searcher: OnnxSearcher, reranker: OnnxReranker | None, manifest: dict):
        self.searcher = searcher
        self.reranker = reranker
        self.devices = {
            "embedder": f"cpu (onnx {manifest['embedder_variant']})",
            "reranker": f"cpu (onnx {manifest['reranker_variant']}, {RERANK_MAX_LENGTH} tokens)",
        }

    @classmethod
    def load(cls, bundle_dir: Path | None = None) -> "OnnxSearchEngine":
        bundle = Path(bundle_dir or os.getenv("NEURAL_SEARCH_BUNDLE") or DEFAULT_BUNDLE_DIR)
        manifest_path = bundle / "manifest.json"
        if not manifest_path.exists():
            raise BundleNotFoundError(f"{manifest_path} not found. Build it with: python -m scripts.export_onnx")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with gzip.open(bundle / "documents.json.gz", "rt", encoding="utf-8") as f:
            documents = json.load(f)
        embeddings = np.load(bundle / "embeddings.npy")
        if embeddings.shape != (len(documents), manifest["dimension"]):
            raise RuntimeError(f"Bundle is inconsistent: {embeddings.shape} vectors for {len(documents)} documents.")
        return cls(OnnxSearcher(bundle, documents, embeddings), OnnxReranker(bundle), manifest)

    def search(self, query: str, top_k: int = 10, rerank: bool = True, candidates: int = 20) -> SearchOutput:
        query = validate_query(query)
        top_k, candidates = validate_search_args(top_k, rerank, candidates, self.reranker is not None)

        start = time.perf_counter()
        vector = self.searcher.encode([query])
        encoded = time.perf_counter()
        retrieved = self.searcher.search_vectors(vector, candidates if rerank else top_k)[0]
        searched = time.perf_counter()

        timings = {"encode_ms": (encoded - start) * 1000, "retrieval_ms": (searched - encoded) * 1000}
        if rerank:
            results = self.reranker.rerank(query, retrieved, top_k)
            timings["rerank_ms"] = (time.perf_counter() - searched) * 1000
        else:
            results = retrieved
        timings["total_ms"] = (time.perf_counter() - start) * 1000
        return SearchOutput(results=results, reranked=rerank, timings_ms=timings)
