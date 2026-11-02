import json

import faiss
import numpy as np
import pytest

from app.metrics import evaluate_rankings, latency_summary, ndcg_at_k, recall_at_k, reciprocal_rank
from app.search import (
    DenseSearcher,
    IndexCorruptedError,
    IndexNotFoundError,
    InvalidQueryError,
    validate_query,
    validate_top_k,
)
from tests.conftest import requires_index

QUERY = "Does vitamin D reduce fracture risk?"


# ---------- input validation ----------

@pytest.mark.parametrize("query", ["", "   ", "\n\t", None, 42])
def test_invalid_queries_rejected(query):
    with pytest.raises(InvalidQueryError):
        validate_query(query)


def test_query_is_stripped():
    assert validate_query("  hello  ") == "hello"


@pytest.mark.parametrize("top_k", [0, -1, 101, True, 2.5, "5"])
def test_invalid_top_k_rejected(top_k):
    with pytest.raises(InvalidQueryError):
        validate_top_k(top_k, maximum=100)


# ---------- index loading errors (no model needed) ----------

def _write_index_files(tmp_path, num_vectors=3, dim=4, documents=None, info=None):
    index = faiss.IndexFlatIP(dim)
    index.add(np.eye(num_vectors, dim, dtype=np.float32))
    paths = {
        "index_path": tmp_path / "test.index",
        "documents_path": tmp_path / "documents.json",
        "info_path": tmp_path / "index_info.json",
    }
    faiss.write_index(index, str(paths["index_path"]))
    if documents is None:
        documents = [{"id": str(i), "title": f"t{i}", "text": f"x{i}"} for i in range(num_vectors)]
    paths["documents_path"].write_text(documents if isinstance(documents, str) else json.dumps(documents))
    info = info or {"embedding_model": "unused", "dimension": dim}
    paths["info_path"].write_text(json.dumps(info))
    return paths


def test_missing_index_raises(tmp_path):
    with pytest.raises(IndexNotFoundError, match="build_index"):
        DenseSearcher.load(
            index_path=tmp_path / "nope.index",
            documents_path=tmp_path / "nope.json",
            info_path=tmp_path / "nope_info.json",
        )


def test_corrupted_metadata_json_raises(tmp_path):
    paths = _write_index_files(tmp_path, documents="{not json")
    with pytest.raises(IndexCorruptedError, match="not valid JSON"):
        DenseSearcher.load(**paths)


def test_metadata_count_mismatch_raises(tmp_path):
    paths = _write_index_files(tmp_path, documents=[{"id": "0", "title": "", "text": ""}])
    with pytest.raises(IndexCorruptedError, match="Rebuild"):
        DenseSearcher.load(**paths)


def test_metadata_missing_fields_raises(tmp_path):
    paths = _write_index_files(tmp_path, documents=[{"id": str(i)} for i in range(3)])
    with pytest.raises(IndexCorruptedError, match="missing"):
        DenseSearcher.load(**paths)


def test_dimension_mismatch_raises(tmp_path):
    paths = _write_index_files(tmp_path, info={"embedding_model": "unused", "dimension": 999})
    with pytest.raises(IndexCorruptedError, match="dimension"):
        DenseSearcher.load(**paths)


def test_corrupted_index_file_raises(tmp_path):
    paths = _write_index_files(tmp_path)
    paths["index_path"].write_bytes(b"garbage")
    with pytest.raises(IndexCorruptedError, match="could not be read"):
        DenseSearcher.load(**paths)


# ---------- metrics ----------

def test_recall_and_reciprocal_rank():
    relevant = {"a": 1, "b": 1}
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 4) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert reciprocal_rank(ranked, relevant, k=1) == 0.0


def test_ndcg_perfect_and_worst():
    relevant = {"a": 1}
    assert ndcg_at_k(["a", "x"], relevant, 10) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "a"], relevant, 10) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k(["x", "y"], relevant, 10) == 0.0


def test_evaluate_rankings_counts_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    metrics = evaluate_rankings({"q1": ["a"]}, qrels)
    assert metrics["recall@1"] == 0.5
    assert metrics["mrr@10"] == 0.5


def test_latency_summary():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])
    assert summary["p50_ms"] == 3.0
    assert summary["mean_ms"] == 22.0
    assert summary["p95_ms"] > summary["p50_ms"]
    with pytest.raises(ValueError):
        latency_summary([])


# ---------- real index ----------

@requires_index
def test_index_loads_with_matching_metadata(engine):
    searcher = engine.searcher
    assert searcher.num_documents == len(searcher.documents) == 5183
    assert searcher.index.d == 384


@requires_index
def test_query_encoding_is_normalized_float32(engine):
    vectors = engine.searcher.encode([QUERY, "smoking and lung cancer"])
    assert vectors.shape == (2, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


@requires_index
@pytest.mark.parametrize("top_k", [1, 5, 10])
def test_dense_result_count_and_order(engine, top_k):
    results = engine.searcher.search(QUERY, top_k=top_k)
    assert len(results) == top_k
    scores = [r.dense_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.dense_rank for r in results] == list(range(1, top_k + 1))
    assert all(-1.0 <= s <= 1.0 + 1e-5 for s in scores)


@requires_index
def test_dense_search_is_deterministic(engine):
    first = [r.document_id for r in engine.searcher.search(QUERY, top_k=10)]
    second = [r.document_id for r in engine.searcher.search(QUERY, top_k=10)]
    assert first == second


@requires_index
def test_reranked_results_are_ordered_and_drawn_from_pool(engine):
    output = engine.search(QUERY, top_k=5, rerank=True, candidates=20)
    assert output.reranked
    assert len(output.results) == 5
    scores = [r.rerank_score for r in output.results]
    assert scores == sorted(scores, reverse=True)
    assert all(1 <= r.dense_rank <= 20 for r in output.results)
    assert {"encode_ms", "retrieval_ms", "rerank_ms", "total_ms"} <= output.timings_ms.keys()


@requires_index
def test_pipeline_rejects_pool_smaller_than_top_k(engine):
    with pytest.raises(InvalidQueryError, match="candidates"):
        engine.search(QUERY, top_k=10, rerank=True, candidates=5)


@requires_index
def test_pipeline_rejects_empty_query(engine):
    with pytest.raises(InvalidQueryError):
        engine.search("   ")
