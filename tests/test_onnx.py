import numpy as np
import pytest

from app.onnx_engine import DEFAULT_BUNDLE_DIR, OnnxSearchEngine
from tests.conftest import requires_index

QUERIES = ["Does vitamin D reduce fracture risk?", "The PRR MDA5 is a sensor of RNA virus infection."]

requires_bundle = pytest.mark.skipif(
    not (DEFAULT_BUNDLE_DIR / "manifest.json").exists(),
    reason="ONNX bundle not built; run: python -m scripts.export_onnx",
)


@pytest.fixture(scope="module")
def onnx_engine() -> OnnxSearchEngine:
    return OnnxSearchEngine.load()


@requires_bundle
@requires_index
def test_onnx_dense_retrieval_matches_torch(onnx_engine, engine):
    if "fp32" not in onnx_engine.devices["embedder"]:
        pytest.skip("parity is only exact for the fp32 embedder")
    onnx_vectors = onnx_engine.searcher.encode(QUERIES)
    torch_vectors = engine.searcher.encode(QUERIES)
    np.testing.assert_allclose((onnx_vectors * torch_vectors).sum(axis=1), 1.0, atol=1e-4)
    for query in QUERIES:
        onnx_ids = [r.document_id for r in onnx_engine.search(query, top_k=10, rerank=False).results]
        torch_ids = [r.document_id for r in engine.search(query, top_k=10, rerank=False).results]
        assert onnx_ids == torch_ids


@requires_bundle
def test_onnx_rerank_ordering_and_pool(onnx_engine):
    output = onnx_engine.search(QUERIES[0], top_k=5, rerank=True, candidates=20)
    assert len(output.results) == 5
    scores = [r.rerank_score for r in output.results]
    assert scores == sorted(scores, reverse=True)
    assert all(1 <= r.dense_rank <= 20 for r in output.results)


@requires_bundle
def test_onnx_query_vectors_are_unit_norm(onnx_engine):
    vectors = onnx_engine.searcher.encode(QUERIES)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
