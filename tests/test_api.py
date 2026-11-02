from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.pipeline import SearchOutput
from app.search import SearchResult
from tests.conftest import requires_index


class StubEngine:
    """Returns canned results so API behaviour can be tested without loading models."""

    def __init__(self):
        self.searcher = SimpleNamespace(num_documents=3)
        self.reranker = object()
        self.devices = {"embedder": "cpu", "reranker": "cpu"}

    def search(self, query: str, top_k: int = 10, rerank: bool = True, candidates: int = 50) -> SearchOutput:
        results = [
            SearchResult(
                document_id=str(i),
                title=f"Doc {i}",
                text="body",
                dense_score=0.9 - 0.1 * i,
                dense_rank=i + 1,
                rerank_score=(5.0 - i) if rerank else None,
            )
            for i in range(top_k)
        ]
        timings = {"encode_ms": 1.0, "retrieval_ms": 0.5, "total_ms": 1.5}
        if rerank:
            timings["rerank_ms"] = 10.0
            timings["total_ms"] = 11.5
        return SearchOutput(results=results, reranked=rerank, timings_ms=timings)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(engine=StubEngine())) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["num_documents"] == 3
    assert body["devices"]["embedder"] == "cpu"


def test_search_response_shape(client):
    response = client.post("/search", json={"query": "vitamin D", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "vitamin D"
    assert body["reranked"] is True
    assert [r["rank"] for r in body["results"]] == [1, 2, 3]
    assert {"document_id", "title", "text", "score", "dense_score", "rerank_score"} <= body["results"][0].keys()
    assert body["latency_ms"] == pytest.approx(11.5)
    assert "rerank_ms" in body["timings_ms"]


def test_search_dense_only(client):
    body = client.post("/search", json={"query": "vitamin D", "top_k": 2, "rerank": False}).json()
    assert body["reranked"] is False
    assert all(r["rerank_score"] is None for r in body["results"])
    assert body["results"][0]["score"] == body["results"][0]["dense_score"]


def test_query_is_stripped(client):
    assert client.post("/search", json={"query": "  vitamin D  "}).json()["query"] == "vitamin D"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "    "},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 1000},
        {"query": "x", "top_k": "five"},
        {"query": "x", "top_k": 20, "candidates": 10},
        {"query": "x" * 1001},
    ],
)
def test_invalid_requests_return_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_pool_smaller_than_top_k_allowed_without_rerank(client):
    response = client.post("/search", json={"query": "x", "top_k": 20, "candidates": 10, "rerank": False})
    assert response.status_code == 200


def test_results_endpoint(client):
    response = client.get("/api/results")
    assert response.status_code == 200
    assert {"evaluation", "benchmark", "examples"} <= response.json().keys()


def test_demo_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Neural Search" in response.text


@requires_index
def test_search_end_to_end_with_real_models(engine):
    with TestClient(create_app(engine=engine)) as real_client:
        assert real_client.get("/health").json()["num_documents"] == 5183
        response = real_client.post("/search", json={"query": "Does vitamin D reduce fracture risk?", "top_k": 5})
        assert response.status_code == 200
        results = response.json()["results"]
        assert len(results) == 5
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert any("vitamin d" in r["title"].lower() for r in results)
