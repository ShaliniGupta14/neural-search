"""FastAPI service for the neural search engine.

Run (from the project root):
    uvicorn app.api:app --port 8000
Then open http://127.0.0.1:8000 for the demo UI or /docs for the OpenAPI explorer.

NEURAL_SEARCH_BACKEND selects the engine: "torch" (default; Sentence Transformers + FAISS) or
"onnx" (ONNX Runtime, used for the serverless deployment). This module imports neither at load
time so the ONNX deployment never pulls in torch.
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from app.core import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, InvalidQueryError, SearchOutput
from app.models import HealthResponse, SearchRequest, SearchResponse, SearchResultItem

logger = logging.getLogger("neural_search")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "app" / "static"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_FILES = {"evaluation": "evaluation.json", "benchmark": "benchmark.json", "examples": "examples.json"}


class Engine(Protocol):
    reranker: object | None
    devices: dict[str, str]

    def search(self, query: str, top_k: int, rerank: bool, candidates: int) -> SearchOutput: ...


def load_engine() -> Engine:
    backend = os.getenv("NEURAL_SEARCH_BACKEND", "torch")
    if backend == "onnx":
        from app.onnx_engine import OnnxSearchEngine

        return OnnxSearchEngine.load()
    if backend == "torch":
        from app.pipeline import SearchEngine

        return SearchEngine.load(
            embed_device=os.getenv("EMBED_DEVICE", "cpu"),
            rerank_device=os.getenv("RERANK_DEVICE", "auto"),
        )
    raise ValueError(f"Unknown NEURAL_SEARCH_BACKEND '{backend}'. Use torch or onnx.")


def create_app(engine: Engine | None = None) -> FastAPI:
    """Build the app. Pass an engine to skip model loading (used by tests)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Loading fails loudly (missing index, corrupt metadata, missing models) so the
        # server never starts in a half-working state.
        app.state.engine = engine if engine is not None else load_engine()
        logger.info("Search engine ready: %s", app.state.engine.devices)
        yield

    def get_engine(request: Request) -> Engine:
        # Serverless runtimes may skip lifespan events; load on first request in that case.
        if getattr(request.app.state, "engine", None) is None:
            request.app.state.engine = engine if engine is not None else load_engine()
        return request.app.state.engine

    app = FastAPI(
        title="Neural Search: SciFact",
        description="Dense retrieval (MiniLM) with cross-encoder reranking over BEIR SciFact.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(InvalidQueryError)
    async def invalid_query_handler(_: Request, exc: InvalidQueryError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        engine = get_engine(request)
        return HealthResponse(
            status="ok",
            num_documents=engine.searcher.num_documents,
            embedding_model=EMBEDDING_MODEL_NAME,
            reranker_model=RERANKER_MODEL_NAME if engine.reranker else None,
            devices=engine.devices,
        )

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        engine = get_engine(request)
        output = engine.search(body.query, top_k=body.top_k, rerank=body.rerank, candidates=body.candidates)
        return SearchResponse(
            query=body.query,
            reranked=output.reranked,
            results=[
                SearchResultItem(
                    rank=rank,
                    document_id=r.document_id,
                    title=r.title,
                    text=r.text,
                    score=round(r.score, 4),
                    dense_score=round(r.dense_score, 4),
                    dense_rank=r.dense_rank,
                    rerank_score=None if r.rerank_score is None else round(r.rerank_score, 4),
                )
                for rank, r in enumerate(output.results, start=1)
            ],
            latency_ms=round(output.timings_ms["total_ms"], 2),
            timings_ms={k: round(v, 2) for k, v in output.timings_ms.items()},
        )

    @app.get("/api/results")
    def results() -> dict:
        """Saved evaluation/benchmark output, or null for any that have not been generated yet."""
        payload = {}
        for key, filename in RESULT_FILES.items():
            path = RESULTS_DIR / filename
            if not path.exists():
                payload[key] = None
                continue
            try:
                payload[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=500, detail=f"{filename} is not valid JSON") from exc
        return payload

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
