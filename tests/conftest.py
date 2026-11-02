import pytest

from app.config import DOCUMENTS_PATH, INDEX_INFO_PATH, INDEX_PATH
from app.pipeline import SearchEngine

requires_index = pytest.mark.skipif(
    not (INDEX_PATH.exists() and DOCUMENTS_PATH.exists() and INDEX_INFO_PATH.exists()),
    reason="Index not built; run: python -m scripts.build_index",
)


@pytest.fixture(scope="session")
def engine() -> SearchEngine:
    # CPU for both models keeps tests deterministic and independent of MPS availability.
    return SearchEngine.load(embed_device="cpu", rerank_device="cpu")
