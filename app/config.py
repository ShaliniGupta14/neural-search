import sys
from pathlib import Path

import torch

from app.core import EMBEDDING_MODEL_NAME, MAX_CANDIDATES, MAX_TOP_K, RERANKER_MODEL_NAME  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

INDEX_PATH = DATA_DIR / "scifact.index"
DOCUMENTS_PATH = DATA_DIR / "documents.json"
INDEX_INFO_PATH = DATA_DIR / "index_info.json"

DATASET_NAME = "BeIR/scifact"
QRELS_DATASET_NAME = "BeIR/scifact-qrels"

SEED = 42


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device. 'auto' prefers Apple MPS, then CUDA, then CPU."""
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device '{requested}'. Use auto, cpu, mps, or cuda.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available on this machine.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_openmp_runtime() -> None:
    """Fail fast instead of segfaulting when torch and faiss would load two libomp copies on macOS.

    See scripts/fix_macos_openmp.py for the background and the fix.
    """
    if sys.platform != "darwin":
        return
    import importlib.util

    spec = importlib.util.find_spec("faiss")
    if spec is None or spec.origin is None:
        return
    faiss_omp = Path(spec.origin).parent / ".dylibs" / "libomp.dylib"
    if faiss_omp.exists() and not faiss_omp.is_symlink():
        raise RuntimeError(
            "faiss-cpu and torch bundle separate OpenMP runtimes, which crashes on macOS. "
            "Run once per environment: python -m scripts.fix_macos_openmp"
        )
