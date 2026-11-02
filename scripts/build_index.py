"""Build a FAISS index over the SciFact corpus.

Usage (from the project root):
    python -m scripts.build_index [--device auto|cpu|mps] [--batch-size 64]
"""

import argparse
import json
import os
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config import (
    DATA_DIR,
    DATASET_NAME,
    DOCUMENTS_PATH,
    EMBEDDING_MODEL_NAME,
    INDEX_INFO_PATH,
    INDEX_PATH,
    PROJECT_ROOT,
    SEED,
    check_openmp_runtime,
    resolve_device,
)
from app.dataset import load_corpus


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_documents(
    model: SentenceTransformer, documents: list[dict[str, str]], batch_size: int
) -> np.ndarray:
    embeddings = model.encode(
        [doc["full_text"] for doc in documents],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("Embeddings are not unit-normalized; cosine via IndexFlatIP would be wrong.")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    # Inner product on unit vectors == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Write to a temp file in the same directory, then rename, so readers never see a partial file."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_fn(tmp_path)
        tmp_path.chmod(0o644)  # mkstemp creates files as 0600
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_json(path: Path, payload: Any) -> None:
    def _write(tmp: Path) -> None:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    atomic_write(path, _write)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SciFact FAISS index.")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda (default: auto)")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_openmp_runtime()
    set_seed(SEED)
    device = resolve_device(args.device)

    print(f"Loading {DATASET_NAME} corpus...")
    documents = load_corpus()
    print(f"Loaded {len(documents)} documents")

    print(f"Loading embedding model {EMBEDDING_MODEL_NAME} on device={device}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    start = time.perf_counter()
    embeddings = encode_documents(model, documents, args.batch_size)
    encode_seconds = time.perf_counter() - start
    print(f"Embeddings: shape={embeddings.shape} dtype={embeddings.dtype} ({encode_seconds:.1f}s)")

    index = build_faiss_index(embeddings)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write(INDEX_PATH, lambda tmp: faiss.write_index(index, str(tmp)))
    write_json(DOCUMENTS_PATH, documents)
    write_json(
        INDEX_INFO_PATH,
        {
            "dataset": DATASET_NAME,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "dimension": int(embeddings.shape[1]),
            "num_documents": len(documents),
            "index_type": "IndexFlatIP",
            "normalized": True,
            "device": device,
            "encode_seconds": round(encode_seconds, 2),
        },
    )

    print(f"Indexed {index.ntotal} documents")
    for label, path in (("FAISS index", INDEX_PATH), ("document metadata", DOCUMENTS_PATH), ("index info", INDEX_INFO_PATH)):
        print(f"Saved {label:<17} -> {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
