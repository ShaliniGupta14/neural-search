"""Export the search bundle used by the ONNX backend (and the Vercel deployment).

Reads the document vectors back out of the FAISS index, so the ONNX backend searches exactly the
vectors the torch pipeline built, and downloads the published ONNX exports of both models.

Usage (from the project root, after build_index):
    python -m scripts.export_onnx [--embedder-variant fp32] [--reranker-variant fp32|quint8_avx2]
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import faiss
import numpy as np
from huggingface_hub import hf_hub_download

from app.config import DOCUMENTS_PATH, EMBEDDING_MODEL_NAME, INDEX_PATH, PROJECT_ROOT, RERANKER_MODEL_NAME
from app.onnx_engine import DEFAULT_BUNDLE_DIR

# fp32 reproduces the torch embeddings and rankings exactly. quint8_avx2 is a quarter of the size and
# faster on x86, but shifts query vectors slightly (cosine ~0.995 to fp32) and can reorder close results.
VARIANT_FILES = {"quint8_avx2": "onnx/model_quint8_avx2.onnx", "fp32": "onnx/model.onnx"}


def fetch_model(repo_id: str, variant: str, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download(repo_id, VARIANT_FILES[variant]), target / "model.onnx")
    shutil.copy(hf_hub_download(repo_id, "tokenizer.json"), target / "tokenizer.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the ONNX search bundle.")
    parser.add_argument("--embedder-variant", choices=sorted(VARIANT_FILES), default="fp32")
    parser.add_argument("--reranker-variant", choices=sorted(VARIANT_FILES), default="fp32")
    parser.add_argument("--out", type=Path, default=DEFAULT_BUNDLE_DIR)
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"{INDEX_PATH} not found. Run: python -m scripts.build_index")
    index = faiss.read_index(str(INDEX_PATH))
    embeddings = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    documents = [
        {"id": d["id"], "title": d["title"], "text": d["text"]}
        for d in json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    ]
    if len(documents) != len(embeddings):
        raise RuntimeError("Index and document metadata disagree; rebuild the index.")

    out: Path = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", embeddings)
    # Compressed: the serverless bundle has a hard size cap, and text shrinks about 3x.
    with gzip.open(out / "documents.json.gz", "wt", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False)
    fetch_model(EMBEDDING_MODEL_NAME, args.embedder_variant, out / "embedder")
    fetch_model(RERANKER_MODEL_NAME, args.reranker_variant, out / "reranker")
    manifest = {
        "embedder_variant": args.embedder_variant,
        "reranker_variant": args.reranker_variant,
        "dimension": int(embeddings.shape[1]),
        "num_documents": len(documents),
    }
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(
        f"Wrote ONNX bundle (embedder {args.embedder_variant}, reranker {args.reranker_variant}, "
        f"{total / 1e6:.1f} MB) to {out.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
