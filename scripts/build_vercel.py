"""Assemble a self-contained Vercel project in deploy/vercel/ (git-ignored).

The bundle contains only the torch-free modules, the ONNX models and vectors, and the saved
results, so the Python function stays under Vercel's 500 MB limit.

Usage (from the project root):
    python -m scripts.export_onnx --reranker-variant quint8_avx2
    python -m scripts.build_vercel
    vercel deploy --prod --cwd deploy/vercel
"""

import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "deploy" / "vercel"
BUNDLE = PROJECT_ROOT / "data" / "onnx"

APP_MODULES = ["__init__.py", "core.py", "models.py", "api.py", "onnx_engine.py"]

ENTRYPOINT = '''import os
from pathlib import Path

os.environ.setdefault("NEURAL_SEARCH_BACKEND", "onnx")
# One vCPU on the free plan: fp32 512-token reranking of 20 abstracts took ~7.5 s. int8 at 256 tokens
# measured the same MRR@10 on the SciFact test set; see README.
os.environ.setdefault("RERANK_MAX_LENGTH", "256")
# The function gets 1 vCPU; extra ONNX Runtime threads contend for it (rerank 2.5 s -> 2.1 s at 1 thread).
os.environ.setdefault("ORT_INTRA_OP_THREADS", "1")
os.environ.setdefault("NEURAL_SEARCH_BUNDLE", str(Path(__file__).resolve().parent / "bundle"))

from app.api import app  # noqa: E402,F401
'''

# Pinned to versions verified to resolve as manylinux wheels for CPython 3.12.
REQUIREMENTS = """fastapi==0.162.0
numpy==2.5.4
onnxruntime==1.31.1
tokenizers==0.23.3
"""


def main() -> None:
    if not (BUNDLE / "manifest.json").exists():
        raise FileNotFoundError(f"{BUNDLE} is missing. Run: python -m scripts.export_onnx")

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "app").mkdir(parents=True)
    for name in APP_MODULES:
        shutil.copy2(PROJECT_ROOT / "app" / name, OUT / "app" / name)
    shutil.copytree(PROJECT_ROOT / "app" / "static", OUT / "app" / "static")
    shutil.copytree(PROJECT_ROOT / "results", OUT / "results")
    shutil.copytree(BUNDLE, OUT / "bundle")

    (OUT / "index.py").write_text(ENTRYPOINT, encoding="utf-8")
    (OUT / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (OUT / ".python-version").write_text("3.12\n", encoding="utf-8")
    # Without an explicit preset, a project created from the CLI deploys as static files.
    (OUT / "vercel.json").write_text(
        json.dumps({"framework": "fastapi", "functions": {"index.py": {"memory": 2048, "maxDuration": 60}}}, indent=2),
        encoding="utf-8",
    )

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)} ({total / 1e6:.0f} MB before dependencies)")
    print("Deploy with: vercel deploy --prod --cwd deploy/vercel")


if __name__ == "__main__":
    main()
