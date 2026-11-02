"""Benchmark per-stage latency across retrieval configurations.

Measures query embedding, FAISS retrieval, reranking, and end-to-end latency for:
  - dense-only search at several top_k values
  - query embedding on CPU vs. MPS
  - reranking with several candidate pool sizes on CPU vs. MPS

Usage (from the project root):
    python -m scripts.benchmark [--num-queries 50]
"""

import argparse

import numpy as np
from tqdm import tqdm

from app.config import SEED, resolve_device
from app.dataset import load_eval_queries
from app.metrics import latency_summary
from app.pipeline import SearchEngine
from app.reranker import CrossEncoderReranker
from app.search import DenseSearcher
from scripts.common import environment_info, markdown_table, save_json

WARMUP_QUERIES = 5


def measure(engine: SearchEngine, queries: list[str], top_k: int, rerank: bool, candidates: int) -> dict:
    for query in queries[:WARMUP_QUERIES]:
        engine.search(query, top_k=top_k, rerank=rerank, candidates=candidates)

    timings: dict[str, list[float]] = {}
    for query in queries:
        output = engine.search(query, top_k=top_k, rerank=rerank, candidates=candidates)
        for stage, value in output.timings_ms.items():
            timings.setdefault(stage, []).append(value)

    row: dict = {}
    for stage in ("encode_ms", "retrieval_ms", "rerank_ms"):
        row[f"{stage.removesuffix('_ms')}_p50_ms"] = (
            float(np.percentile(timings[stage], 50)) if stage in timings else None
        )
    total = latency_summary(timings["total_ms"])
    row.update({f"total_{k}": v for k, v in total.items()})
    row["qps"] = 1000.0 / total["mean_ms"]
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark search latency.")
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    parser.add_argument("--candidates", type=int, nargs="+", default=[20, 50, 100])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries_by_id, _ = load_eval_queries("test")
    rng = np.random.default_rng(SEED)
    all_queries = list(queries_by_id.values())
    queries = [all_queries[i] for i in rng.choice(len(all_queries), size=args.num_queries, replace=False)]
    print(f"Benchmarking on {len(queries)} SciFact test queries (seed {SEED})")

    devices = ["cpu"] + (["mps"] if resolve_device("auto") == "mps" else [])
    searchers = {d: DenseSearcher.load(device=d) for d in devices}
    rerankers = {d: CrossEncoderReranker(device=d) for d in devices}

    plan = []
    for k in args.top_k:
        plan.append(("dense", "cpu", None, k, False, 10))
    for d in devices[1:]:
        plan.append(("dense", d, None, 10, False, 10))
    for d in devices:
        for c in args.candidates:
            plan.append(("rerank", "cpu", d, 10, True, c))

    rows = []
    for mode, embed_device, rerank_device, top_k, rerank, candidates in tqdm(plan, desc="configs"):
        engine = SearchEngine(searchers[embed_device], rerankers[rerank_device] if rerank_device else None)
        row = {
            "mode": mode,
            "embed_device": embed_device,
            "rerank_device": rerank_device or "-",
            "top_k": top_k,
            "candidates": candidates if rerank else "-",
            **measure(engine, queries, top_k, rerank, candidates),
        }
        rows.append(row)

    save_json(
        "benchmark.json",
        {"num_queries": len(queries), "environment": environment_info(), "results": rows},
    )

    columns = [
        "mode", "embed_device", "rerank_device", "top_k", "candidates",
        "encode_p50_ms", "retrieval_p50_ms", "rerank_p50_ms", "total_mean_ms", "total_p50_ms", "total_p95_ms", "qps",
    ]
    printable = [{k: ("-" if v is None else v) for k, v in row.items()} for row in rows]
    print("\n" + markdown_table(printable, columns, float_format="{:.2f}"))
    print("\nSaved results/benchmark.json")


if __name__ == "__main__":
    main()
