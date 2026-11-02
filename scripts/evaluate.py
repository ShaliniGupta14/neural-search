"""Evaluate dense retrieval vs. dense + cross-encoder reranking on SciFact.

Every query goes through the same SearchEngine.search() call the API uses, one at a time,
so the reported latencies are per-request latencies.

Usage (from the project root):
    python -m scripts.evaluate [--split test] [--candidates 20 50 100] [--rerank-device auto]
"""

import argparse
import time

from tqdm import tqdm

from app.config import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME
from app.dataset import load_eval_queries
from app.metrics import evaluate_rankings, latency_summary, reciprocal_rank
from app.pipeline import SearchEngine, SearchOutput
from scripts.common import environment_info, markdown_table, save_json

TOP_K = 10
WARMUP_QUERIES = 5


def run_config(
    engine: SearchEngine, queries: dict[str, str], rerank: bool, candidates: int
) -> tuple[dict[str, SearchOutput], list[float]]:
    for query in list(queries.values())[:WARMUP_QUERIES]:
        engine.search(query, top_k=TOP_K, rerank=rerank, candidates=candidates)

    outputs: dict[str, SearchOutput] = {}
    latencies = []
    label = f"rerank@{candidates}" if rerank else "dense"
    for query_id, query in tqdm(queries.items(), desc=label, leave=False):
        output = engine.search(query, top_k=TOP_K, rerank=rerank, candidates=candidates)
        outputs[query_id] = output
        latencies.append(output.timings_ms["total_ms"])
    return outputs, latencies


def candidate_recall(engine: SearchEngine, queries: dict[str, str], qrels, pool_sizes: list[int]) -> dict[int, float]:
    """Recall of the dense candidate pool: the ceiling on what reranking that pool can reach."""
    query_ids = list(queries)
    vectors = engine.searcher.encode([queries[qid] for qid in query_ids])
    pools = engine.searcher.search_vectors(vectors, max(pool_sizes))
    rankings = {qid: [r.document_id for r in pool] for qid, pool in zip(query_ids, pools)}
    return {
        size: evaluate_rankings(rankings, qrels, recall_ks=(size,))[f"recall@{size}"] for size in pool_sizes
    }


def pick_examples(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    dense: dict[str, SearchOutput],
    reranked: dict[str, SearchOutput],
    n_improved: int = 5,
    n_regressed: int = 1,
) -> list[dict]:
    """Queries where reranking moved the first relevant document the most, in both directions."""
    deltas = []
    for qid in queries:
        dense_rr = reciprocal_rank([r.document_id for r in dense[qid].results], qrels[qid])
        rerank_rr = reciprocal_rank([r.document_id for r in reranked[qid].results], qrels[qid])
        deltas.append((rerank_rr - dense_rr, qid))
    deltas.sort(key=lambda d: (d[0], d[1]), reverse=True)
    chosen = [d for d in deltas[:n_improved] if d[0] > 0] + [d for d in deltas[::-1][:n_regressed] if d[0] < 0]

    def serialize(output: SearchOutput, relevant: dict[str, int]) -> list[dict]:
        return [
            {
                "rank": rank,
                "document_id": r.document_id,
                "title": r.title,
                "dense_rank": r.dense_rank,
                "dense_score": round(r.dense_score, 4),
                "rerank_score": None if r.rerank_score is None else round(r.rerank_score, 4),
                "relevant": r.document_id in relevant,
            }
            for rank, r in enumerate(output.results[:5], start=1)
        ]

    return [
        {
            "query_id": qid,
            "query": queries[qid],
            "rr_delta": round(delta, 4),
            "relevant_ids": sorted(qrels[qid]),
            "dense": serialize(dense[qid], qrels[qid]),
            "reranked": serialize(reranked[qid], qrels[qid]),
        }
        for delta, qid in chosen
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality and latency on SciFact.")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--candidates", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--embed-device", default="cpu")
    parser.add_argument("--rerank-device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N queries (for smoke tests).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    queries, qrels = load_eval_queries(args.split)
    if args.limit:
        queries = dict(sorted(queries.items(), key=lambda kv: int(kv[0]))[: args.limit])
        qrels = {qid: qrels[qid] for qid in queries}
    print(f"Evaluating {len(queries)} {args.split} queries ({sum(len(v) for v in qrels.values())} relevant judgments)")

    engine = SearchEngine.load(embed_device=args.embed_device, rerank_device=args.rerank_device)
    print(f"Devices: {engine.devices}")

    configs = [("Dense (bi-encoder)", False, TOP_K)] + [
        (f"Dense + rerank (pool {c})", True, c) for c in args.candidates
    ]
    rows = []
    outputs_by_config: dict[str, dict[str, SearchOutput]] = {}
    started = time.perf_counter()
    for name, rerank, candidates in configs:
        outputs, latencies = run_config(engine, queries, rerank, candidates)
        outputs_by_config[name] = outputs
        rankings = {qid: [r.document_id for r in out.results] for qid, out in outputs.items()}
        stage_means = {
            stage: sum(o.timings_ms.get(stage, 0.0) for o in outputs.values()) / len(outputs)
            for stage in ("encode_ms", "retrieval_ms", "rerank_ms")
        }
        row = {
            "config": name,
            "reranked": rerank,
            "candidates": candidates if rerank else None,
            **evaluate_rankings(rankings, qrels),
            **latency_summary(latencies),
            **{f"mean_{k}": v for k, v in stage_means.items()},
        }
        rows.append(row)
        print(f"  {name:<26} MRR@10={row['mrr@10']:.4f}  nDCG@10={row['ndcg@10']:.4f}  p50={row['p50_ms']:.1f}ms")

    pool_recall = candidate_recall(engine, queries, qrels, sorted({10, *args.candidates}))

    dense_row = rows[0]
    for row in rows[1:]:
        row["mrr_gain_pct"] = 100 * (row["mrr@10"] - dense_row["mrr@10"]) / dense_row["mrr@10"]
        row["ndcg_gain_pct"] = 100 * (row["ndcg@10"] - dense_row["ndcg@10"]) / dense_row["ndcg@10"]

    best_rerank = max(rows[1:], key=lambda r: r["mrr@10"])
    examples = pick_examples(
        queries, qrels, outputs_by_config[rows[0]["config"]], outputs_by_config[best_rerank["config"]]
    )

    payload = {
        "dataset": f"BeIR/scifact ({args.split})",
        "num_queries": len(queries),
        "num_documents": engine.searcher.num_documents,
        "top_k": TOP_K,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "devices": engine.devices,
        "environment": environment_info(),
        "wall_time_s": round(time.perf_counter() - started, 1),
        "results": rows,
        "candidate_pool_recall": {str(k): v for k, v in pool_recall.items()},
    }
    if not args.limit:
        save_json("evaluation.json", payload)
        save_json("examples.json", {"compared_config": best_rerank["config"], "examples": examples})

    columns = ["config", "recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10", "mean_ms", "p50_ms", "p95_ms"]
    table = markdown_table(rows, columns)
    print("\n" + table)
    print("\nDense candidate-pool recall (upper bound for reranking): "
          + ", ".join(f"R@{k}={v:.3f}" for k, v in pool_recall.items()))
    for row in rows[1:]:
        print(f"{row['config']}: MRR@10 {row['mrr_gain_pct']:+.1f}%, nDCG@10 {row['ndcg_gain_pct']:+.1f}% vs dense")
    if not args.limit:
        print("\nSaved results/evaluation.json and results/examples.json")


if __name__ == "__main__":
    main()
