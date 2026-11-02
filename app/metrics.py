import math
from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked_ids: Sequence[str], relevant: dict[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant: dict[str, int], k: int | None = None) -> float:
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant: dict[str, int], k: int) -> float:
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(rank + 1) for rank, doc_id in enumerate(ranked_ids[:k], start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    recall_ks: Sequence[int] = (1, 5, 10),
    mrr_k: int = 10,
    ndcg_k: int = 10,
) -> dict[str, float]:
    """Average IR metrics over every judged query; a query with no ranking counts as zero."""
    totals: dict[str, float] = {f"recall@{k}": 0.0 for k in recall_ks}
    totals[f"mrr@{mrr_k}"] = 0.0
    totals[f"ndcg@{ndcg_k}"] = 0.0

    for query_id, relevant in qrels.items():
        ranked = rankings.get(query_id, [])
        for k in recall_ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
        totals[f"mrr@{mrr_k}"] += reciprocal_rank(ranked, relevant, mrr_k)
        totals[f"ndcg@{ndcg_k}"] += ndcg_at_k(ranked, relevant, ndcg_k)

    return {name: value / len(qrels) for name, value in totals.items()}


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("No latency samples.")
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }
