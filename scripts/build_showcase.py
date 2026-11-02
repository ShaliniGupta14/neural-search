"""Render the static results showcase from results/*.json.

Writes docs/index.html (a standalone page, suitable for GitHub Pages). With --fragment PATH it
also writes the page body without the <html>/<head> wrapper.

Usage (from the project root, after evaluate and benchmark):
    python -m scripts.build_showcase
"""

import argparse
import json
from pathlib import Path

from app.config import PROJECT_ROOT, RESULTS_DIR

DOCS_DIR = PROJECT_ROOT / "docs"
TEMPLATE_PATH = DOCS_DIR / "showcase.template.html"

EVAL_KEYS = {
    "config", "reranked", "candidates", "recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10",
    "mean_ms", "p50_ms", "p95_ms", "mrr_gain_pct", "ndcg_gain_pct",
}
BENCH_KEYS = (
    "mode", "embed_device", "rerank_device", "top_k", "candidates",
    "encode_p50_ms", "retrieval_p50_ms", "rerank_p50_ms", "total_p50_ms", "total_p95_ms", "qps",
)
EXAMPLE_KEYS = ("rank", "dense_rank", "title", "relevant")


def load_result(name: str) -> dict:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts.evaluate and scripts.benchmark first.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_data() -> dict:
    evaluation = load_result("evaluation.json")
    benchmark = load_result("benchmark.json")
    examples = load_result("examples.json")
    return {
        "eval": {
            "num_queries": evaluation["num_queries"],
            "num_documents": evaluation["num_documents"],
            "pool_recall": evaluation["candidate_pool_recall"],
            "rows": [{k: v for k, v in row.items() if k in EVAL_KEYS} for row in evaluation["results"]],
        },
        "bench": [{k: row[k] for k in BENCH_KEYS} for row in benchmark["results"]],
        "examples": [
            {
                "query": ex["query"],
                "rr_delta": ex["rr_delta"],
                "dense": [{k: r[k] for k in EXAMPLE_KEYS} for r in ex["dense"]],
                "reranked": [{k: r[k] for k in EXAMPLE_KEYS} for r in ex["reranked"]],
            }
            for ex in examples["examples"]
        ],
        "env": {k: benchmark["environment"][k] for k in ("machine", "python", "torch")},
    }


def render_fragment() -> str:
    # "</" is escaped so document titles can never close the inline <script>.
    data = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE_PATH.read_text(encoding="utf-8").replace("__DATA__", data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the results showcase page.")
    parser.add_argument("--fragment", type=Path, help="Also write the page body without the HTML wrapper here.")
    args = parser.parse_args()

    fragment = render_fragment()
    page = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "</head>\n<body>\n" + fragment + "\n</body>\n</html>\n"
    )
    out = DOCS_DIR / "index.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out.relative_to(PROJECT_ROOT)}")
    if args.fragment:
        args.fragment.write_text(fragment, encoding="utf-8")
        print(f"Wrote {args.fragment}")


if __name__ == "__main__":
    main()
