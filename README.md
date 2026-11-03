# Neural Search over SciFact

Two-stage neural retrieval for scientific claims: a bi-encoder (`all-MiniLM-L6-v2`) with an exact FAISS index finds candidates in milliseconds, then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks them for precision. Served with FastAPI, with a browser demo, and evaluated against the BEIR SciFact relevance judgments.

**Results on the 300 SciFact test queries** (Apple Silicon, measured with `scripts/evaluate.py` and `scripts/benchmark.py`):

| | MRR@10 | nDCG@10 | Recall@10 | p50 latency |
|---|---|---|---|---|
| Dense retrieval | 0.607 | 0.648 | 0.788 | 5.1 ms |
| Dense + cross-encoder rerank (pool 20) | **0.658** | **0.689** | **0.812** | 303 ms |
| Change | **+8.5%** | **+6.3%** | +3.0% | |

FAISS search over all 5,183 documents takes 0.26 ms p50; the full dense path, query embedding included, runs at 4.7 ms p50 (about 207 queries/s on one process).

**Live demo:** <https://neural-search-one.vercel.app> (search UI; API docs at [`/docs`](https://neural-search-one.vercel.app/docs))
**Results page:** <https://shalinigupta14.github.io/neural-search/> (charts, benchmark tables, and per-query before/after examples)

## Problem

Keyword search fails on scientific text because the same finding can be phrased many ways: *"vitamin D lowers fracture risk"* and *"cholecalciferol supplementation and bone fracture incidence"* share almost no terms. Dense embeddings match on meaning, but a single-vector bi-encoder compresses each document before it ever sees the query, which limits ranking precision. A cross-encoder reads the query and document together and ranks far better, but is too slow to run over a whole corpus.

This project combines them the standard way (retrieve with the fast model, rerank a small candidate pool with the accurate one) and measures exactly what that trade costs and buys.

## Architecture

```
                 offline (scripts/build_index.py)
 SciFact corpus ──► title + abstract ──► MiniLM bi-encoder ──► L2-normalize ──► FAISS IndexFlatIP
  (5,183 docs)                                (384-d, float32)                   + documents.json
                                                                                 + index_info.json

                 online (app/pipeline.py, served by app/api.py)
 query ──► MiniLM encode ──► FAISS top-N (cosine) ──► cross-encoder scores N pairs ──► top-K
           (~5 ms)          (<1 ms)                   (dominant cost)
```

| Module | Responsibility |
|---|---|
| `app/config.py` | Paths (relative to the project root), model names, device selection |
| `app/dataset.py` | Loads SciFact corpus, queries, and qrels from Hugging Face |
| `app/search.py` | `DenseSearcher`: loads and validates the index, encodes queries, FAISS search |
| `app/reranker.py` | `CrossEncoderReranker`: rescores candidates |
| `app/pipeline.py` | `SearchEngine`: dense retrieval + optional reranking with per-stage timings |
| `app/metrics.py` | Recall@K, MRR, nDCG, latency percentiles |
| `app/core.py` | Result types and validation shared by both backends (no torch import) |
| `app/onnx_engine.py` | `OnnxSearchEngine`: the same pipeline on ONNX Runtime, for serverless deployment |
| `app/models.py`, `app/api.py` | Pydantic schemas and the FastAPI app (+ demo UI in `app/static/`) |
| `scripts/` | `build_index`, `evaluate`, `benchmark`, `export_onnx`, `build_vercel`, `build_showcase`, `fix_macos_openmp` |

The API, the evaluation, and the benchmark all call the same `SearchEngine.search()`, so the reported metrics describe exactly what the service does.

## Setup

Tested on macOS (Apple Silicon) with Python 3.14. CPU works everywhere; Apple MPS and CUDA are used when available. No CUDA assumptions.

```bash
git clone https://github.com/ShaliniGupta14/neural-search.git && cd neural-search
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS only, once per environment (see "macOS note" below)
python -m scripts.fix_macos_openmp

python -m scripts.build_index        # ~30 s on M-series MPS, ~60 s on CPU
uvicorn app.api:app --port 8000      # open http://127.0.0.1:8000
```

Run every script as a module (`python -m scripts.<name>`) from the project root so `app` is importable.

### macOS note: two OpenMP runtimes

The macOS wheels of `torch` and `faiss-cpu` each bundle their own `libomp.dylib`. When both load into one process, CPU inference segfaults, FAISS search deadlocks, or OpenMP aborts with `OMP: Error #15`. The common workaround `KMP_DUPLICATE_LIB_OK=TRUE` is documented by Intel as unsafe. `scripts/fix_macos_openmp.py` instead replaces faiss's bundled copy with a symlink to torch's, so exactly one runtime is loaded (`--revert` restores it). `app.config.check_openmp_runtime()` fails fast with that instruction rather than letting the process crash.

## Dataset

[BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact): 5,183 paper abstracts, 1,109 claims, and expert relevance judgments (`BeIR/scifact-qrels`). Evaluation uses the 300-query **test** split. The corpus requires an explicit config:

```python
load_dataset("BeIR/scifact", "corpus", split="corpus")
```

## How it works

**Indexing.** Each document is embedded as `"{title}. {abstract}"` in batches with `normalize_embeddings=True` and stored as contiguous float32. The build asserts unit norms, because cosine similarity via `IndexFlatIP` is only correct on normalized vectors. Exact (flat) search is the right choice at this corpus size: 5,183 × 384 floats is 8 MB, and brute-force inner product takes well under a millisecond, so an approximate index would only cost recall. Three artifacts are written atomically (temp file + rename): the FAISS index, document metadata, and `index_info.json` recording the model and dimension. On load, the searcher checks that the index, metadata, and model all agree, and raises `IndexNotFoundError` / `IndexCorruptedError` with a rebuild instruction otherwise.

**Search.** The query is embedded with the same model and normalization, then FAISS returns the top-N by cosine similarity.

**Reranking.** The top-N candidates (default 50) are scored as `(query, title + abstract)` pairs by the cross-encoder, which attends across both texts jointly. Results are re-sorted by that score (a stable sort, so dense order breaks ties) and cut to top-K. Each result keeps its original dense rank so the effect of reranking stays visible.

**Device placement.** The embedder and reranker can run on different devices. Measured on Apple Silicon: encoding one short query is faster on CPU (kernel-launch overhead dominates on MPS), while scoring dozens of 512-token pairs is roughly 2.4× faster on MPS. The defaults (`EMBED_DEVICE=cpu`, `RERANK_DEVICE=auto`) follow those measurements and can be overridden by environment variables.

## API

Interactive OpenAPI docs: <http://127.0.0.1:8000/docs>. Demo UI: <http://127.0.0.1:8000/>.

```bash
curl -s localhost:8000/health
```
```json
{"status": "ok", "num_documents": 5183, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
 "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "devices": {"embedder": "cpu", "reranker": "mps:0"}}
```

```bash
curl -s localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "Does vitamin D reduce fracture risk?", "top_k": 3, "rerank": true, "candidates": 50}'
```

Real response (abstract text truncated here). The alendronate fracture trial was dense rank 13 and the cross-encoder moved it to 2:

```json
{
  "query": "Does vitamin D reduce fracture risk?",
  "reranked": true,
  "results": [
    {"rank": 1, "document_id": "9555784", "title": "Vitamin d deficiency in postmenopausal women - biological correlates.",
     "text": "INTRODUCTION Low vitamin D (VD) is associated with ...", "score": 3.0407,
     "dense_score": 0.6717, "dense_rank": 2, "rerank_score": 3.0407},
    {"rank": 2, "document_id": "11246427", "title": "Effect of alendronate on risk of fracture in women with low bone density ...",
     "text": "CONTEXT Alendronate sodium reduces fracture risk ...", "score": 2.3223,
     "dense_score": 0.5336, "dense_rank": 13, "rerank_score": 2.3223},
    {"rank": 3, "document_id": "30720103", "title": "Vitamin D status: measurement, interpretation, and clinical application.",
     "text": "Vitamin D, the sunshine vitamin, ...", "score": 1.4553,
     "dense_score": 0.6193, "dense_rank": 4, "rerank_score": 1.4553}
  ],
  "latency_ms": 341.58,
  "timings_ms": {"encode_ms": 20.73, "retrieval_ms": 1.12, "rerank_ms": 319.72, "total_ms": 341.58}
}
```

`score` is the cosine similarity for dense-only results and the cross-encoder logit when reranked; both raw scores are always returned.

| Field | Default | Constraint |
|---|---|---|
| `query` | required | 1–1000 chars, not blank |
| `top_k` | 5 | 1–100 |
| `rerank` | `true` | dense-only when `false` |
| `candidates` | 50 | 1–200, ≥ `top_k` when reranking |

Invalid input returns `422` with a message. A missing or corrupt index stops the server at startup with the rebuild command rather than serving errors.

`GET /api/results` returns the saved evaluation and benchmark JSON that the demo's results tab renders.

## Evaluation methodology

```bash
python -m scripts.evaluate      # writes results/evaluation.json and results/examples.json
python -m scripts.benchmark     # writes results/benchmark.json
python -m scripts.build_showcase  # renders docs/index.html from results/*.json
```

`docs/index.html` is a self-contained results page (charts, tables, per-query before/after examples). Enable GitHub Pages on the `docs/` folder to host it.

- **Queries:** all 300 SciFact test claims; relevance from `BeIR/scifact-qrels` (339 positive judgments; some claims have several relevant papers).
- **Recall@K:** fraction of a query's relevant documents in the top K, averaged over queries.
- **MRR@10:** reciprocal rank of the first relevant document within the top 10 (0 if none).
- **nDCG@10:** graded discounted gain over the top 10, normalized by the ideal ranking. This is BEIR's headline metric, so results are comparable to published numbers.
- **Latency:** every query is issued individually through `SearchEngine.search()` after 5 warm-up queries, so mean/p50/p95 are per-request latencies, not amortized batch throughput.
- **Candidate-pool recall:** dense Recall@N for the pool size N, the ceiling on what reranking that pool can achieve.

## Results

All numbers below come from `results/*.json` in this repo. Hardware: Apple Silicon (arm64), macOS, Python 3.14, torch 2.15. Embedder on CPU, reranker on MPS unless noted.

### Retrieval quality (300 test queries, top-10)

| Configuration | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | mean | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| Dense (bi-encoder) | 0.485 | 0.741 | 0.788 | 0.607 | 0.648 | 5.3 ms | 5.1 ms | 6.6 ms |
| Dense + rerank (pool 20) | **0.551** | 0.749 | **0.812** | **0.658** | **0.689** | 304 ms | 303 ms | 377 ms |
| Dense + rerank (pool 50) | 0.546 | 0.749 | 0.809 | 0.658 | 0.688 | 583 ms | 569 ms | 720 ms |
| Dense + rerank (pool 100) | 0.545 | **0.752** | 0.809 | 0.656 | 0.687 | 1032 ms | 1007 ms | 1257 ms |

Dense candidate-pool recall (the ceiling for reranking): R@20 = 0.844, R@50 = 0.889, R@100 = 0.925.

**Findings**
- Reranking improves MRR@10 by **8.5%** and Recall@1 by **13.7%** relative to dense retrieval. The largest gains are on claims phrased very differently from the paper's title; for *"The PRR MDA5 is a sensor of RNA virus infection"*, the relevant review moves from dense rank 20 to rank 1.
- **Bigger pools do not help.** Growing the pool from 20 to 100 lifts the relevant documents available from 84% to 93%, yet top-10 quality stays flat or dips slightly while latency triples. The extra candidates are mostly hard negatives the cross-encoder occasionally promotes. The API therefore defaults to a pool of 20.
- **Failure mode: negation.** The largest regression is *"The treatment of cancer patients with co-IR blockade does not cause any adverse autoimmune events"*, where dense retrieval ranked the relevant paper first and the cross-encoder pushed it out of the top 3 in favor of topically similar checkpoint-blockade papers. The MS MARCO cross-encoder scores topical relevance and is weak on negated claims. See `results/examples.json`.
- Dense nDCG@10 of 0.648 is in line with published BEIR figures for all-MiniLM-L6-v2 on SciFact, which serves as a sanity check on the evaluation code.

### Latency by stage (30 sampled queries, p50)

| Mode | Embed | Rerank | Top-K | Pool | Embed | FAISS | Rerank | Total p50 | Total p95 | QPS |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | cpu | – | 1 | – | 4.40 ms | 0.22 ms | – | 4.62 ms | 5.64 ms | 211 |
| dense | cpu | – | 10 | – | 4.41 ms | 0.26 ms | – | 4.67 ms | 5.83 ms | 207 |
| dense | cpu | – | 100 | – | 4.45 ms | 0.45 ms | – | 4.90 ms | 6.00 ms | 200 |
| dense | mps | – | 10 | – | 10.17 ms | 0.43 ms | – | 10.52 ms | 200.20 ms | 23 |
| rerank | cpu | cpu | 10 | 20 | 5.42 ms | 0.29 ms | 653 ms | 659 ms | 696 ms | 1.6 |
| rerank | cpu | cpu | 10 | 100 | 5.18 ms | 0.65 ms | 2478 ms | 2484 ms | 2801 ms | 0.4 |
| rerank | cpu | mps | 10 | 20 | 6.27 ms | 0.54 ms | 234 ms | 241 ms | 248 ms | 4.2 |
| rerank | cpu | mps | 10 | 50 | 6.85 ms | 0.75 ms | 545 ms | 553 ms | 807 ms | 1.8 |
| rerank | cpu | mps | 10 | 100 | 6.89 ms | 0.81 ms | 1015 ms | 1022 ms | 1149 ms | 1.0 |

Full table: `results/benchmark.json`.

- Exact FAISS search is never the bottleneck: under 1 ms even at top-100.
- The cross-encoder accounts for over 97% of reranked latency, and it scales linearly with pool size.
- MPS makes reranking **2.8× faster** than CPU (234 vs. 653 ms at pool 20), but makes single-query embedding **slower** (10.2 vs. 4.4 ms p50, with a 200 ms p95 tail from GPU dispatch). Hence the split device defaults.

## Deployment

The live demo is a **Vercel serverless Python function** on the free Hobby plan (1 vCPU, 2 GB). PyTorch alone is larger than Vercel's function size limit, so the deployment runs the same models through **ONNX Runtime** (`app/onnx_engine.py`) behind the same FastAPI app and UI. It uses no torch or FAISS: exact cosine search over the 5,183 vectors is a single NumPy matrix-vector product.

```bash
python -m scripts.export_onnx --reranker-variant quint8_avx2   # vectors from the FAISS index + ONNX models
python -m scripts.build_vercel                                  # self-contained project in deploy/vercel/
vercel deploy --prod --cwd deploy/vercel
```

**Parity.** The fp32 ONNX embedder reproduces the torch query vectors (cosine 1.0000) and dense rankings exactly; `tests/test_onnx.py` checks this. Retrieval in production is identical to the evaluated system.

**Fitting the reranker into one vCPU.** The fp32 cross-encoder over 20 full-length (512-token) abstracts took about 7.5 s per query on Vercel. Every option below was measured on all 300 test queries before choosing:

| Reranker | Max tokens | Pool | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| fp32 (published config) | 512 | 20 | 0.658 | 0.689 |
| fp32 | 256 | 20 | 0.654 | 0.685 |
| fp32 | 192 | 20 | 0.637 | 0.671 |
| fp32 | 128 | 20 | 0.598 | 0.637 |
| **int8 (`quint8_avx2`), deployed** | **256** | **20** | **0.658** | **0.686** |
| int8 | 256 | 10 | 0.650 | 0.680 |

Truncating to 128 tokens makes reranking *worse than no reranking* (dense MRR@10 is 0.607): the cross-encoder needs the part of the abstract that states the finding. The int8 model at 256 tokens matches the published MRR@10.

**Live latency** (Vercel Hobby, warm function, reranked, pool 20): about 5 ms query embedding, 1 ms retrieval, and 2.1 s reranking. With a pool of 10, reranking takes 1.1 s. Pinning ONNX Runtime to one thread (`ORT_INTRA_OP_THREADS=1`) cut reranking from 2.5 s to 2.1 s, because the function's CPU quota is smaller than the core count the runtime detects. The first request after the function has been idle adds a few seconds of cold start.

The `Dockerfile` packages the full torch + FAISS service for any container host (for example Google Cloud Run). The results page in `docs/` is served by GitHub Pages.

## Tests

```bash
pytest
```

Covers input validation (empty, whitespace-only, and non-string queries; out-of-range and non-integer `top_k`), every index-loading failure (missing files, invalid JSON, count and field mismatches, dimension mismatch, unreadable index), the metric implementations, real-index behavior (unit-norm float32 query vectors, result counts, score ordering, determinism, reranked results drawn from the candidate pool), and the API (health, response schema, 422 handling, demo page, and an end-to-end search against the real models), and ONNX-vs-torch parity of the deployed engine. Tests that need the built index or ONNX bundle skip cleanly if they are absent.

## Project structure

```
neural-search/
├── app/
│   ├── config.py  dataset.py  search.py  reranker.py  pipeline.py  metrics.py
│   ├── core.py  onnx_engine.py  models.py  api.py
│   └── static/index.html          # demo UI
├── scripts/
│   ├── build_index.py  evaluate.py  benchmark.py  build_showcase.py  fix_macos_openmp.py  common.py
│   ├── export_onnx.py  build_vercel.py
├── docs/                          # results showcase (index.html is generated; GitHub Pages ready)
├── results/                       # committed evaluation + benchmark output
├── data/                          # generated index and ONNX bundle (git-ignored)
├── tests/
│   ├── conftest.py  test_search.py  test_api.py  test_onnx.py
├── .github/workflows/ci.yml       # tests on every push
├── Dockerfile  requirements.txt  pyproject.toml  LICENSE
```

## Future improvements

- **Hybrid retrieval:** fuse BM25 with dense scores (reciprocal rank fusion) to recover exact-term matches such as gene names and drug codes that MiniLM misses.
- **Domain-adapted models:** swap in a biomedical bi-encoder or cross-encoder and compare on the same harness.
- **Cheaper reranking:** distill the cross-encoder, or rerank adaptively only when the top dense scores are close.
- **Scale:** move to `IndexHNSWFlat` or IVF-PQ for corpora in the millions and report the recall/latency curve against the exact baseline.
- **Serving:** request batching for the reranker and a result cache for repeated queries.
