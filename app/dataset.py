from collections import defaultdict

from datasets import load_dataset

from app.config import DATASET_NAME, QRELS_DATASET_NAME


class DatasetLoadError(RuntimeError):
    pass


def _load(*args: str, **kwargs: str):
    try:
        return load_dataset(*args, **kwargs)
    except Exception as exc:
        raise DatasetLoadError(
            f"Failed to load {args} from Hugging Face. Check your network connection or HF cache."
        ) from exc


def load_corpus() -> list[dict[str, str]]:
    """Return SciFact documents with id, title, text, and the combined text that gets embedded."""
    dataset = _load(DATASET_NAME, "corpus", split="corpus")

    documents = []
    for row in dataset:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        documents.append(
            {
                "id": str(row["_id"]),
                "title": title,
                "text": text,
                "full_text": f"{title}. {text}" if title else text,
            }
        )

    if not documents:
        raise DatasetLoadError("Loaded corpus is empty.")
    ids = [doc["id"] for doc in documents]
    if len(set(ids)) != len(ids):
        raise DatasetLoadError("Corpus contains duplicate document IDs.")
    return documents


def load_queries() -> dict[str, str]:
    dataset = _load(DATASET_NAME, "queries", split="queries")
    return {str(row["_id"]): row["text"].strip() for row in dataset}


def load_qrels(split: str = "test") -> dict[str, dict[str, int]]:
    """Return {query_id: {document_id: relevance}} keeping only positive judgments."""
    dataset = _load(QRELS_DATASET_NAME, split=split)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in dataset:
        if int(row["score"]) > 0:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])
    if not qrels:
        raise DatasetLoadError(f"No positive relevance judgments found in {split} split.")
    return dict(qrels)


def load_eval_queries(split: str = "test") -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Return the judged queries for a split along with their qrels."""
    queries = load_queries()
    qrels = load_qrels(split)
    missing = [qid for qid in qrels if qid not in queries]
    if missing:
        raise DatasetLoadError(f"{len(missing)} judged queries are missing from the queries set.")
    return {qid: queries[qid] for qid in qrels}, qrels
