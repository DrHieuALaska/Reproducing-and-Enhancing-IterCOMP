"""Strict input-only loader. Oracle access lives in a separate module."""
import json
from pathlib import Path
from .schemas import InputSample, Paragraph


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                yield row


def input_sample(row):
    if set(row) != {"id", "question", "paragraphs"}:
        raise ValueError("Inference requires input-only records: id, question, paragraphs; gold fields are forbidden")
    if not isinstance(row["id"], str) or not row["id"] or not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError("Invalid sample ID or question")
    paragraphs = []
    for p in row["paragraphs"]:
        if set(p) != {"idx", "title", "paragraph_text"}:
            raise ValueError("Inference paragraph contains unexpected fields (possibly gold labels)")
        if isinstance(p["idx"], bool) or not isinstance(p["idx"], int) or p["idx"] < 0:
            raise ValueError("Paragraph idx must be a nonnegative integer")
        if not all(isinstance(p[k], str) for k in ("title", "paragraph_text")):
            raise ValueError("Paragraph title and body must be strings")
        paragraphs.append(Paragraph(**p))
    if len({p.idx for p in paragraphs}) != len(paragraphs):
        raise ValueError("Duplicate paragraph IDs")
    return InputSample(row["id"], row["question"], tuple(paragraphs))


def load_inputs(path):
    result = [input_sample(row) for row in read_jsonl(path)]
    if not result or len({s.id for s in result}) != len(result):
        raise ValueError("Input dataset must be nonempty with unique sample IDs")
    return result


def select_samples(samples, limit=None, sample_ids=None):
    if sample_ids is not None:
        requested = set(sample_ids)
        if not requested or len(requested) != len(sample_ids):
            raise ValueError("Subset IDs must be nonempty and unique")
        missing = requested - {s.id for s in samples}
        if missing:
            raise ValueError(f"Unknown subset sample IDs: {sorted(missing)}")
        samples = [s for s in samples if s.id in requested]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    return samples
