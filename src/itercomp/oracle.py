"""The only inference code path allowed to open gold annotations."""
from .dataset import read_jsonl, input_sample


def oracle_supports(gold_path, samples):
    expected = {s.id: s for s in samples}
    result = {}
    seen = set()
    for row in read_jsonl(gold_path):
        rid = row["id"]
        if rid in seen:
            raise ValueError(f"Duplicate Oracle gold ID: {rid}")
        seen.add(rid)
        if rid not in expected:
            continue
        # Verify identity, then expose only supporting paragraph IDs to the baseline.
        stripped = {"id": rid, "question": row["question"], "paragraphs": [
            {k: p[k] for k in ("idx", "title", "paragraph_text")} for p in row["paragraphs"]]}
        if input_sample(stripped) != expected[rid]:
            raise ValueError("Oracle gold and input content differ")
        if any(not isinstance(p["is_supporting"], bool) for p in row["paragraphs"]):
            raise ValueError("Invalid Oracle support labels")
        selected = {p["idx"] for p in row["paragraphs"] if p["is_supporting"]}
        if not selected:
            raise ValueError("Oracle sample has no supporting paragraphs")
        result[rid] = selected
    if set(result) != set(expected):
        raise ValueError("Oracle gold is missing input IDs")
    return result
