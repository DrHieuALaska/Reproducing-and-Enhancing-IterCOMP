"""Separate CPU-only scoring; explicitly materialize matching gold for partial runs."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import read_jsonl, write_jsonl, evaluate, sha256, dump
from diagnostics import diagnose


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh evaluation output directory; existing results are preserved")
    predictions = list(read_jsonl(args.predictions))
    ids = {p["id"] for p in predictions}
    if not ids or len(ids) != len(predictions):
        raise ValueError("Empty or duplicate predictions")
    gold = list(read_jsonl(args.gold))
    gold_ids = {g["id"] for g in gold}
    if len(gold_ids) != len(gold) or not ids <= gold_ids:
        raise ValueError("Unknown predictions or duplicate gold IDs")
    partial = ids != gold_ids
    if partial and not args.allow_partial:
        raise ValueError("Partial predictions: explicitly pass --allow-partial to create a matching gold subset")
    out.mkdir(parents=True, exist_ok=True)
    gold_path = args.gold
    if partial:
        gold_path = out / "gold_subset.jsonl"
        write_jsonl(gold_path, [g for g in gold if g["id"] in ids])
    dump(out / "evaluation_metadata.json", {"complete_split": not partial, "n_predictions": len(ids),
                                           "n_full_gold": len(gold_ids), "source_gold_sha256": sha256(args.gold),
                                           "predictions_sha256": sha256(args.predictions)})
    evaluate(gold_path, args.predictions, out / "scored")
    diagnose(gold_path, args.predictions, out / "diagnostics")


if __name__ == "__main__":
    main()
