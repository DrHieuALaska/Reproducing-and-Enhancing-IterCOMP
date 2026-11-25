"""Recompute paired reader calibration results from exported predictions on CPU."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
from statistics import fmean
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import answer_scores, normalize, read_jsonl, sha256

CONTEXTS = ("raw", "oracle", "itercomp")
CURRENT = "current"
REVISED = "no_abstention_instruction"
SENTINEL = "INSUFFICIENT_INFORMATION"
REMOVED = b" If the information is insufficient, return exactly INSUFFICIENT_INFORMATION."
REFUSAL_PHRASE = re.compile(
    r"\b(?:not (?:provided|mentioned|specified|stated|given|available|possible to determine)"
    r"|does not (?:provide|mention|specify|state|give|say|contain|include|indicate)"
    r"|cannot (?:be determined|determine|answer)|insufficient information)\b", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jsonl_bytes(payload):
    rows = [json.loads(line) for line in io.BytesIO(payload) if line.strip()]
    indexed = {row["id"]: row for row in rows}
    require(len(rows) == len(indexed), "Duplicate prediction or artifact IDs")
    return indexed


def answer_phrase(text, answers):
    normalized = f" {normalize(text)} "
    return any(f" {phrase} " in normalized for a in answers if (phrase := normalize(a)))


def measure(pred, gold, cap):
    answers = [gold["answer"], *gold["answer_aliases"]]
    em, f1 = answer_scores(pred["predicted_answer"], answers)
    support = {p["idx"] for p in gold["paragraphs"] if p["is_supporting"]}
    supports_retained = len(support & set(pred["predicted_support_idxs"]))
    return {
        "em": em, "f1": f1,
        "abstain": not pred["predicted_answer"].strip() and not pred["error"],
        "sentinel": (pred["reader_raw_response"] or "").strip() == SENTINEL,
        "refusal_phrase": bool(REFUSAL_PHRASE.search(pred["predicted_answer"])),
        "error": bool(pred["error"]), "output_cap_hit": pred["reader_output_tokens"] == cap,
        "complete_gold_paragraphs": supports_retained == len(support),
        "support_recall": supports_retained / len(support),
        "context_contains_answer": answer_phrase(pred["compressed_context"], answers),
        "non_em_contains_answer": not em and answer_phrase(pred["predicted_answer"], answers),
        "context_tokens": pred["compressed_context_tokens"],
        "input_tokens": pred["reader_input_tokens"], "output_tokens": pred["reader_output_tokens"],
        "reader_seconds": pred["reader_seconds"], "compression_seconds": pred["compression_seconds"],
    }


def summarize(rows):
    means = ("em", "f1", "support_recall", "context_tokens", "input_tokens", "output_tokens",
             "reader_seconds", "compression_seconds")
    counts = ("abstain", "sentinel", "refusal_phrase", "error", "output_cap_hit",
              "complete_gold_paragraphs", "context_contains_answer", "non_em_contains_answer")
    return {"n": len(rows), **{key: fmean(r[key] for r in rows) for key in means},
            **{key: sum(r[key] for r in rows) for key in counts}}


def paired(before, after, hops):
    # Draw matched questions within hop strata, preserving the 50/50/50 mixture.
    strata = defaultdict(list)
    for a, b, hop in zip(before, after, hops, strict=True):
        strata[hop].append((b["em"] - a["em"], b["f1"] - a["f1"]))
    rng = random.Random(42)
    draws = [[], []]
    n = len(hops)
    for _ in range(10_000):
        em = f1 = 0.0
        for pairs in strata.values():
            for a, b in rng.choices(pairs, k=len(pairs)):
                em += a
                f1 += b
        draws[0].append(100 * em / n)
        draws[1].append(100 * f1 / n)
    result = {}
    for i, key in enumerate(("em", "f1")):
        draws[i].sort()
        result[f"delta_{key}_points"] = 100 * fmean(b[key] - a[key] for a, b in zip(before, after))
        result[f"approximate_95pct_interval_{key}_points"] = [draws[i][250], draws[i][9749]]
    result["exact_match_transitions"] = dict(Counter(
        f'{int(a["em"])}_to_{int(b["em"])}' for a, b in zip(before, after)))
    formerly_abstained = [b for a, b in zip(before, after) if a["abstain"]]
    if formerly_abstained:
        result["formerly_abstained_after"] = summarize(formerly_abstained)
        result["exact_gains_from_abstention"] = sum(b["em"] for b in formerly_abstained)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--original-export", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT/"data/benchmark/calibration")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "Choose a fresh audit output directory")
    gold = {r["id"]: r for r in read_jsonl(args.data/"gold.jsonl")}
    inputs = {r["id"]: r for r in read_jsonl(args.data/"inputs.jsonl")}
    predictions, conditions, metrics, detail = {}, {}, {}, []
    with ZipFile(args.archive) as z, ZipFile(args.original_export) as original:
        manifest = json.loads(z.read("experiment_manifest.json"))
        ids = manifest["sample_ids"]
        require(len(ids) == len(set(ids)) == manifest["n"] and set(ids) == gold.keys() == inputs.keys(),
                "Mismatched sample IDs")
        hops = [len(gold[rid]["question_decomposition"]) for rid in ids]
        require(sha256(args.data/"gold.jsonl") == manifest["gold_sha256"], "Gold checksum mismatch")
        require(sha256(args.data/"inputs.jsonl") == manifest["inputs_sha256"], "Input checksum mismatch")
        prompts = {name: z.read(f"prompts/reader_{name}.txt") for name in (CURRENT, REVISED)}
        require(prompts[CURRENT].count(REMOVED) == 1 and prompts[CURRENT].replace(REMOVED, b"", 1) == prompts[REVISED],
                "Unexpected reader prompt difference")
        for name, payload in prompts.items():
            require(digest(payload) == manifest["prompt_sha256"][name], "Prompt checksum mismatch")

        compression_meta = json.loads(z.read("provenance/compression_metadata.json"))
        archived_meta = json.loads(z.read("provenance/archived_reader_metadata.json"))

        def matching_original(filename, metadata):
            candidates = [original.read(name) for name in original.namelist() if name.endswith("/" + filename)]
            matches = [data for data in candidates if digest(data) == metadata["output_sha256"]]
            require(len(matches) == 1, f"Could not identify original {filename}")
            return matches[0]

        sources = {c: z.read(f"contexts/{c}/compression_results.jsonl") for c in ("raw", "oracle")}
        sources["itercomp"] = matching_original("compression_results.jsonl", compression_meta)
        require(digest(sources["itercomp"]) == manifest["compression_sha256"], "Original compression checksum mismatch")
        saved = jsonl_bytes(matching_original("predictions.jsonl", archived_meta))
        require(saved.keys() == gold.keys() and archived_meta["identity"]["compression_file_sha256"] == manifest["compression_sha256"],
                "Archived control differs in IDs or compression")
        environments = []
        for context in CONTEXTS:
            artifacts = jsonl_bytes(sources[context])
            require(artifacts.keys() == gold.keys(), f"Missing context records: {context}")
            for name in (CURRENT, REVISED):
                condition = f"{context}_{name}"
                prefix = f"readers/{condition}"
                data = z.read(f"{prefix}/predictions.jsonl")
                meta = json.loads(z.read(f"{prefix}/run_metadata.json"))
                identity = meta["identity"]
                reader = meta["component_initialization"]["reader"]
                summary = json.loads(z.read(f"{prefix}/reports/scored/summary.json"))["overall"]
                require(meta["complete_split"] and meta["selected_subset_complete"]
                        and meta["completed_count"] == len(ids) and digest(data) == meta["output_sha256"],
                        f"Incomplete or changed predictions: {condition}")
                require(identity["selected_ids"] == ids and identity["method"] == context
                        and identity["dataset_sha256"] == manifest["inputs_sha256"]
                        and identity["reader"] == manifest["reader"]
                        and identity["reader_source_sha256"] == manifest["source_sha256"]
                        and identity["reader_prompt_sha256"] == manifest["prompt_sha256"][name]
                        and identity["compression_file_sha256"] == digest(sources[context])
                        and reader["resolved_revision"] == manifest["reader"]["revision"],
                        f"Run identity mismatch: {condition}")
                environments.append({
                    "environment": meta["details"]["environment"],
                    "quantization": reader["quantization_runtime"],
                    "gpu_names": [d["name"] for d in meta["gpu_after_loading"]["devices"]],
                })
                rows = jsonl_bytes(data)
                require(rows.keys() == gold.keys(), f"Unexpected prediction IDs: {condition}")
                measured = {}
                for rid, row in rows.items():
                    artifact = artifacts[rid]
                    require(all(row[k] == artifact[k] for k in
                                ("compressed_context", "raw_context", "predicted_support_idxs", "artifact_sha256"))
                            and row["question"] == inputs[rid]["question"],
                            f"Saved context changed in {condition}/{rid}")
                    measured[rid] = measure(row, gold[rid], manifest["reader"]["max_new_tokens"])
                    detail.append({"condition": condition, "id": rid, "hop": len(gold[rid]["question_decomposition"]),
                                   "question": row["question"], "gold_answer": gold[rid]["answer"],
                                   "prediction": row["predicted_answer"], **measured[rid]})
                overall = summarize(list(measured.values()))
                for metric in ("em", "f1"):
                    require(math.isclose(overall[metric], summary[f"answer_{metric}"], abs_tol=1e-12),
                            f"Score mismatch: {condition}/{metric}")
                predictions[condition], metrics[condition] = rows, measured
                conditions[condition] = {"overall": overall, "by_hop": {
                    hop: summarize([measured[rid] for rid, h in zip(ids, hops) if h == hop])
                    for hop in sorted(set(hops))}}
        require(all(env == environments[0] for env in environments), "Reader runtime identities differ")

    def metrics_for(context, prompt):
        return [metrics[f"{context}_{prompt}"][rid] for rid in ids]

    contrasts = {}
    for context in CONTEXTS:
        contrasts[f"prompt_removal_{context}"] = paired(
            metrics_for(context, CURRENT), metrics_for(context, REVISED), hops)
    for prompt in (CURRENT, REVISED):
        for before, after in (("raw", "itercomp"), ("itercomp", "oracle")):
            contrasts[f"{after}_minus_{before}_{prompt}"] = paired(
                metrics_for(before, prompt), metrics_for(after, prompt), hops)

    itercomp = metrics[f"itercomp_{REVISED}"]
    oracle = metrics[f"oracle_{REVISED}"]
    groups = {name: [rid for rid in ids if itercomp[rid]["complete_gold_paragraphs"] == full]
              for name, full in (("all_gold_paragraphs_represented", True), ("missing_gold_paragraphs", False))}
    coverage = {name: {
        "n": len(group), "itercomp": summarize([itercomp[rid] for rid in group]),
        "oracle": summarize([oracle[rid] for rid in group]),
        "oracle_exact_itercomp_non_exact": sum(oracle[rid]["em"] > itercomp[rid]["em"] for rid in group),
    } for name, group in groups.items() if group}

    control = predictions[f"itercomp_{CURRENT}"]
    replay = {key: sum(control[rid][key] != saved[rid][key] for rid in ids)
              for key in ("predicted_answer", "reader_raw_response")}
    stop_reasons = Counter(row["stop_reason"] for row in control.values())
    trace = [step for row in control.values() for step in row["iteration_trace"]]
    result = {
        "archive_sha256": sha256(args.archive), "original_archive_sha256": sha256(args.original_export),
        "recomputed_conditions": conditions, "paired_contrasts": contrasts,
        "reader": manifest["reader"], "runtime": environments[0],
        "archived_control_changed_count": replay,
        "itercomp_coverage_groups": coverage,
        "compression_trace": {"stop_reasons": dict(stop_reasons), "iterations": len(trace),
                              "repeat_query_flags": sum(t["repeated_query"] for t in trace),
                              "no_new_evidence_flags": sum(t["no_new_evidence"] for t in trace)},
        "notes": [
            "EM and F1 in condition summaries are fractions. Contrast deltas and intervals are points.",
            "Intervals are exploratory paired percentile bootstrap intervals stratified by hop. They do not account for related questions sharing source material, multiplicity, or domain shift.",
            "Paragraph presence, normalized answer phrase matches, and refusal phrase matches are diagnostic proxies. They do not establish sufficiency, correctness, or the true abstention rate.",
            "Reader seconds exclude model initialization and compression seconds.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out/"per_question.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    print(f"Verified and rescored {len(detail)} predictions. Audit: {args.out}")


if __name__ == "__main__":
    main()
