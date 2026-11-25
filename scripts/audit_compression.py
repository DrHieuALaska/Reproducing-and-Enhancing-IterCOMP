"""Audit saved IterCOMP compression traces on CPU, without reader predictions."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import normalize, read_jsonl, sha256


def contains_phrase(normalized_text, normalized_phrase):
    return bool(normalized_phrase) and f" {normalized_phrase} " in f" {normalized_text} "


def assess_sample(row, gold):
    segments = {s["id"]: s for s in row["segment_catalog"]}
    paragraphs = {p["idx"]: p for p in gold["paragraphs"]}
    gold_paragraphs = {p["idx"] for p in paragraphs.values() if p["is_supporting"]}
    normalized_segments = {sid: normalize(s["text"]) for sid, s in segments.items()}
    hops = []
    for hop in gold["question_decomposition"]:
        pid = hop["paragraph_support_idx"]
        phrase = normalize(hop["answer"])
        body_hit = contains_phrase(normalize(paragraphs[pid]["paragraph_text"]), phrase)
        title_hit = contains_phrase(normalize(paragraphs[pid]["title"]), phrase)
        eligible_ids = {
            sid for sid, s in segments.items()
            if s["paragraph_idx"] == pid and contains_phrase(normalized_segments[sid], phrase)
        }
        hops.append((hop, eligible_ids, body_hit, title_hit))

    trace = row["iteration_trace"]
    first, final = [set(step["accumulated_evidence_ids"]) for step in (trace[0], trace[-1])]
    first_paragraphs, final_paragraphs = [{segments[s]["paragraph_idx"] for s in ids} for ids in (first, final)]
    eligible = [ids for _, ids, _, _ in hops if ids]
    retained_first = sum(bool(first & ids) for ids in eligible)
    retained_final = sum(bool(final & ids) for ids in eligible)
    prior_contexts, observed_states = {}, set()
    controller_prompt_repeats = response_changes = repeated_states = 0
    duplicate_prompt_seconds = 0.0
    for step in trace:
        context = step["controller_context"]
        reply = step["controller_calls"][-1]["raw_response"]
        if context in prior_contexts:
            controller_prompt_repeats += 1
            response_changes += prior_contexts[context] != reply
            duplicate_prompt_seconds += step["controller_seconds"]
        prior_contexts[context] = reply
        state = step["query"], tuple(step["accumulated_evidence_ids"])
        repeated_states += state in observed_states
        observed_states.add(state)
    no_new = [t["no_new_evidence"] for t in trace[1:]]
    first_stall = next((i for i, stalled in enumerate(no_new) if stalled), len(no_new))
    resumed_growth = any(not stalled for stalled in no_new[first_stall:])
    added = final - first
    initial_ranking = trace[0]["ranked_evidence"]
    dense_only = {
        score["id"] for score in sorted(initial_ranking, key=lambda s: -s["semantic"])[:len(first)]
    }
    terms = []
    for hop, ids, body_hit, title_hit in hops:
        if ids and not ids & final:
            best = min(
                (rank, step["iteration"], len(step["retrieved_evidence_ids"]))
                for step in trace for rank, score in enumerate(step["ranked_evidence"], 1)
                if score["id"] in ids
            )
            terms.append({
                "hop_question": hop["question"], "hop_answer": hop["answer"],
                "support_title": paragraphs[hop["paragraph_support_idx"]]["title"],
                "best_rank": best[0], "iteration": best[1], "cutoff_at_best_rank": best[2],
                "candidate_sentences": [segments[sid]["text"] for sid in sorted(ids)],
            })

    detail = {
        "id": row["id"], "hops": len(hops), "stop_reason": row["stop_reason"],
        "iterations": len(trace), "first_sentences": len(first), "final_sentences": len(final),
        "first_controller_context_tokens": trace[0]["controller_context_tokens"],
        "final_controller_context_tokens": trace[-1]["controller_context_tokens"],
        "first_complete_paragraph_coverage": gold_paragraphs <= first_paragraphs,
        "final_complete_paragraph_coverage": gold_paragraphs <= final_paragraphs,
        "eligible_annotated_hop_spans": len(eligible),
        "annotated_hops_with_body_span": sum(body for _, _, body, _ in hops),
        "annotated_hops_with_title_span_only": sum(title and not body for _, _, body, title in hops),
        "annotated_hops_with_body_span_but_no_segment_span": sum(body and not ids for _, ids, body, _ in hops),
        "first_retained_annotated_hop_spans": retained_first,
        "final_retained_annotated_hop_spans": retained_final,
        "first_dense_only_annotated_hop_spans_equal_count": sum(bool(dense_only & ids) for ids in eligible),
        "first_dense_only_selection_differs": dense_only != first,
        "all_hops_have_candidate_spans": len(eligible) == len(hops),
        "first_all_annotated_hop_spans_retained": retained_first == len(hops),
        "final_all_annotated_hop_spans_retained": retained_final == len(hops),
        "missing_span_from_already_retained_paragraph": sum(
            bool(ids) and not bool(ids & final) and hop["paragraph_support_idx"] in final_paragraphs
            for hop, ids, _, _ in hops),
        "followup_sentences_added": len(added),
        "followup_sentences_from_gold_paragraphs": sum(segments[s]["paragraph_idx"] in gold_paragraphs for s in added),
        "retrieval_followup_passes": len(trace) - 1,
        "followup_passes_no_new_evidence": sum(no_new),
        "growth_resumed_after_no_new_evidence": resumed_growth,
        "controller_prompt_repeats": controller_prompt_repeats,
        "controller_replies_changed_on_repeat": response_changes,
        "duplicate_prompt_generation_seconds": duplicate_prompt_seconds,
        "exact_repeated_query_and_selection": repeated_states,
        "candidate_sentences": len(segments),
        "initial_selected_above_rounded_top_decile": len(first) > (len(segments) + 9) // 10,
        "encoder_truncations": len(row["embedding"]["truncations"]),
    }
    example = {
        "id": row["id"], "question": row["question"], "answer": gold["answer"],
        "stop_reason": row["stop_reason"], "queries": [t["query"] for t in trace],
        "new_sentences_per_pass": [len(t["new_evidence_ids"]) for t in trace],
        "missing_annotated_hops": terms, "retained_context": row["compressed_context"],
    }
    return detail, example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Choose a fresh audit output directory")
    data = ROOT / "data" / "benchmark" / args.split
    gold = {r["id"]: r for r in read_jsonl(data / "gold.jsonl")}
    rows, failed = [], []
    seen = set()
    digest = hashlib.sha256()
    examples = defaultdict(list)

    with ZipFile(args.archive) as archive:
        candidates = []
        for name in archive.namelist():
            if name.endswith("/compression_results.jsonl"):
                base = name.rsplit("/", 1)[0]
                meta = json.loads(archive.read(f"{base}/run_metadata.json"))
                if (meta["identity"]["stage"] == "compress"
                        and meta["identity"]["method"] == "itercomp"
                        and set(meta["identity"]["selected_ids"]) == gold.keys()):
                    candidates.append((name, meta))
        if len(candidates) != 1:
            raise ValueError("Expected exactly one complete matching compression artifact")
        name, meta = candidates[0]
        if meta["identity"]["dataset_sha256"] != sha256(data / "inputs.jsonl"):
            raise ValueError("Input checksum differs from the saved run")
        for line in archive.open(name):
            digest.update(line)
            row = json.loads(line)
            rid = row["id"]
            if rid in seen or rid not in gold:
                raise ValueError(f"Unexpected or duplicate ID: {rid}")
            seen.add(rid)
            if row["error"]:
                failed.append({"id": rid, "error": row["error"]})
                continue
            if row["question"] != gold[rid]["question"]:
                raise ValueError(f"Question text differs: {rid}")
            detail, example = assess_sample(row, gold[rid])
            rows.append(detail)
            cohorts = (
                ("stalled_with_missing_spans", detail["controller_prompt_repeats"] >= 2 and
                 bool(example["missing_annotated_hops"])),
                ("all_gold_paragraphs_but_missing_candidate_span", detail["final_complete_paragraph_coverage"] and
                 detail["missing_span_from_already_retained_paragraph"] > 0),
                ("cap_with_all_annotated_spans", row["stop_reason"] == "max_iterations" and
                 detail["final_all_annotated_hop_spans_retained"]),
            )
            for label, include in cohorts:
                if include and len(examples[label]) < 8:
                    examples[label].append(example)

    if (seen != gold.keys() or digest.hexdigest() != meta["output_sha256"]
            or {r["id"] for r in failed} != set(meta["failed_sample_ids"])):
        raise ValueError("Saved rows, errors, or checksum differ from metadata")

    def summarize(records):
        totals = Counter()
        for detail in records:
            totals.update({k: v for k, v in detail.items() if isinstance(v, (int, float))})
        return {"n": len(records), "stop_reasons": dict(Counter(r["stop_reason"] for r in records)),
                "means": {k: fmean(r[k] for r in records) for k in (
                    "iterations", "first_sentences", "final_sentences", "first_controller_context_tokens",
                    "final_controller_context_tokens")}, "totals": dict(totals)}

    groups = {"all": rows}
    for hop in sorted({r["hops"] for r in rows}):
        groups[f"{hop}_hop"] = [r for r in rows if r["hops"] == hop]
    for stop in sorted({r["stop_reason"] for r in rows}):
        groups[stop] = [r for r in rows if r["stop_reason"] == stop]
    report = {
        "split": args.split, "archive_sha256": sha256(args.archive),
        "verified_compression_sha256": digest.hexdigest(),
        "gold_sha256": sha256(data / "gold.jsonl"),
        "compression_contract": meta["identity"]["compression_contract"],
        "component_initialization": meta["component_initialization"],
        "failed": failed, "groups": {key: summarize(group) for key, group in groups.items()},
        "examples": dict(examples),
        "notes": [
            "A hop is eligible if its normalized reference sub-answer occurs as a whole phrase in an individual sentence from its annotated supporting paragraph.",
            "Span coverage and gold paragraph coverage are diagnostics. They do not establish sufficiency, reasoning quality, or entailment, and may miss aliases or evidence elsewhere.",
            "Repeated controller contexts imply the same user prompt within a sample because the original question and template stay fixed.",
            "The dense-only replay uses the same initial sentence count and saved first-query scores. It cannot measure a changed compression loop.",
            "The first/final context lengths in this audit use the saved controller tokenizer, not the reader tokenizer.",
        ],
    }
    args.out.mkdir(parents=True)
    (args.out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.out / "per_question.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"split": args.split, "verified": len(seen), "failed": len(failed),
                      "overall": report["groups"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
