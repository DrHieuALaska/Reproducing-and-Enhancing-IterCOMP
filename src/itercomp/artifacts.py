"""Compression artifact verification before any reader model is loaded."""
import json
from pathlib import Path
from .config import canonical_hash, compression_settings
from .dataset import read_jsonl
from .prompt_builder import render_paragraphs, render_segments
from .schemas import EvidenceSegment
from .serialization import file_hash


def source_hash():
    root = Path(__file__).parent
    return canonical_hash({str(p.relative_to(root)).replace("\\", "/"): file_hash(p)
                           for p in sorted(root.rglob("*.py"))})


def compression_contract(config, method, hashes):
    return {"settings": compression_settings(config, method, hashes),
            "algorithm_source_sha256": source_hash(), "schema_version": 1}


def seal_artifact(row, dataset_sha256, contract):
    row["dataset_sha256"] = dataset_sha256
    row["compression_contract_sha256"] = canonical_hash(contract)
    row["artifact_sha256"] = artifact_digest(row)
    return row


def artifact_digest(row):
    return canonical_hash({k: v for k, v in row.items() if k not in ("artifact_sha256", "run_identity_sha256")})


def validate_artifact_row(row, sample, config, dataset_sha256, contract):
    if row.get("artifact_sha256") != artifact_digest(row):
        raise ValueError(f"Compression artifact record checksum mismatch: {row.get('id')}")
    if row.get("dataset_sha256") != dataset_sha256 or row.get("compression_contract_sha256") != canonical_hash(contract):
        raise ValueError("Compression artifact dataset or settings mismatch")
    if row["id"] != sample.id or row["question"] != sample.question:
        raise ValueError("Compression artifact sample identity mismatch")
    if row["raw_context"] != render_paragraphs(sample.paragraphs, config.context):
        raise ValueError("Compression artifact raw context mismatch")
    paragraph_ids = {p.idx for p in sample.paragraphs}
    if not set(row["predicted_support_idxs"]) <= paragraph_ids:
        raise ValueError("Compression artifact contains unknown source paragraphs")
    if row["error"]:
        return
    if row["method"] == "itercomp":
        segments = [EvidenceSegment(**s) for s in row["segment_catalog"]]
        catalog = {s.id: s for s in segments}
        if len(catalog) != len(segments):
            raise ValueError("Duplicate catalog IDs in artifact")
        paragraphs = {p.idx: p for p in sample.paragraphs}
        for s in segments:
            p = paragraphs[s.paragraph_idx]
            if p.paragraph_text[s.start_char:s.end_char] != s.text or s.title != p.title:
                raise ValueError("Artifact sentence provenance mismatch")
        selected = row["selected_evidence_ids"]
        if len(set(selected)) != len(selected) or not set(selected) <= catalog.keys():
            raise ValueError("Invalid artifact evidence selection")
        ordered = [s for s in segments if s.id in set(selected)]
        if [s.id for s in ordered] != selected or row["compressed_context"] != render_segments(ordered, config.context):
            raise ValueError("Artifact final context/ordering mismatch")
        if {s.paragraph_idx for s in ordered} != set(row["predicted_support_idxs"]):
            raise ValueError("Artifact final supporting paragraphs mismatch")
        trace = row.get("iteration_trace", [])
        if not trace or len(trace) != row["iterations"] or trace[-1]["accumulated_evidence_ids"] != selected:
            raise ValueError("Artifact trace does not match final selection")
    else:
        retained = [p for p in sample.paragraphs if p.idx in row["predicted_support_idxs"]]
        if render_paragraphs(retained, config.context) != row["compressed_context"]:
            raise ValueError("Baseline artifact context mismatch")


def inspect_artifacts(path, samples, config, dataset_sha256, hashes):
    path = Path(path)
    meta_path = path.parent / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    identity = meta["identity"]
    if identity["stage"] != "compress" or identity["dataset_sha256"] != dataset_sha256:
        raise ValueError("Expected compression metadata for the same input dataset")
    method = identity["method"]
    contract = compression_contract(config, method, hashes)
    if identity["compression_contract"] != contract:
        raise ValueError("Compression settings changed (encoder/controller/prompt/decomposition/scoring/filtering/format/source). Recompress or select the matching config")
    if meta["status"] in ("running", "interrupted") or not meta.get("selected_subset_complete"):
        raise ValueError("Finish or resume compression before reading its artifacts")
    if meta.get("output_sha256") != file_hash(path):
        raise ValueError("Compression file hash differs from metadata; finish compression or repair it via --resume")
    sample_map = {s.id: s for s in samples}
    seen = set()
    for row in read_jsonl(path):
        if row["id"] in seen or row["id"] not in sample_map:
            raise ValueError("Duplicate or unknown compression artifact ID")
        if row.get("run_identity_sha256") != canonical_hash(identity):
            raise ValueError("Artifact run identity mismatch")
        seen.add(row["id"])
        validate_artifact_row(row, sample_map[row["id"]], config, dataset_sha256, contract)
    if seen != set(identity["selected_ids"]):
        raise ValueError("Compression artifact ID set differs from its manifest")
    return meta, seen
