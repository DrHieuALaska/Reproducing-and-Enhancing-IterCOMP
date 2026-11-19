"""Resumable staged runner. All expensive initialization occurs after validation."""
from dataclasses import asdict
import json
from pathlib import Path
from .artifacts import compression_contract, seal_artifact, inspect_artifacts, source_hash
from .backends import create_backend
from .config import canonical_hash
from .dataset import load_inputs, read_jsonl, select_samples
from .embedding import create_encoder
from .pipeline import CompressionPipeline, ReaderPipeline
from .prompt_builder import load_prompts, prompt_hashes
from .serialization import RunStore, file_hash
from .telemetry import environment_info, release_memory, gpu_memory


def _resolved_config(store, config):
    import yaml
    path = store.directory / "resolved_config.yaml"
    if not path.exists():
        path.write_text(yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _details(config, inputs, hashes):
    return {"inputs_path": str(Path(inputs).resolve()), "config": asdict(config),
            "prompt_hashes": hashes, "environment": environment_info(),
            "reproduction_assumptions": ["Mistral controller and instruction-tuned Llama reader variants are choices",
                                        "spaCy blank-language sentencizer; body-only scoring/rendering by default",
                                        "linear percentiles; source ordering; initial pass counts as iteration one",
                                        "No warmup; no caches across samples; exact reader abstention sentinel"]}


def _record_initialization(store, current):
    previous = store.metadata.get("component_initialization")
    if previous:
        for role, info in current.items():
            old_revision = previous.get(role, {}).get("resolved_revision")
            new_revision = info.get("resolved_revision")
            if old_revision is not None and new_revision is not None and old_revision != new_revision:
                raise ValueError(f"{role} resolved revision changed during resume; pin the original model revision")
    else:
        store.metadata["component_initialization"] = current
    store.metadata["sessions"][-1]["component_initialization"] = current


def run_compression(config, inputs_path, output_dir, method="itercomp", gold_path=None,
                    limit=None, sample_ids=None, resume=False):
    config.validate()
    if method != "oracle" and gold_path is not None:
        raise ValueError("Gold paths are forbidden outside the explicit Oracle method")
    all_samples = load_inputs(inputs_path)
    samples = select_samples(all_samples, limit, sample_ids)
    prompts = load_prompts(config)
    hashes = prompt_hashes(prompts)
    dataset_hash = file_hash(inputs_path)
    contract = compression_contract(config, method, hashes)
    identity = {"stage": "compress", "method": method, "dataset_sha256": dataset_hash,
                "dataset_count": len(all_samples), "selected_ids": [s.id for s in samples],
                "compression_contract": contract}
    supports = None
    if method == "oracle":
        if gold_path is None:
            raise ValueError("Oracle requires --oracle-gold explicitly")
        from .oracle import oracle_supports
        supports = oracle_supports(gold_path, samples)
        identity["oracle_gold_sha256"] = file_hash(gold_path)
    backend = encoder = None
    with RunStore(output_dir, "compression_results.jsonl", identity,
                  _details(config, inputs_path, hashes), resume) as store:
        _resolved_config(store, config)
        pending = [s for s in samples if s.id not in store.ids]
        if not pending:
            return store.path
        try:
            if method == "itercomp":
                encoder = create_encoder(config.embedding)
                backend = create_backend(config.controller)
                backend.preflight()
                encoder.prepare()
                backend.prepare()
                _record_initialization(store, {"encoder": encoder.initialization, "controller": backend.initialization})
            pipeline = CompressionPipeline(config, prompts, encoder, backend)
            if method == "itercomp":
                pipeline.decomposer.prepare()
            store.metadata["gpu_after_loading"] = gpu_memory()
            store.save_metadata()
            for sample in pending:
                row = pipeline.compress(sample, method, supports[sample.id] if supports else None)
                seal_artifact(row, dataset_hash, contract)
                store.append(row)
                print(f"compress {len(store.ids)}/{len(samples)} {sample.id}: {row['stop_reason']}", flush=True)
        finally:
            if backend:
                backend.close()
            if encoder:
                encoder.close()
            backend = encoder = None
            store.metadata["gpu_after_release"] = release_memory()
    return Path(output_dir) / "compression_results.jsonl"


def run_reader(config, inputs_path, compression_path, output_dir, limit=None, sample_ids=None, resume=False):
    config.validate()
    all_samples = load_inputs(inputs_path)
    prompts = load_prompts(config)
    hashes = prompt_hashes(prompts)
    dataset_hash = file_hash(inputs_path)
    compression_meta, available_ids = inspect_artifacts(compression_path, all_samples, config, dataset_hash, hashes)
    samples = select_samples([s for s in all_samples if s.id in available_ids], limit, sample_ids)
    selected_ids = [s.id for s in samples]
    identity = {"stage": "read", "method": compression_meta["identity"]["method"],
                "dataset_sha256": dataset_hash, "dataset_count": len(all_samples),
                "selected_ids": selected_ids, "compression_file_sha256": file_hash(compression_path),
                "compression_contract": compression_meta["identity"]["compression_contract"],
                "reader": asdict(config.reader), "reader_prompt_sha256": hashes["reader"],
                "reference_tokenizer": asdict(config.reference_tokenizer),
                "reader_source_sha256": source_hash()}
    backend = reference = None
    with RunStore(output_dir, "predictions.jsonl", identity,
                  _details(config, inputs_path, hashes), resume) as store:
        _resolved_config(store, config)
        pending = set(selected_ids) - store.ids
        if not pending:
            return store.path
        try:
            # Reader stage never constructs a BGE encoder or controller backend.
            backend = create_backend(config.reader)
            backend.preflight()
            backend.prepare()
            if config.reference_tokenizer.model_id:
                from .tokenization import ReferenceTokenizer
                reference = ReferenceTokenizer(config.reference_tokenizer)
                reference.prepare()
            _record_initialization(store, {"reader": backend.initialization})
            store.metadata["compression_initialization"] = compression_meta.get("component_initialization")
            if reference:
                store.metadata["reference_tokenizer_initialization_seconds"] = reference.initialization_seconds
            store.metadata["gpu_after_loading"] = gpu_memory()
            store.save_metadata()
            pipeline = ReaderPipeline(config, prompts, backend, reference)
            for artifact in read_jsonl(compression_path):
                if artifact["id"] not in pending:
                    continue
                row = pipeline.read(artifact)
                row["reader_config_sha256"] = canonical_hash(asdict(config.reader))
                row["reader_tokenizer"] = {"model_id": config.reader.tokenizer_id or config.reader.model_id,
                                           "revision": config.reader.tokenizer_revision or config.reader.revision,
                                           "synthetic": config.reader.backend == "mock"}
                if reference:
                    row["reference_tokenizer"] = asdict(config.reference_tokenizer)
                store.append(row)
                print(f"read {len(store.ids)}/{len(samples)} {artifact['id']}: {'error' if row.get('error') else 'done'}", flush=True)
        finally:
            if backend:
                backend.close()
            if reference:
                reference.close()
            backend = reference = None
            store.metadata["gpu_after_release"] = release_memory()
    return Path(output_dir) / "predictions.jsonl"


def merge_runs(paths, output_dir):
    """Merge completed disjoint shards with matching stage/data/model identities."""
    metadata = []
    paths = [Path(p) for p in paths]
    for path in paths:
        meta = json.loads((path.parent / "run_metadata.json").read_text(encoding="utf-8"))
        if meta["status"] not in ("complete", "partial") or not meta.get("selected_subset_complete"):
            raise ValueError("Merge requires completed shards (possibly partial splits)")
        if meta.get("output_sha256") != file_hash(path):
            raise ValueError("Shard file checksum mismatch")
        metadata.append(meta)
    identities = [m["identity"] for m in metadata]
    def comparable(identity):
        # Different compression-file hashes are expected for disjoint reader shards.
        return {k: v for k, v in identity.items() if k not in ("selected_ids", "compression_file_sha256")}
    if any(comparable(i) != comparable(identities[0]) for i in identities[1:]):
        raise ValueError("Shard configurations/datasets/stages differ")
    def revisions(meta):
        return {name: {role: info.get("resolved_revision") for role, info in (meta.get(name) or {}).items()}
                for name in ("component_initialization", "compression_initialization")}
    if any(revisions(m) != revisions(metadata[0]) for m in metadata[1:]):
        raise ValueError("Resolved model revisions differ across shards; pin the intended revisions")
    ids = [rid for identity in identities for rid in identity["selected_ids"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Shards overlap; duplicate IDs are forbidden")
    identity = {**identities[0], "selected_ids": ids}
    if identity["stage"] == "read":
        identity["compression_file_sha256"] = canonical_hash([i["compression_file_sha256"] for i in identities])
    filename = "compression_results.jsonl" if identity["stage"] == "compress" else "predictions.jsonl"
    with RunStore(output_dir, filename, identity, {"merged_from": [str(p) for p in paths]}, False) as store:
        store.metadata["component_initialization"] = metadata[0].get("component_initialization")
        store.metadata["compression_initialization"] = metadata[0].get("compression_initialization")
        for path, original in zip(paths, identities):
            shard_seen = set()
            for row in read_jsonl(path):
                if row["id"] in shard_seen or row["id"] not in original["selected_ids"] or row.get("run_identity_sha256") != canonical_hash(original):
                    raise ValueError("Invalid shard record identity or duplicate ID")
                shard_seen.add(row["id"])
                store.append(row)
            if shard_seen != set(original["selected_ids"]):
                raise ValueError("Shard coverage differs from metadata")
    return Path(output_dir) / filename
