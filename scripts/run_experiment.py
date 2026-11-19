"""Cloud GPU entry point. Merely importing this file never loads checkpoints."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from itercomp.config import load_config
from itercomp.experiment import run_compression, run_reader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("compress", "read", "both"), default="both")
    parser.add_argument("--method", choices=("itercomp", "raw", "oracle"), default="itercomp")
    parser.add_argument("--config", default="configs/itercomp_llama_mistral.yaml")
    parser.add_argument("--split", choices=("smoke", "calibration", "evaluation"), default="smoke")
    parser.add_argument("--data-dir", default="data/benchmark")
    parser.add_argument("--inputs", help="Explicit input-only JSONL, also required to verify reader artifacts")
    parser.add_argument("--oracle-gold", help="Allowed exclusively for --method oracle compression")
    parser.add_argument("--compression-results")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-ids", help="JSON array file, or newline-delimited IDs; dataset order is preserved")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trace-verbosity", choices=("full", "compact"))
    for component in ("embedding", "controller", "reader"):
        parser.add_argument(f"--{component}-device")
        parser.add_argument(f"--{component}-model")
    args = parser.parse_args(argv)
    overrides = {}
    if args.seed is not None:
        overrides = {"seed": args.seed, "controller": {"seed": args.seed}, "reader": {"seed": args.seed}}
    for component in ("embedding", "controller", "reader"):
        for flag, key in (("device", "device"), ("model", "model_id")):
            value = getattr(args, f"{component}_{flag}")
            if value:
                overrides.setdefault(component, {})[key] = value
    if args.trace_verbosity:
        overrides["itercomp"] = {"full_rankings": args.trace_verbosity == "full"}
    config = load_config(args.config, overrides)
    sample_ids = None
    if args.sample_ids:
        import json
        text = Path(args.sample_ids).read_text(encoding="utf-8").strip()
        sample_ids = json.loads(text) if text.startswith("[") else text.splitlines()
    inputs = args.inputs or str(Path(args.data_dir) / args.split / "inputs.jsonl")
    if args.stage == "read" and (not args.compression_results or args.oracle_gold):
        parser.error("--stage read requires --compression-results and never accepts --oracle-gold")
    if args.stage == "compress" and args.compression_results:
        parser.error("--compression-results is an input for read/both, not compress")
    if args.stage == "both" and args.compression_results and args.oracle_gold:
        parser.error("Reusing compression never accepts --oracle-gold")
    compression = args.compression_results
    if args.stage in ("compress", "both") and not compression:
        out = Path(args.output_dir) / "compression" if args.stage == "both" else args.output_dir
        compression = run_compression(config, inputs, out, args.method, args.oracle_gold,
                                      args.limit, sample_ids, args.resume)
    if args.stage in ("read", "both"):
        run_reader(config, inputs, compression, args.output_dir, args.limit, sample_ids, args.resume)


if __name__ == "__main__":
    main()
