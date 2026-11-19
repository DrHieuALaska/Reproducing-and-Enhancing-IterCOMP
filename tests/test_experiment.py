from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmark import write_jsonl, read_jsonl, evaluate
from diagnostics import diagnose
from itercomp.artifacts import inspect_artifacts
from itercomp.config import canonical_hash
from itercomp.experiment import run_compression, run_reader, merge_runs
from itercomp.prompt_builder import load_prompts, prompt_hashes
from itercomp.serialization import RunStore, recover_jsonl, file_hash
from tests.test_core import config, ROOT


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.c = config()
        self.inputs = self.root / "inputs.jsonl"
        self.gold = self.root / "gold.jsonl"
        self.raw = json.loads((ROOT / "tests/fixtures/inputs.jsonl").read_text())
        self.truth = json.loads((ROOT / "tests/fixtures/gold.jsonl").read_text())
        write_jsonl(self.inputs, [self.raw])
        write_jsonl(self.gold, [self.truth])
        self.capture = redirect_stdout(io.StringIO())
        self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None, None, None)
        self.temp.cleanup()

    def compress(self, **kwargs):
        return run_compression(self.c, self.inputs, self.root / "compress", **kwargs)

    def read(self, artifact, **kwargs):
        return run_reader(self.c, self.inputs, artifact, self.root / "read", **kwargs)

    def two_samples(self):
        second = deepcopy(self.raw)
        second["id"] = "synthetic-second"
        truth = deepcopy(self.truth)
        truth["id"] = second["id"]
        write_jsonl(self.inputs, [self.raw, second])
        write_jsonl(self.gold, [self.truth, truth])

    def test_staged_pipeline_benchmark_diagnostics_compatibility(self):
        artifact = self.compress()
        with patch("itercomp.experiment.create_encoder", side_effect=AssertionError("reader loaded encoder")):
            predictions = self.read(artifact)
        row = next(read_jsonl(predictions))
        self.assertEqual(row["predicted_answer"], "Paris")
        self.assertEqual(row["iterations"], 2)
        self.assertEqual(row["predicted_support_idxs"], [3, 7])
        evaluate(self.gold, predictions, self.root / "scored")
        diagnose(self.gold, predictions, self.root / "diagnostics")
        score = json.loads((self.root / "scored/summary.json").read_text())
        self.assertEqual(score["overall"]["answer_em"], 1)
        diag = json.loads((self.root / "diagnostics/diagnostic_summary.json").read_text())
        self.assertEqual(diag["trace_coverage"]["traced"], 1)
        for dirname, name in (("scored", "per_sample.csv"), ("scored", "official_predictions.jsonl"),
                              ("diagnostics", "per_iteration.csv"), ("diagnostics", "diagnostic_per_sample.csv"),
                              ("diagnostics", "annotation_queue.jsonl")):
            self.assertTrue((self.root / dirname / name).is_file())

    def test_failure_record_passes_evaluators_and_preserves_errors(self):
        self.c.controller.mock_responses = ["malformed"]
        artifact = self.compress()
        predictions = self.read(artifact)
        row = next(read_jsonl(predictions))
        self.assertEqual(row["predicted_answer"], "")
        self.assertEqual(row["stop_reason"], "error")
        self.assertNotIn("iteration_trace", row)
        self.assertTrue((self.root / "compress/errors.jsonl").exists())
        self.assertTrue((self.root / "read/errors.jsonl").exists())
        evaluate(self.gold, predictions, self.root / "scored")
        diagnose(self.gold, predictions, self.root / "diagnostics")

    def test_raw_and_isolated_oracle(self):
        for method, supports in (("raw", [3, 7, 12]), ("oracle", [3, 7])):
            with self.subTest(method=method), patch("itercomp.experiment.create_encoder", side_effect=AssertionError("baseline encoder")), patch("itercomp.experiment.create_backend", side_effect=AssertionError("baseline controller")):
                artifact = run_compression(self.c, self.inputs, self.root / method, method,
                                           self.gold if method == "oracle" else None)
            row = next(read_jsonl(artifact))
            self.assertIsNone(row["error"])
            self.assertEqual(row["predicted_support_idxs"], supports)
            self.assertEqual(row["controller_input_tokens"], 0)
            self.assertEqual(row["iterations"], 0)
            self.assertNotIn("iteration_trace", row)
            self.assertNotIn("paragraph_support_idx", row["compressed_context"])
        with self.assertRaises(ValueError):
            self.compress(gold_path=self.gold)
        with self.assertRaises(ValueError):
            run_compression(self.c, self.gold, self.root / "leak")

    def test_reader_change_reuses_identical_compression(self):
        artifact = self.compress()
        original_hash = file_hash(artifact)
        first = self.read(artifact)
        self.c.reader.model_id = "synthetic-another-reader"
        self.c.reader.mock_responses = ["London"]
        second = run_reader(self.c, self.inputs, artifact, self.root / "other_reader")
        a, b = next(read_jsonl(first)), next(read_jsonl(second))
        self.assertEqual(a["compressed_context"], b["compressed_context"])
        self.assertEqual(a["compression_seconds"], b["compression_seconds"])
        self.assertEqual(file_hash(artifact), original_hash)
        self.assertEqual(b["predicted_answer"], "London")

    def test_reuse_rejects_compression_setting_changes_before_loading(self):
        artifact = self.compress()
        changes = [("embedding", "model_id", "other"), ("embedding", "batch_size", 1),
                   ("controller", "model_id", "other"), ("controller", "quantization", "int8"),
                   ("itercomp", "percentile", 80), ("itercomp", "semantic_weight", .5),
                   ("itercomp", "max_iterations", 4), ("context", "include_titles", True)]
        for section, name, value in changes:
            c = deepcopy(self.c)
            setattr(getattr(c, section), name, value)
            with self.subTest(section=section, name=name), patch("itercomp.experiment.create_backend", side_effect=AssertionError("must validate first")), self.assertRaises(ValueError):
                run_reader(c, self.inputs, artifact, self.root / f"invalid_{section}_{name}")

    def test_changed_prompt_dataset_and_tampered_artifact_rejected(self):
        artifact = self.compress()
        prompt = self.root / "controller.txt"
        prompt.write_text("Changed $question $context", encoding="utf-8")
        self.c.prompts.controller = str(prompt)
        with self.assertRaises(ValueError):
            self.read(artifact)
        self.c.prompts.controller = None
        row = deepcopy(self.raw)
        row["question"] = "Changed question?"
        write_jsonl(self.inputs, [row])
        with self.assertRaises(ValueError):
            self.read(artifact)
        write_jsonl(self.inputs, [self.raw])
        with artifact.open("ab") as handle:
            handle.write(b" ")
        with self.assertRaises(ValueError):
            self.read(artifact)

    def test_compression_resume_skips_completed_and_repairs_tail(self):
        artifact = self.compress()
        original = artifact.read_bytes()
        with artifact.open("ab") as handle:
            handle.write(b'{"id":"unfinished')
        with patch("itercomp.experiment.create_backend", side_effect=AssertionError("completed sample rerun")):
            self.compress(resume=True)
        self.assertEqual(artifact.read_bytes(), original)
        self.assertEqual(len(list(artifact.parent.glob("*.incomplete-*"))), 1)
        with self.assertRaises(FileExistsError):
            self.compress()
        self.c.itercomp.percentile = 80
        with self.assertRaises(ValueError):
            self.compress(resume=True)

    def test_interrupted_compression_resume_and_independent_reader_resume(self):
        self.two_samples()
        append = RunStore.append
        def interrupt_after_first(store, row):
            append(store, row)
            raise KeyboardInterrupt("synthetic notebook disconnect")
        with patch.object(RunStore, "append", interrupt_after_first), self.assertRaises(KeyboardInterrupt):
            self.compress()
        meta = json.loads((self.root / "compress/run_metadata.json").read_text())
        self.assertEqual(meta["status"], "interrupted")
        self.assertEqual(meta["completed_count"], 1)
        artifact = self.compress(resume=True)
        self.assertEqual(len(list(read_jsonl(artifact))), 2)
        with patch.object(RunStore, "append", interrupt_after_first), self.assertRaises(KeyboardInterrupt):
            self.read(artifact)
        predictions = self.read(artifact, resume=True)
        self.assertEqual(len(list(read_jsonl(predictions))), 2)
        self.assertTrue(all(row["predicted_answer"] == "Paris" for row in read_jsonl(predictions)))
        with patch("itercomp.experiment.create_backend", side_effect=AssertionError("reader rerun")):
            self.read(artifact, resume=True)

    def test_partial_metadata_matching_gold_and_disjoint_merge(self):
        self.two_samples()
        first = self.compress(limit=1)
        meta = json.loads((first.parent / "run_metadata.json").read_text())
        self.assertEqual(meta["status"], "partial")
        self.assertFalse(meta["complete_split"])
        self.assertTrue(meta["selected_subset_complete"])
        second = run_compression(self.c, self.inputs, self.root / "second", sample_ids=["synthetic-second"])
        merged = merge_runs([first, second], self.root / "merged")
        self.assertEqual(len(list(read_jsonl(merged))), 2)
        predictions = self.read(merged)
        evaluate(self.gold, predictions, self.root / "scored")
        with self.assertRaises(ValueError):
            merge_runs([first, first], self.root / "overlap")
        partial_predictions = run_reader(self.c, self.inputs, first, self.root / "partial_read")
        result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_run.py"),
                                 "--gold", str(self.gold), "--predictions", str(partial_predictions),
                                 "--out", str(self.root / "partial_scored"), "--allow-partial"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        evaluation_meta = json.loads((self.root / "partial_scored/evaluation_metadata.json").read_text())
        self.assertFalse(evaluation_meta["complete_split"])

    def test_reader_shards_merge(self):
        self.two_samples()
        artifact = self.compress()
        first = run_reader(self.c, self.inputs, artifact, self.root / "r1", sample_ids=[self.raw["id"]])
        second = run_reader(self.c, self.inputs, artifact, self.root / "r2", sample_ids=["synthetic-second"])
        merged = merge_runs([first, second], self.root / "rmerged")
        self.assertEqual(len(list(read_jsonl(merged))), 2)
        evaluate(self.gold, merged, self.root / "scored")

    def test_combined_cli_is_synthetic_only_in_this_test(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_experiment.py"),
                                 "--stage", "both", "--config", str(ROOT / "configs/mock_synthetic.yaml"),
                                 "--inputs", str(self.inputs), "--output-dir", str(self.root / "combined")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "combined/predictions.jsonl").is_file())

    def test_jsonl_tail_recovery_and_corruption(self):
        path = self.root / "tail.jsonl"
        path.write_bytes(b'{"id":"ok"}')
        self.assertEqual(recover_jsonl(path, repair=True)[0]["id"], "ok")
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        path.write_bytes(b'{"id":"ok"}\nBAD\n{"id":"next"}\n')
        with self.assertRaises(ValueError):
            recover_jsonl(path, repair=True)
        path.write_bytes(b'{"id":"same"}\n{"id":"same"}\n')
        with self.assertRaises(ValueError):
            recover_jsonl(path, repair=True)
        path.write_bytes(b'{"id":"ok"}\nBAD\n')
        with self.assertRaises(ValueError):
            recover_jsonl(path, repair=True)

    def test_resume_recovers_separate_incomplete_error_log(self):
        self.c.controller.mock_responses = ["bad"]
        artifact = self.compress()
        error_path = artifact.parent / "errors.jsonl"
        error_path.write_bytes(b'{"id":"unfinished')
        self.compress(resume=True)
        errors = list(read_jsonl(error_path))
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["id"], self.raw["id"])
        self.assertTrue(errors[0]["recovered"])
        self.assertEqual(errors[0]["controller_calls"][0]["raw_response"], "bad")
        meta = json.loads((artifact.parent / "run_metadata.json").read_text())
        self.assertEqual(meta["failed_sample_ids"], [self.raw["id"]])

    def test_imports_do_not_load_gpu_libraries(self):
        code = "import sys; sys.path.insert(0, 'src'); import itercomp.experiment; assert not {'torch', 'transformers', 'FlagEmbedding', 'bitsandbytes'} & set(sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_quantization_failure_prevents_encoder_and_llm_loading(self):
        from itercomp.backends.mock import MockBackend
        from itercomp.embedding import MockEncoder
        backend = MockBackend([])
        encoder = MockEncoder(self.c.embedding)
        with patch.object(backend, "preflight", side_effect=RuntimeError("native CUDA backend missing")), patch.object(backend, "prepare") as llm_prepare, patch.object(encoder, "prepare") as encoder_prepare, patch("itercomp.experiment.create_backend", return_value=backend), patch("itercomp.experiment.create_encoder", return_value=encoder):
            with self.assertRaisesRegex(RuntimeError, "native CUDA backend missing"):
                self.compress()
            llm_prepare.assert_not_called()
            encoder_prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
