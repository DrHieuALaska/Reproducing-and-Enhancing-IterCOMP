"""Regression parsing of captured text; pipeline tests use synthetic QA only."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from benchmark import evaluate, write_jsonl
from diagnostics import diagnose
from itercomp.backends.mock import MockBackend
from itercomp.config import config_from_dict
from itercomp.controller import Controller, parse_judgment
from itercomp.pipeline import ReaderPipeline
from itercomp.prompt_builder import load_prompts
from tests.test_core import ROOT, config, sample, compression


CASES = json.loads((ROOT / "tests/fixtures/controller_follow_up_regressions.json").read_text(encoding="utf-8"))["cases"]
WARNING = "unused_follow_up_ignored_for_answerable"


class ControllerNormalizationTests(unittest.TestCase):
    def test_all_three_captured_failures_have_audited_normalization(self):
        self.assertEqual(len(CASES), 3)
        for case in CASES:
            with self.subTest(id=case["id"]):
                with self.assertRaisesRegex(ValueError, "requires a null follow-up"):
                    parse_judgment(case["raw_response"])
                judgment = parse_judgment(case["raw_response"], answerable_follow_up_policy="normalize_unused")
                self.assertTrue(judgment.answerable)
                self.assertIsNone(judgment.follow_up_question)
                self.assertEqual(judgment.format_warnings, (WARNING,))

    def test_valid_decisions_are_unchanged(self):
        for text in ('{"answer":"answerable","follow_up_question":null}',
                     '{"answer":"unanswerable","follow_up_question":"Where was Alice born?"}'):
            self.assertEqual(parse_judgment(text), parse_judgment(text, answerable_follow_up_policy="normalize_unused"))

    def test_malformed_or_ambiguous_decisions_still_fail(self):
        invalid = ["bad JSON", '[]', '{}',
                   '{"answer":"yes","follow_up_question":"Where?"}',
                   '{"answer":"unanswerable","follow_up_question":null}',
                   '{"answer":"unanswerable","follow_up_question":" "}',
                   '{"answer":"answerable","follow_up_question":false}',
                   '{"answer":"answerable","follow_up_question":3}',
                   '{"answer":"answerable","follow_up_question":[]}',
                   '{"answer":"answerable","follow_up_question":{}}',
                   '{"answer":"answerable"}',
                   '{"answer":"answerable","follow_up_question":null,"extra":true}',
                   '{"answer":"answerable","answer":"unanswerable","follow_up_question":"Where?"}']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_judgment(text, answerable_follow_up_policy="normalize_unused")

    def test_controller_preserves_raw_response_warning_usage_and_one_call(self):
        for case in CASES:
            with self.subTest(id=case["id"]):
                backend = MockBackend([case["raw_response"]])
                controller = Controller(backend, "$question\n$context", answerable_follow_up_policy="normalize_unused")
                result = controller.judge("Synthetic question?", "Synthetic information.")
                self.assertTrue(result.answerable)
                self.assertEqual(len(controller.ledger.calls), 1)
                call = controller.ledger.calls[0]
                self.assertEqual(call["raw_response"], case["raw_response"])
                self.assertEqual(call["parsed_judgment"], {"answer": "answerable", "follow_up_question": None})
                self.assertEqual(call["format_warnings"], [WARNING])
                self.assertNotIn("parse_error", call)
                self.assertEqual(controller.ledger.total("output_tokens"), backend.count_text(case["raw_response"]))

    def test_synthetic_pipeline_reaches_reader_and_keeps_model_judgment_distinct(self):
        for case in CASES:
            with self.subTest(id=case["id"]):
                pipeline, _, _ = compression(responses=[case["raw_response"]])
                artifact = pipeline.compress(sample())
                self.assertIsNone(artifact["error"])
                self.assertEqual(artifact["stop_reason"], "answerable")
                self.assertEqual(artifact["iterations"], 1)
                self.assertEqual(artifact["controller_llm_calls"], 1)
                self.assertEqual(artifact["controller_format_warning_count"], 1)
                self.assertEqual(artifact["iteration_trace"][0]["controller_format_warnings"], [WARNING])
                self.assertIsNone(artifact["iteration_trace"][0]["follow_up_question"])
                backend = MockBackend(["INSUFFICIENT_INFORMATION"])
                prediction = ReaderPipeline(config(), load_prompts(config()), backend).read(artifact)
                self.assertIsNone(prediction["error"])
                self.assertEqual(prediction["reader_llm_calls"], 1)
                self.assertEqual(prediction["llm_calls"], 2)
                self.assertFalse(prediction["predicted_answerable"])
                self.assertEqual(prediction["stop_reason"], "answerable")

    def test_strict_policy_remains_available(self):
        c = config()
        c.itercomp.answerable_follow_up_policy = "strict"
        pipeline, _, _ = compression(c, [CASES[0]["raw_response"]])
        artifact = pipeline.compress(sample())
        self.assertEqual(artifact["stop_reason"], "error")
        self.assertEqual(artifact["controller_format_warning_count"], 0)
        self.assertNotIn("iteration_trace", artifact)
        with self.assertRaises(ValueError):
            config_from_dict({"itercomp": {"answerable_follow_up_policy": "guess"}})

    def test_normalization_trace_passes_existing_benchmark_and_diagnostics(self):
        pipeline, _, _ = compression(responses=[CASES[0]["raw_response"]])
        artifact = pipeline.compress(sample())
        prediction = ReaderPipeline(config(), load_prompts(config()), MockBackend(["INSUFFICIENT_INFORMATION"])).read(artifact)
        truth = json.loads((ROOT / "tests/fixtures/gold.jsonl").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            write_jsonl(root / "gold.jsonl", [truth])
            write_jsonl(root / "predictions.jsonl", [prediction])
            evaluate(root / "gold.jsonl", root / "predictions.jsonl", root / "scored")
            diagnose(root / "gold.jsonl", root / "predictions.jsonl", root / "diagnostics")
            self.assertEqual(json.loads((root / "scored/summary.json").read_text())["overall"]["n"], 1)
            self.assertEqual(json.loads((root / "diagnostics/diagnostic_summary.json").read_text())["trace_coverage"]["traced"], 1)


if __name__ == "__main__":
    unittest.main()
