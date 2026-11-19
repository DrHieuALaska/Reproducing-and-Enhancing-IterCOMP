from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

from itercomp.accumulation import EvidenceAccumulator
from itercomp.backends.mock import MockBackend
from itercomp.config import Config, load_config, config_from_dict
from itercomp.controller import Controller, parse_judgment
from itercomp.dataset import input_sample, load_inputs
from itercomp.decomposition import SentenceDecomposer
from itercomp.embedding import MockEncoder
from itercomp.filtering import percentile_filter
from itercomp.pipeline import CompressionPipeline, ReaderPipeline
from itercomp.prompt_builder import load_prompts, render_segments
from itercomp.relevance import semantic_score, lexical_score, score_evidence
from itercomp.schemas import EvidenceSegment, EvidenceScore, Representation

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/mock_synthetic.yaml")


def sample():
    return load_inputs(ROOT / "tests/fixtures/inputs.jsonl")[0]


def compression(c=None, responses=None):
    c = c or config()
    encoder = MockEncoder(c.embedding)
    backend = MockBackend(responses if responses is not None else c.controller.mock_responses)
    pipeline = CompressionPipeline(c, load_prompts(c), encoder, backend)
    return pipeline, encoder, backend


class CoreTests(unittest.TestCase):
    def test_sentence_offsets_abbreviations_unicode_empty(self):
        s = input_sample({"id": "SECRET_ID", "question": "Q?", "paragraphs": [
            {"idx": 9, "title": "T", "paragraph_text": "Dr. Smith visited Hà Nội.  She stayed!\nReally?"},
            {"idx": 2, "title": "", "paragraph_text": "  \n\t"},
            {"idx": 3, "title": "U", "paragraph_text": "One sentence."}]})
        decomposer = SentenceDecomposer(config().decomposition)
        segments = decomposer.decompose(s.paragraphs)
        self.assertEqual([x.id for x in segments], ["p9_s0", "p9_s1", "p9_s2", "p3_s0"])
        self.assertEqual(segments[0].text, "Dr. Smith visited Hà Nội.")
        self.assertEqual(segments, decomposer.decompose(s.paragraphs))
        for p in s.paragraphs:
            local = [x for x in segments if x.paragraph_idx == p.idx]
            for seg in local:
                self.assertEqual(p.paragraph_text[seg.start_char:seg.end_char], seg.text)
            self.assertTrue(all(a.end_char <= b.start_char for a, b in zip(local, local[1:])))

    def test_semantic_inner_product_is_not_cosine(self):
        self.assertEqual(semantic_score((2, 3), (4, 5)), 23)
        with self.assertRaises(ValueError):
            semantic_score((1,), (1, 2))

    def test_sparse_hand_calculation(self):
        self.assertEqual(lexical_score({"10": 2, "20": 3, "30": 7}, {"10": 4, "20": 5, "40": 99}), 23)
        self.assertEqual(lexical_score({"10": 1}, {"20": 1}), 0)
        self.assertEqual(lexical_score({}, {}), 0)

    def test_dual_scores_precision_ties_and_order(self):
        segments = [EvidenceSegment(f"p{i}_s0", i, 0, "S", "") for i in (9, 2)]
        query = Representation((2, 3), {"t": 2})
        evidence = Representation((4, 5), {"t": 3})
        scores = score_evidence(query, [evidence, evidence], segments)
        self.assertAlmostEqual(scores[0].dual, .6 * 23 + .4 * 6)
        self.assertEqual([s.id for s in scores], ["p9_s0", "p2_s0"])

    def test_nonfinite_scores_rejected(self):
        segment = EvidenceSegment("p0_s0", 0, 0, "S", "")
        for bad in (math.inf, -math.inf, math.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_evidence(Representation((1,), {}), [Representation((bad,), {})], [segment])

    def test_percentile_linear_and_ties(self):
        scores = [EvidenceScore(str(i), float(i), 0, float(i)) for i in range(10)]
        threshold, ids = percentile_filter(scores)
        self.assertAlmostEqual(threshold, 8.1)
        self.assertEqual(ids, ["9"])
        tied = [EvidenceScore(str(i), 1, 0, 1) for i in range(10)]
        self.assertEqual(len(percentile_filter(tied)[1]), 10)
        self.assertEqual(len(percentile_filter(scores, 0)[1]), 10)
        self.assertEqual(percentile_filter(scores, 100)[1], ["9"])
        self.assertEqual(percentile_filter(scores[:1])[1], ["0"])
        with self.assertRaises(ValueError):
            percentile_filter([])
        with self.assertRaises(ValueError):
            percentile_filter([EvidenceScore("s", 0, 0, math.nan)])

    def test_monotonic_deduplication_original_order(self):
        segments = SentenceDecomposer(config().decomposition).decompose(sample().paragraphs)
        accumulator = EvidenceAccumulator(segments)
        self.assertEqual(accumulator.update(["p7_s0", "p7_s0"]), ["p7_s0"])
        self.assertEqual(accumulator.update(["p3_s0"]), ["p3_s0", "p7_s0"])
        self.assertEqual(accumulator.update([]), ["p3_s0", "p7_s0"])
        with self.assertRaises(ValueError):
            accumulator.update(["unknown"])

    def test_controller_json(self):
        self.assertTrue(parse_judgment('```json\n{"answer":"answerable","follow_up_question":null}\n```').answerable)
        self.assertEqual(parse_judgment('{"answer":"unanswerable","follow_up_question":"Where?"}').follow_up_question, "Where?")
        invalid = ["not JSON", "[]", "{}", '{"answer":"yes","follow_up_question":null}',
                   '{"answer":"answerable","follow_up_question":"Why?"}',
                   '{"answer":"unanswerable","follow_up_question":null}',
                   '{"answer":"unanswerable","follow_up_question":"  "}',
                   '{"answer":"answerable","follow_up_question":null,"extra":1}',
                   '{"answer":"answerable","answer":"unanswerable","follow_up_question":"Where?"}']
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_judgment(text)

    def test_controller_bounded_retry_counts_invalid_response(self):
        backend = MockBackend(["bad", '{"answer":"answerable","follow_up_question":null}'])
        controller = Controller(backend, "$question\n$context", retries=1)
        self.assertTrue(controller.judge("Q?", "C").answerable)
        self.assertEqual(len(controller.ledger.calls), 2)
        self.assertIn("parse_error", controller.ledger.calls[0])
        self.assertGreater(controller.ledger.total("output_tokens"), backend.count_text(backend.messages[-1][0]["content"]))

    def test_two_hop_original_question_provenance_full_rankings_tokens(self):
        pipeline, encoder, backend = compression()
        s = sample()
        row = pipeline.compress(s)
        self.assertIsNone(row["error"])
        self.assertEqual(row["iterations"], 2)
        self.assertEqual(row["stop_reason"], "answerable")
        self.assertEqual(row["selected_evidence_ids"], ["p3_s0", "p7_s0"])
        self.assertEqual(row["predicted_support_idxs"], [3, 7])
        self.assertEqual(row["iteration_trace"][0]["retrieved_evidence_ids"], ["p3_s0"])
        self.assertEqual(row["iteration_trace"][1]["retrieved_evidence_ids"], ["p7_s0"])
        self.assertEqual(encoder.encoded_texts, 5)  # 3 candidates once, 2 queries
        self.assertEqual(encoder.forward_calls, 3)
        self.assertEqual(row["controller_llm_calls"], 2)
        self.assertEqual(row["controller_input_tokens"], sum(c["input_tokens"] for c in row["controller_calls"]))
        for step in row["iteration_trace"]:
            ranking = step["ranked_evidence"]
            self.assertEqual(set(v["id"] for v in ranking), set(s["id"] for s in row["segment_catalog"]))
            self.assertEqual([v["dual"] for v in ranking], sorted([v["dual"] for v in ranking], reverse=True))
        self.assertEqual(row["iteration_trace"][1]["query"], "Where was Alice born?")
        for messages in backend.messages:
            prompt = messages[0]["content"]
            self.assertIn("Question: " + s.question, prompt)
            self.assertNotIn(s.id, prompt)
        reader_backend = MockBackend(["Paris"])
        result = ReaderPipeline(config(), load_prompts(config()), reader_backend).read(row)
        self.assertEqual(result["predicted_answer"], "Paris")
        self.assertTrue(result["predicted_answerable"])
        self.assertEqual(result["llm_calls"], 3)
        self.assertEqual(result["raw_context_tokens"], reader_backend.count_text(row["raw_context"]))
        self.assertEqual(result["compressed_context_tokens"], reader_backend.count_text(row["compressed_context"]))
        self.assertEqual(result["total_seconds"], row["compression_seconds"] + result["reader_seconds"])

    def test_early_answerable(self):
        pipeline, _, _ = compression(responses=['{"answer":"answerable","follow_up_question":null}'])
        row = pipeline.compress(sample())
        self.assertEqual(row["iterations"], 1)
        self.assertEqual(row["stop_reason"], "answerable")

    def test_repeated_queries_and_no_progress_reach_five(self):
        q = sample().question
        response = json.dumps({"answer": "unanswerable", "follow_up_question": q})
        pipeline, _, _ = compression(responses=[response] * 5)
        row = pipeline.compress(sample())
        self.assertEqual(row["stop_reason"], "max_iterations")
        self.assertEqual(row["iterations"], 5)
        self.assertTrue(row["iteration_trace"][-1]["repeated_query"])
        self.assertTrue(row["iteration_trace"][-1]["no_new_evidence"])
        self.assertFalse(row["iteration_trace"][-1]["answerable"])
        backend = MockBackend(["Paris"])
        prediction = ReaderPipeline(config(), load_prompts(config()), backend).read(row)
        # Reader answerability is independent of final controller judgment.
        self.assertTrue(prediction["predicted_answerable"])
        self.assertEqual(prediction["stop_reason"], "max_iterations")

    def test_abstention_independent_of_controller(self):
        pipeline, _, _ = compression()
        row = pipeline.compress(sample())
        backend = MockBackend(["INSUFFICIENT_INFORMATION"])
        prediction = ReaderPipeline(config(), load_prompts(config()), backend).read(row)
        self.assertEqual(prediction["predicted_answer"], "")
        self.assertFalse(prediction["predicted_answerable"])
        self.assertEqual(prediction["stop_reason"], "answerable")

    def test_bad_controller_emits_failure_without_fabricated_trace(self):
        pipeline, _, _ = compression(responses=["bad"])
        row = pipeline.compress(sample())
        self.assertEqual(row["stop_reason"], "error")
        self.assertNotIn("iteration_trace", row)
        self.assertEqual(row["controller_llm_calls"], 1)
        self.assertEqual(row["controller_calls"][0]["raw_response"], "bad")
        self.assertEqual(row["debug"]["failed_iteration_state"]["controller_context"], "The author of Book X is Alice.")
        self.assertNotIn("answerable", row["debug"]["failed_iteration_state"])
        backend = MockBackend(["unused"])
        result = ReaderPipeline(config(), load_prompts(config()), backend).read(row)
        self.assertFalse(result["predicted_answerable"])
        self.assertEqual(result["reader_llm_calls"], 0)
        self.assertEqual(result["llm_calls"], 1)

    def test_gold_fields_rejected_and_ids_do_not_affect_decisions(self):
        raw = json.loads((ROOT / "tests/fixtures/inputs.jsonl").read_text())
        for field in ("answer", "answer_aliases", "question_decomposition", "answerable", "hop"):
            candidate = deepcopy(raw)
            candidate[field] = "SECRET_GOLD"
            with self.subTest(field=field), self.assertRaises(ValueError):
                input_sample(candidate)
        candidate = deepcopy(raw)
        candidate["paragraphs"][0]["is_supporting"] = True
        with self.assertRaises(ValueError):
            input_sample(candidate)
        a, _, backend_a = compression()
        b, _, backend_b = compression()
        first = a.compress(sample())
        second = b.compress(replace(sample(), id="4hop_SECRET_GOLD"))
        self.assertEqual(first["selected_evidence_ids"], second["selected_evidence_ids"])
        self.assertEqual(backend_a.messages, backend_b.messages)

    def test_config_profiles_and_validation(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            c = load_config(path)
            self.assertEqual(c.itercomp.semantic_weight, .6)
            self.assertEqual(c.itercomp.percentile, 90)
            self.assertEqual(c.itercomp.max_iterations, 5)
        for update in ({"typo": 1}, {"embedding": {"typo": 1}},
                       {"itercomp": {"max_iterations": 0}}, {"itercomp": {"percentile": 101}},
                       {"itercomp": {"semantic_weight": float("nan")}},
                       {"reader": {"quantization": "magic"}}):
            with self.subTest(update=update), self.assertRaises((ValueError, TypeError)):
                config_from_dict(update)

    def test_titles_explicitly_configurable(self):
        c = config()
        segments = SentenceDecomposer(c.decomposition).decompose(sample().paragraphs)
        self.assertNotIn("Book X:", render_segments(segments, c.context))
        c.context.include_titles = True
        self.assertIn("Book X: The author", render_segments(segments, c.context))

    def test_fixed_reference_tokenizer_additional_counts(self):
        pipeline, _, _ = compression()
        artifact = pipeline.compress(sample())
        class Reference:
            def count_text(self, text):
                return len(text.split())
        result = ReaderPipeline(config(), load_prompts(config()), MockBackend(["Paris"]), Reference()).read(artifact)
        self.assertEqual(result["reference_raw_context_tokens"], len(artifact["raw_context"].split()))
        self.assertEqual(result["reference_compressed_context_tokens"], len(artifact["compressed_context"].split()))
        self.assertEqual(result["compressed_context"], artifact["compressed_context"])


if __name__ == "__main__":
    unittest.main()
