"""Original filtering/controller loop, independent of reader execution."""
from copy import deepcopy
from dataclasses import asdict
import re
import time
import traceback
from .accumulation import EvidenceAccumulator
from .controller import Controller
from .decomposition import SentenceDecomposer
from .filtering import percentile_filter
from .prompt_builder import render_paragraphs, render_segments
from .reader import Reader
from .relevance import score_evidence
from .stopping import stop_reason
from .telemetry import measured, sum_known


class CompressionPipeline:
    def __init__(self, config, prompts, encoder=None, controller_backend=None):
        self.config = config
        self.decomposer = SentenceDecomposer(config.decomposition)
        self.encoder = encoder
        self.controller = Controller(
            controller_backend, prompts["controller"], config.itercomp.controller_retries,
            config.itercomp.answerable_follow_up_policy) if controller_backend else None

    def compress(self, sample, method="itercomp", oracle_support_idxs=None):
        c = self.config
        result = {"id": sample.id, "question": sample.question, "method": method,
                  "raw_context": render_paragraphs(sample.paragraphs, c.context),
                  "segment_catalog": [], "selected_evidence_ids": [], "predicted_support_idxs": [],
                  "iterations": 0, "controller_input_tokens": 0, "controller_output_tokens": 0,
                  "controller_llm_calls": 0, "stop_reason": "error", "timings": {},
                  "compressed_context": "", "error": None}
        trace = []
        pending_step = None
        timings = result["timings"]
        if self.encoder:
            self.encoder.reset_telemetry()
        if self.controller:
            self.controller.reset()
            self.controller.backend.begin_sample()
        start = time.perf_counter()
        try:
            if method in ("raw", "oracle"):
                paragraphs = sample.paragraphs
                if method == "oracle":
                    if oracle_support_idxs is None:
                        raise ValueError("Oracle requires explicitly isolated supporting paragraph selection")
                    paragraphs = [p for p in paragraphs if p.idx in oracle_support_idxs]
                result["compressed_context"] = render_paragraphs(paragraphs, c.context)
                result["predicted_support_idxs"] = sorted(p.idx for p in paragraphs)
                result["stop_reason"] = "baseline"
            elif method == "itercomp":
                if self.encoder is None or self.controller is None:
                    raise ValueError("IterCOMP requires encoder and controller")
                with measured(timings, "decomposition_seconds", "cpu"):
                    segments = self.decomposer.decompose(sample.paragraphs)
                result["segment_catalog"] = [asdict(s) for s in segments]
                if not segments:
                    raise ValueError("No nonempty evidence sentences")
                texts = [f"{s.title}: {s.text}" if c.embedding.include_titles and s.title else s.text for s in segments]
                with measured(timings, "evidence_encoding_seconds", c.embedding.device):
                    representations = self.encoder.encode(texts, [s.id for s in segments])
                accumulator = EvidenceAccumulator(segments)
                current_query = sample.question
                seen_queries = set()
                for iteration in range(1, c.itercomp.max_iterations + 1):
                    step_times = {}
                    with measured(step_times, "query_encoding_seconds", c.embedding.device):
                        query = self.encoder.encode([current_query], [f"query_{iteration}"])[0]
                    with measured(step_times, "filter_seconds", "cpu"):
                        ranked = score_evidence(query, representations, segments, c.itercomp.semantic_weight)
                        threshold, retrieved = percentile_filter(ranked, c.itercomp.percentile, c.itercomp.percentile_method)
                        previous = set(accumulator.ids())
                        accumulated = accumulator.update(retrieved)
                    context = render_segments(accumulator.segments(), c.context)
                    result["selected_evidence_ids"] = accumulated
                    result["compressed_context"] = context
                    result["predicted_support_idxs"] = sorted({s.paragraph_idx for s in accumulator.segments()})
                    # Preserve the actual supplied state even when no valid judgment
                    # can be parsed. This is debugging data, never a synthetic trace.
                    pending_step = {"iteration": iteration, "query": current_query,
                                    "retrieved_evidence_ids": retrieved,
                                    "accumulated_evidence_ids": accumulated,
                                    "controller_context": context, "threshold": threshold,
                                    "ranked_evidence": [asdict(s) for s in ranked] if c.itercomp.full_rankings else None}
                    before = len(self.controller.ledger.calls)
                    with measured(step_times, "controller_wall_seconds", c.controller.device):
                        judgment = self.controller.judge(sample.question, context)
                    calls = self.controller.ledger.calls[before:]
                    key = " ".join(re.findall(r"\w+", current_query.casefold()))
                    step = {"iteration": iteration, "query": current_query,
                            "retrieved_evidence_ids": retrieved, "accumulated_evidence_ids": accumulated,
                            "new_evidence_ids": [sid for sid in accumulated if sid not in previous],
                            "removed_evidence_ids": [], "repeated_query": key in seen_queries,
                            "no_new_evidence": not bool(set(accumulated) - previous),
                            "controller_context": context, "answerable": judgment.answerable,
                            "follow_up_question": judgment.follow_up_question,
                            "controller_format_warnings": list(judgment.format_warnings),
                            "threshold": threshold, "percentile": c.itercomp.percentile,
                            "percentile_method": c.itercomp.percentile_method,
                            "controller_calls": deepcopy(calls),
                            "controller_input_tokens": sum_known(*(v["input_tokens"] for v in calls)),
                            "controller_output_tokens": sum_known(*(v["output_tokens"] for v in calls)),
                            "controller_seconds": sum(v["latency_seconds"] for v in calls),
                            "controller_context_tokens": self.controller.backend.count_text(context),
                            **step_times}
                    if c.itercomp.full_rankings:
                        step.update(ranking_complete=True, ranked_evidence=[asdict(s) for s in ranked])
                    trace.append(step)
                    pending_step = None
                    seen_queries.add(key)
                    result["iterations"] = len(trace)
                    reason = stop_reason(judgment.answerable, iteration, c.itercomp.max_iterations)
                    if reason:
                        result["stop_reason"] = reason
                        break
                    current_query = judgment.follow_up_question
                result["iteration_trace"] = trace
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["debug"] = {"partial_iteration_trace": trace, "failed_iteration_state": pending_step,
                               "traceback": traceback.format_exc()}
            # No invented controller judgment or validator-compatible partial trace.
            result.pop("iteration_trace", None)
            result["stop_reason"] = "error"
        finally:
            if method == "itercomp" and self.controller:
                ledger = self.controller.ledger
                result.update(controller_input_tokens=ledger.total("input_tokens"),
                              controller_output_tokens=ledger.total("output_tokens"),
                              controller_llm_calls=len(ledger.calls), controller_calls=deepcopy(ledger.calls))
                result["controller_format_warning_count"] = sum(len(call.get("format_warnings", [])) for call in ledger.calls)
            if method == "itercomp" and self.encoder:
                result["embedding"] = {"forward_calls": self.encoder.forward_calls,
                                       "encoded_texts": self.encoder.encoded_texts,
                                       "truncations": list(self.encoder.events),
                                       "candidate_cache": "per-sample only; reused across iterations"}
            result["compression_seconds"] = time.perf_counter() - start
        return result


class ReaderPipeline:
    def __init__(self, config, prompts, backend, reference_tokenizer=None):
        self.config = config
        self.reader = Reader(backend, prompts["reader"])
        self.reference_tokenizer = reference_tokenizer

    def read(self, artifact):
        record = deepcopy(artifact)
        record.update(predicted_answer="", predicted_answerable=False, reader_raw_response=None,
                      reader_input_tokens=0, reader_output_tokens=0)
        start = time.perf_counter()
        self.reader.ledger.calls.clear()
        self.reader.backend.begin_sample()
        try:
            # Reader-specific accounting is recomputed, without changing any context text.
            with measured(record.setdefault("reader_timings", {}), "token_counting_seconds", "cpu"):
                record["raw_context_tokens"] = self.reader.backend.count_text(artifact["raw_context"])
                record["compressed_context_tokens"] = self.reader.backend.count_text(artifact["compressed_context"])
                for step in record.get("iteration_trace", []):
                    step["context_tokens"] = self.reader.backend.count_text(step["controller_context"])
                if self.reference_tokenizer:
                    record["reference_raw_context_tokens"] = self.reference_tokenizer.count_text(artifact["raw_context"])
                    record["reference_compressed_context_tokens"] = self.reference_tokenizer.count_text(artifact["compressed_context"])
            if not artifact.get("error"):
                answer, answerable, raw = self.reader.answer(artifact["question"], artifact["compressed_context"])
                record.update(predicted_answer=answer, predicted_answerable=answerable, reader_raw_response=raw)
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc), "stage": "read"}
            record["stop_reason"] = "error"
        finally:
            ledger = self.reader.ledger
            record.update(reader_input_tokens=ledger.total("input_tokens"),
                          reader_output_tokens=ledger.total("output_tokens"),
                          reader_llm_calls=len(ledger.calls), reader_calls=deepcopy(ledger.calls))
            record["reader_seconds"] = time.perf_counter() - start
            record["reader_inference_seconds"] = sum(v["latency_seconds"] for v in ledger.calls)
            record["total_seconds"] = artifact["compression_seconds"] + record["reader_seconds"]
            record["llm_calls"] = artifact["controller_llm_calls"] + len(ledger.calls)
        return record
