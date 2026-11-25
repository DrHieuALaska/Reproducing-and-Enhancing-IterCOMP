# MuSiQue-Ans benchmark starter

Python 3.10+. The benchmark and diagnostic tools use only the standard library. A modular original IterCOMP implementation is available under `src/itercomp`, with separate cloud GPU compression and reader stages.

Download the exact v1.0 answerable train/dev JSONL files from the [official repository](https://github.com/stonybrooknlp/musique) or [mirror](https://huggingface.co/datasets/voidful/MuSiQue/tree/main). Place under `data/raw/`.

```sh
python benchmark.py analyze data/raw/musique_ans_v1.0_dev.jsonl
python benchmark.py analyze data/raw/musique_ans_v1.0_train.jsonl
python benchmark.py prepare --train data/raw/musique_ans_v1.0_train.jsonl --dev data/raw/musique_ans_v1.0_dev.jsonl
```

Default output: 30 train smoke examples, 150 disjoint train calibration examples, all 2,417 dev questions for evaluation. Seed and hashes recorded in manifest. Use only `inputs.jsonl` in the ordinary model pipeline. `gold.jsonl` includes labels for scoring and explicitly labeled oracle experiments. Hide hop information and ID from actual LLM prompts (IDs encode information about the question structure). Analyze reports word counts, not tokenizer counts.

Each method should write one prediction per input row, e.g. (illustrative):

```json
{"id":"example-id","predicted_answer":"Paris","predicted_support_idxs":[3,7],"predicted_answerable":true,"raw_context_tokens":2500,"compressed_context_tokens":350,"controller_input_tokens":800,"controller_output_tokens":50,"reader_input_tokens":420,"reader_output_tokens":5,"compression_seconds":1.2,"reader_seconds":0.5,"total_seconds":1.7,"iterations":2,"llm_calls":3}
```

Only ID, answer, support IDs and answerability are mandatory. For compression experiments, use *all source paragraph IDs represented in the retained evidence* as `predicted_support_idxs` (a paragraph coverage proxy; not proof the required sentence was selected). For Raw, use all paragraph IDs; Oracle uses gold IDs only in the explicitly labeled upper-bound run. `predicted_answerable` refers to the prediction on the original input task, not necessarily the controller's last judgment; log that separately. Use one identical reader tokenizer and consistent context formatting to count raw/compressed context, excluding fixed instruction and question tokens. Controller and reader input/output counts include all respective calls and overhead. Zero is appropriate for known absent stages; omit unknown telemetry. Measure actual total wall-clock time. Missing telemetry is reported through metric counts; don't compare partial averages blindly.

```sh
python benchmark.py evaluate --gold data/benchmark/evaluation/gold.jsonl --predictions results/raw/predictions.jsonl --out results/raw/scored
```

Outputs summary.json (metrics 0–1, not 0–100), per_sample.csv, official_predictions.jsonl (ordered according to gold). Check official scores after cloning the original repository:

```sh
python path/to/musique/evaluate_v1.0.py results/raw/scored/official_predictions.jsonl data/benchmark/evaluation/gold.jsonl
```

The local answer metrics implement the standard MuSiQue-style normalized answer EM and multiset token F1 (maximum across gold answer and aliases); support F1 is paragraph set overlap. Verify against official evaluator before publication. Mean compression ratio is a per-example macro average; record the same definition across methods and check the paper's aggregation before claiming exact reproduction. All methods must cover the same IDs. A run with failed inference should record an empty answer and appropriate telemetry rather than silently skip the sample.

Sources: [dataset schema](https://github.com/stonybrooknlp/musique/blob/main/raw_data_to_official_format.py), [official evaluation](https://github.com/stonybrooknlp/musique/blob/main/evaluate_v1.0.py), [answer metric](https://github.com/stonybrooknlp/musique/blob/main/metrics/answer.py), [support metric](https://github.com/stonybrooknlp/musique/blob/main/metrics/support.py).

---

# Version 2: architecture diagnostics

The original CLI and mandatory prediction schema remain compatible. `benchmark.py evaluate` additionally emits support EM and token-weighted compression ratio. All new diagnostic experiments use `diagnostics.py`; no inference, model training, or download is performed by these scripts.

## 1. Trace contract

Store a complete `segment_catalog` for each example (all segments before filtering). Every segment has a stable unique `id`, original `paragraph_idx`, `sentence_idx` and exact `text`. For alternative window methods, retain stable IDs and original source paragraph; if a segment spans multiple paragraphs, adapt the contract to explicit multi-source mapping before using it. Within-paragraph windows are supported as single-source units. Do not compare raw segment-count novelty across different segmentation schemes without qualification.

In addition to the previous prediction fields, add these fields for diagnostic runs. The example below is synthetic, using only three candidates for brevity; real catalogs must include every candidate. Preserve original order in the lists supplied to the controller; include actual rendered `controller_context` (including titles and ordering) for trustworthy human labeling. Record `retrieved_evidence_ids` *before* accumulation/pruning and `accumulated_evidence_ids` *after* accumulation/pruning. `selected_evidence_ids` must match the last accumulated set. If an extension performs a final extra prune, log it separately or adjust this trace contract rather than silently changing the final IDs. `predicted_support_idxs` must equal unique final source paragraph IDs. `predicted_answerable` refers to the original input task; do not conflate it with the controller's judgment on the compressed evidence.

```json
{
  "id": "example-id",
  "predicted_answer": "Paris",
  "predicted_support_idxs": [3, 7],
  "predicted_answerable": true,
  "iterations": 2,
  "stop_reason": "answerable",
  "segment_catalog": [
    {"id": "p3_s0", "paragraph_idx": 3, "sentence_idx": 0, "text": "Book X was written by Alice."},
    {"id": "p7_s0", "paragraph_idx": 7, "sentence_idx": 0, "text": "Alice was born in Paris."},
    {"id": "p7_s1", "paragraph_idx": 7, "sentence_idx": 1, "text": "She moved in 2000."}
  ],
  "selected_evidence_ids": ["p3_s0", "p7_s0"],
  "iteration_trace": [
    {
      "iteration": 1,
      "query": "Where was the author of Book X born?",
      "retrieved_evidence_ids": ["p3_s0"],
      "accumulated_evidence_ids": ["p3_s0"],
      "controller_context": "Book X was written by Alice.",
      "answerable": false,
      "follow_up_question": "Where was Alice born?"
    },
    {
      "iteration": 2,
      "query": "Where was Alice born?",
      "retrieved_evidence_ids": ["p7_s0"],
      "accumulated_evidence_ids": ["p3_s0", "p7_s0"],
      "controller_context": "Book X was written by Alice. Alice was born in Paris.",
      "answerable": true,
      "follow_up_question": null,
      "ranking_complete": true,
      "ranked_evidence": [
        {"id": "p7_s0", "semantic": 0.8, "lexical": 0.5, "dual": 0.68},
        {"id": "p3_s0", "semantic": 0.4, "lexical": 0.2, "dual": 0.32},
        {"id": "p7_s1", "semantic": 0.2, "lexical": 0.1, "dual": 0.16}
      ]
    }
  ]
}
```

Optional iteration fields: `context_tokens`, `controller_input_tokens`, `controller_output_tokens`, `controller_seconds`, `filter_seconds`. Top-level `context_token_budget` is optional, to evaluate budget violations (not part of the original algorithm). Use `stop_reason` values such as `answerable`, `max_iterations`, `no_progress`, `error`. When the model fails to return a valid controller decision, record failure separately rather than fabricating a trace; the main evaluator still requires one prediction per sample. Raw Documents and Oracle can omit the entire trace. The optional ranking must be an actual descending dual-score ranking of all candidates (complete), or an actual top-prefix (not an arbitrary subset). Full rankings enable score distribution analysis; top prefixes enable Recall@K only when at least K entries exist. The catalog and full ranking are larger; store full rankings initially on a diagnostic subset if necessary. Do not subset predictions alone: gold and prediction IDs must match. All IDs and rank/score data are telemetry, not gold annotations.

## 2. Evaluate and diagnose

```sh
python benchmark.py evaluate --gold data/benchmark/evaluation/gold.jsonl --predictions results/itercomp/predictions.jsonl --out results/itercomp/scored
python diagnostics.py diagnose --gold data/benchmark/evaluation/gold.jsonl --predictions results/itercomp/predictions.jsonl --out results/itercomp/diagnostics
```

Additional files:
- `diagnostic_summary.json`: per-sample, hop and heuristic decomposition-structure summaries; iteration-conditioned means and counts; exact-match failure buckets; stop reasons.
- `per_iteration.csv`: cumulative and current-retrieval paragraph metrics, segment novelty/removal, normalized exact repeated-query flags, ranking Recall@K, full-pool score statistics, optional telemetry, and judgment proxies.
- `diagnostic_per_sample.csv`: coverage buckets, repeated-query/no-new rates, peak context tokens if fully logged, final controller judgment, optional budget violation.
- `annotation_queue.jsonl`: selected segments and exact controller context if provided, for human sufficiency assessment.

None of the paragraph coverage metrics establishes factual entailment. A wrong answer despite all gold paragraphs being represented is NOT automatically a reader failure. A stop with incomplete paragraph coverage is NOT automatically a false positive. Later iteration aggregates include only examples reaching those iterations; compare matched cohorts for causal claims. Exact repeated queries are only one proxy for stalling. `structure` is heuristic based on #n references, not validated gold topology. Inspect real dataset format before using it for conclusions.

## 3. Human annotation of actual controller states

Inspect exact information presented to the controller, not the original full paragraphs. Label whether the original question can be logically answered from that information alone; use null for ambiguous cases. Hide controller decision and gold answer during initial annotation when possible. After labeling, check disagreements with gold annotations and adjudicate. To score follow-up usefulness, independently inspect the generated question in `per_iteration.csv` against missing facts, rather than simply matching its wording. Store optional labels as JSONL:

```jsonl
{"id":"example-id","iteration":1,"sufficient":false,"follow_up_useful":true}
{"id":"example-id","iteration":2,"sufficient":true,"follow_up_useful":null}
```

```sh
python diagnostics.py diagnose --gold data/benchmark/evaluation/gold.jsonl --predictions results/itercomp/predictions.jsonl --annotations annotations.jsonl --out results/itercomp/diagnostics
```

Judgment FPR and FNR are averaged only over explicitly labeled insufficient and sufficient states, respectively. Counts show denominators; unknown labels are excluded. For a representative human-labeled subset, sample by hop and stop reason; report the sampling design. The annotation queue is not a model input.

## 4. Controlled controller probes

```sh
python diagnostics.py prepare-probes --gold data/benchmark/calibration/gold.jsonl --out data/probes/calibration --per-hop 20
```

Generates four matched conditions per question: raw full, gold only, raw with one gold paragraph removed, gold only with one gold paragraph removed. Also produces six paired synthetic coreference tests. Default with 20 samples per hop = 240 MuSiQue probes + 12 synthetic probes. Sample source calibration or a separately predeclared held-out evaluation subset; never tune on the final probe results. Probe inputs contain only original question and available paragraphs, not expected labels, variants, gold IDs, hop or source ID. IDs are shuffled and randomized by a fixed seed. Gold annotation use in probe construction is intentional and isolated from ordinary compression.

Run the controller directly on the provided question and context, with no additional retrieval. For each probe, output `{"id":"probe_000000","predicted_answerable":true}`. Optionally include `predicted_answer` for a reader evaluation on sufficient contexts. Then:

```sh
python diagnostics.py score-probes --gold data/probes/calibration/gold.jsonl --predictions results/probe_predictions.jsonl --out results/probes/scored
```

Full contexts are labeled by dataset assumption. Drop-one labels are **provisional** because alternative evidence might remain. Synthetic fragment labels test strict grounding rather than likely guesses. Audit them for ambiguity. To record verified labels, make `verified.jsonl` with `{"id":"probe_000000","expected_answerable":false}` and pass `--verified verified.jsonl`. Results are separated by label provenance and variant, including paired results; never treat provisional labels as established negative ground truth.

## 5. Compare methods

```sh
python diagnostics.py compare --left results/raw/scored/per_sample.csv --right results/itercomp/scored/per_sample.csv --out results/comparison.json
```

Paired question-level bootstrap 95% intervals and score differences (right minus left). Same sample IDs required. Missing metric pairs excluded and denominators reported. Keep consistent reader/tokenizer/prompts/generation settings/hardware, and record seeds and model versions. Report QA at comparable context budgets and separately report all LLM input/output tokens and measured wall time. No monetary cost estimate is generated without model-specific prices or hardware costs. The main evaluator's official predictions are still for the original MuSiQue-Ans split, not probes.

## 6. Test

```sh
python -m unittest -v
```

The tests use synthetic data (not actual MuSiQue or model runs). The package does not verify local metrics against the official evaluator, establish sentence-level ground truth, or automatically judge semantic query drift/coreference failure. Those require real-data runs and human labels. Keep the fixed reproduction protocol and original hyperparameters; only change a component in separately named experiments.

---

# Original IterCOMP implementation and cloud execution

The implementation follows the original algorithm: independent sentences, frozen BGE-M3 dense/sparse scores, `0.6 * semantic + 0.4 * lexical`, the global inclusive 90th percentile, monotonic evidence accumulation, the Appendix A controller prompt, and at most five filtering passes. The controller always judges the original question. No proposed architectural improvements are included. See [architecture and compatibility audit](docs/architecture.md) and [reproduction assumptions and measurement conventions](docs/reproduction_notes.md).

**Local validation uses only synthetic fixtures and mocks.** Do not download weights, run pretrained inference on the local CPU, or treat mock answers as experimental results. Actual model inference is a manual Kaggle/Colab action. Existing splits and prior result files are preserved.

## Install and run local software tests

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest -v
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest -v
```

spaCy uses its blank English tokenizer plus sentencizer; no pretrained sentence model download is needed. GPU libraries are imported lazily. The original `test_benchmark.py` and the new `tests/` suite run with CPU-only requirements.

An explicit synthetic staged demonstration (use fresh output directories):

```bash
python scripts/run_experiment.py --stage compress --config configs/mock_synthetic.yaml --inputs tests/fixtures/inputs.jsonl --output-dir mock_runs/compression
python scripts/run_experiment.py --stage read --config configs/mock_synthetic.yaml --inputs tests/fixtures/inputs.jsonl --compression-results mock_runs/compression/compression_results.jsonl --output-dir mock_runs/reader
python scripts/evaluate_run.py --gold tests/fixtures/gold.jsonl --predictions mock_runs/reader/predictions.jsonl --out mock_runs/reports
```

The fixture is a synthetic two-hop QA example. Mock token counts are exact UTF-8 byte counts for an explicitly synthetic tokenizer, not measurements from a pretrained tokenizer.

## Manual Kaggle / Colab setup

Open [Kaggle notebook](notebooks/run_itercomp_kaggle.ipynb) or [Colab notebook](notebooks/run_itercomp_colab.ipynb). They are thin repository clients with no executed cells. Configure repository/dataset paths, choose a GPU, authenticate with `HF_TOKEN` from notebook secrets, and verify checkpoint access. Kaggle supports `/kaggle/input` checkpoints and datasets, `/kaggle/working` outputs, and an explicit offline mode. Colab supports optional Drive outputs and a separate temporary model cache. Preserve outputs through Drive, a saved Kaggle version, or an exported archive.

Intended GPU environment: Linux/Python 3.10–3.12, existing CUDA PyTorch 2.6/2.7, T4/NF4/FP16. Install only on the cloud runtime:

```bash
python -m pip install -r requirements-gpu.txt
python -m pip install -e . --no-deps
```

The GPU requirements intentionally do not reinstall PyTorch. The pinned inference stack is **not GPU-verified**; complete the notebook's checks before a real run. Llama may require accepting model terms with an authorized Hugging Face account. Never put credentials into configuration files or source control.

## Profiles and overrides

Names are reader followed by controller:

| Profile | Reader | Controller |
| --- | --- | --- |
| `configs/itercomp_llama_mistral.yaml` (default) | Llama-3-8B-Instruct | Mistral-7B-Instruct-v0.3 |
| `configs/itercomp_qwen_mistral.yaml` | Qwen3-4B-Instruct-2507 | Mistral-7B-Instruct-v0.3 |
| `configs/itercomp_llama_qwen.yaml` | Llama-3-8B-Instruct | Qwen3-4B-Instruct-2507 |
| `configs/itercomp_qwen_qwen.yaml` | Qwen3-4B-Instruct-2507 | Qwen3-4B-Instruct-2507 |

The default instruction variant, controller checkpoint, quantization and generation limits are documented choices. Qwen is an alternate model configuration; it does not change the algorithm. `itercomp_original.yaml` aliases the default. Pin `revision` and `tokenizer_revision` in research runs. Paths to complete attached checkpoints can replace model IDs with `local_files_only: true`. The BGE sparse projection files must be present.

CLI overrides include `--embedding-device`, `--controller-device`, `--reader-device`, the corresponding `--*-model` options, `--seed`, and `--trace-verbosity full|compact`. All other fields are configurable in YAML; unknown fields fail validation. For dual T4s, explicitly place BGE on `cuda:1` and the controller on `cuda:0`. Two GPUs do not form a shared memory pool.

The Transformers backend supports NF4/double quantization, INT8, and nonquantized FP16/BF16/FP32. Use FP16 compute on T4. CPU loading is implemented only as an explicit FP32/unquantized configuration; **do not run local pretrained CPU inference for this task**. There is no automatic CPU or precision fallback after OOM.

For an OpenAI-compatible serving endpoint, set `backend: openai_compatible`, `api_base`, `api_key_env`, and the served `model_id`. The API key comes from that environment variable. Provider-reported token usage is used; missing usage remains null. Set `tokenizer_id` to the exact serving tokenizer if context counts are required. Remote servers must disable silent prompt truncation. `api_send_seed` is optional because endpoint arguments vary. No OpenAI SDK is required.

## Cloud-only smoke, calibration and evaluation runs

For the six reader calibration conditions, use [run_reader_calibration_6_conditions_kaggle.ipynb](notebooks/run_reader_calibration_6_conditions_kaggle.ipynb). Attach this repository directory (including the prepared calibration inputs/gold) and `itercomp_runs_export_calibration.zip` (or its automatically extracted Kaggle dataset folder), enable a GPU and the approved account's `HF_TOKEN` secret, and run the cells in order. It verifies and reuses the 150 saved IterCOMP contexts, constructs Raw/Oracle contexts on CPU, and runs both reader prompts for each context (900 reader calls). It pins the archived reader revision, resumes interrupted conditions, and exports per-condition reports plus paired comparisons. It never reruns BGE or the controller.

Start manually with a small real smoke subset on a GPU. Compression loads BGE-M3 plus the controller, then releases them:

```bash
python scripts/run_experiment.py --stage compress --method itercomp --config configs/itercomp_llama_mistral.yaml --split smoke --limit 2 --output-dir results/smoke_mistral_compression_2
python scripts/run_experiment.py --stage read --config configs/itercomp_llama_mistral.yaml --split smoke --compression-results results/smoke_mistral_compression_2/compression_results.jsonl --output-dir results/smoke_llama_mistral_2
python scripts/evaluate_run.py --gold data/benchmark/smoke/gold.jsonl --predictions results/smoke_llama_mistral_2/predictions.jsonl --out results/smoke_llama_mistral_2/reports --allow-partial
```

Reading loads only the reader. The same compression can be reused without modification by another reader when compression settings match:

```bash
python scripts/run_experiment.py --stage read --config configs/itercomp_qwen_mistral.yaml --split smoke --compression-results results/smoke_mistral_compression_2/compression_results.jsonl --output-dir results/smoke_qwen_mistral_2
```

After cloud smoke validation, select the predefined calibration split in a separate run. This command is provided for the user to execute; it was not run during implementation:

```bash
python scripts/run_experiment.py --stage both --method itercomp --config configs/itercomp_llama_mistral.yaml --split calibration --output-dir results/calibration_llama_mistral
python scripts/evaluate_run.py --gold data/benchmark/calibration/gold.jsonl --predictions results/calibration_llama_mistral/predictions.jsonl --out results/calibration_llama_mistral/reports
```

Use `--data-dir PATH` for an attached prepared benchmark directory, or `--inputs PATH` for an explicit input-only file. Final evaluation uses `--split evaluation`, a fresh directory, and the existing evaluation gold only at scoring time. No full evaluation is launched automatically. For sufficient hardware, `--stage both` runs the stages sequentially, still releasing compression models first.

## Raw Documents and isolated Oracle

Both use the same reader prompt, tokenizer, and generation settings as IterCOMP. Raw uses all original paragraph IDs; Oracle uses supporting paragraph labels through a separate loader, without passing answers/decompositions to the reader:

```bash
python scripts/run_experiment.py --stage both --method raw --config configs/raw_documents.yaml --split smoke --output-dir results/raw_smoke
python scripts/run_experiment.py --stage both --method oracle --config configs/oracle.yaml --split smoke --oracle-gold data/benchmark/smoke/gold.jsonl --output-dir results/oracle_smoke
```

Gold paths are rejected for ordinary Raw/IterCOMP compression. Inference inputs containing extra gold fields are rejected. Sample IDs identify records but never enter prompts or retrieval decisions. Oracle is a diagnostic gold-paragraph reference, not a guaranteed performance bound.

## Outputs, interruption recovery and comparison

Compression directories contain `compression_results.jsonl`, `resolved_config.yaml`, `run_metadata.json`, per-session logs under `logs/`, and `errors.jsonl` when failures occur. Reader directories contain `predictions.jsonl` and their own config/metadata/error files. Combined mode stores compression in its `compression/` subdirectory. Predictions preserve the full catalog, actual controller contexts/outputs, scores, thresholds, selected IDs, final context and raw reader response. Full rankings are on by default and can be large; compact traces omit ranking-based diagnostics.

Run metadata records configuration, data/prompt/source hashes, dependency versions, initialization/loading time, memory snapshots, serialization cost and coverage state. End-to-end algorithm time is the sum of actual compression and reader times, never the elapsed gap between notebook sessions. Reader changes recalculate context counts with that tokenizer. Optional `reference_tokenizer.model_id` and `revision` add fixed-reference counts for cross-reader comparisons.

Resume either stage with its identical original command plus `--resume`. Completed IDs, including logged errors, are skipped. Interrupted final JSONL bytes are quarantined and repaired; duplicates/interior corruption fail. Changed data/configuration/subsets require fresh directories. `--sample-ids ids.json` accepts a JSON array or a newline-delimited ID file and retains dataset order. Limited runs are explicitly partial. To retry failures, create a new selected run; do not overwrite old predictions.

Merge only completed disjoint shards:

```bash
python scripts/merge_results.py results/shard1/compression_results.jsonl results/shard2/compression_results.jsonl --output-dir results/merged_compression
```

The same command supports `predictions.jsonl` shards. It rejects overlapping IDs or inconsistent settings. Shard sample-ID sets, hashes, and split coverage remain explicit.

CPU scoring still works with the existing `benchmark.py evaluate` and `diagnostics.py diagnose` commands above. The evaluation wrapper adds partial-run matching and refuses to overwrite old reports. It emits `reports/scored/{summary.json,per_sample.csv,official_predictions.jsonl}` and `reports/diagnostics/{diagnostic_summary.json,diagnostic_per_sample.csv,per_iteration.csv,annotation_queue.jsonl}`. With `--allow-partial` it also saves `gold_subset.jsonl`; this is a matching report input, not a regenerated dataset split. Pass that subset to the official evaluator for partial runs. Compare complete compatible results using the existing `diagnostics.py compare` command.

## Troubleshooting

* **Gated model / authentication:** accept the checkpoint's access terms and set the notebook `HF_TOKEN` secret. Metadata checks alone do not establish gated weight access.
* **No CUDA or unsupported BF16:** enable a GPU and choose explicit devices; use FP16 for T4. CPU fallback is disabled.
* **GPU OOM:** use independent stages, explicit dual-GPU placement, or explicitly choose a quantization/model profile. No model precision or evidence behavior changes automatically.
* **Context limit / embedding truncation:** inspect the saved error. Prompts are never silently shortened. BGE defaults to failing overlength inputs; explicitly choosing `truncation_policy: log` records every truncation and must be disclosed.
* **Malformed controller JSON:** raw outputs and parse errors are preserved, with no fabricated trace. Default retries are zero; bounded retries must be configured and are fully counted.
* **Answerable reply with a follow-up string:** default `itercomp.answerable_follow_up_policy: normalize_unused` retains the explicit answerable label, normalizes its unused follow-up to null, and records a formatting warning. The raw reply remains in the trace and no extra call is made. Use `strict` for the earlier null-only behavior. This does not establish evidence sufficiency; inspect flagged controller states separately. Malformed JSON, invalid types/labels, and absent required questions still fail. See [smoke error analysis and rerun commands](docs/smoke_error_analysis.md).
* **Artifact mismatch / changed subset:** select the matching compression profile, original input-only file and subset, or use a fresh run. Reader changes alone do not invalidate compression.
* **Notebook reset:** recreate paths/config/secrets, restore persisted outputs, and resume the interrupted stage. `/content` and unsaved Kaggle outputs may disappear.

### Kaggle setup failures: HF_TOKEN and Protobuf

The message about an unavailable `HF_TOKEN` secret is an authentication warning, independent of package import errors. Create a Hugging Face [read token](https://huggingface.co/settings/tokens), add it in Kaggle **Add-ons > Secrets** with the name `HF_TOKEN`, and enable this notebook's access. Request access to the [Llama reader checkpoint](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) using the same account. A token does not automatically grant gated-model approval. The authentication cell now reuses an existing login, verifies available credentials online, and gives specific missing-secret guidance without printing credentials. See [Hugging Face gated-model documentation](https://huggingface.co/docs/hub/models-gated).

If an older notebook raises `cannot import name 'configure_hf_auth'`, the working repository or cached module predates that helper. The current authentication cells use the Hugging Face and notebook secrets APIs directly, so they work independently of the local package version. Refresh the working repository separately to obtain dependency and inference fixes; restarting a kernel alone cannot update an old file on disk.

If NF4 loading reports a missing `libbitsandbytes_cuda130.so`, followed by `libbitsandbytes_cpu.so: undefined symbol: cquantize_blockwise_fp16_nf4`, the native CUDA quantization backend was not loaded. This is not a missing checkpoint or HF permission error. In Kaggle run `%pip install --no-cache-dir --force-reinstall --no-deps bitsandbytes==0.49.2`, restart the kernel, and check the native backend in a fresh process before retrying inference. The notebook installation cell now performs this check; the runner also checks before encoder/model downloads. Keep `bitsandbytes==0.49.2` in the **working** requirements file so an older copy cannot reinstall the incompatible build. `--no-deps` leaves platform PyTorch unchanged. Do not switch to CPU or unquantized FP16 to bypass the error; that changes the intended resource setup. Checkpoint downloads already cached on disk can be reused.

An older requirements file constrained Protobuf below version 6. In the reported Kaggle image this downgraded `7.36.2` to `5.29.6`, breaking preinstalled TensorFlow (`gencode 6.31.1 runtime 5.29.6`). The current requirements remove that constraint, and the package/notebooks explicitly select PyTorch-only Transformers imports with `USE_TORCH=1`, `USE_TF=0`, `USE_FLAX=0`. TensorFlow/Flax are not used by IterCOMP. The bitsandbytes pin is now `0.49.2`, which [documents CUDA 13 builds](https://huggingface.co/docs/bitsandbytes/v0.49.2/installation) for the CUDA 13 runtime shown in the log; this does not establish full-stack GPU verification.

For that already-modified Kaggle environment, run this repair cell once (cloud only):

```python
%pip install --upgrade "protobuf==7.36.2" "bitsandbytes==0.49.2"
```

Then restart the kernel and use the updated repository **as well as** the updated notebook. Do not rerun the old requirements file: it would downgrade Protobuf again. A fresh runtime with the updated requirements preserves its original platform Protobuf. The existing repository setup cell copies only when `REPO` does not exist; updating an attached dataset does not refresh an old working copy. To copy the updated attached repository without overwriting anything, use a new path such as `REPO = Path('/kaggle/working/musique_itercomp_fixed')`. Set a fresh `OUTPUT_ROOT`, such as `/kaggle/working/itercomp_runs_fixed`, because the failed run's metadata refers to older source code. Recreate the remaining variables by running cells from the top.

Real GPU model loading, memory use, dense/sparse adapter outputs, dependency compatibility and real smoke behavior still need manual cloud validation. No empirical MuSiQue performance is claimed by mock tests.
