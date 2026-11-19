# Full-smoke error analysis and targeted correction

Inspected the user-provided `itercomp_runs_export.zip` without extracting or modifying its experimental artifacts.

Source archive SHA256: `8c514b46d840b7b0b4c7a31eb964bd1b661478c690215766b6d6c6a011325a57`.

## Observed failures

The 30-sample compression run contained 5 answerable stops, 22 iteration-limit stops, and 3 errors. It saved 122 actual controller calls. The three failures occurred on the first filtering pass, all with `ValueError: An answerable decision requires a null follow-up`:

| Sample ID | Actual controller label | Actual follow-up string |
| --- | --- | --- |
| `3hop1__10879_10446_10423` | `answerable` | In which geographical portion of the country, along with Arabia and the country they migrated from, do Somalis live that has songs similar to those from Somalia? |
| `4hop1__233309_727246_158589_81091` | `answerable` | Who is the spouse of Karin Larsson? |
| `4hop3__393686_620110_61746_261712` | `answerable` | Which US state is Tuolumne County, where Jamestown and Deadwood are located, part of? |

Each response was valid JSON containing both required fields. Each also violated the prompt's null-follow-up convention for an answerable response. The previous parser treated this redundant field as fatal. The reader inherited the three error records and made 27 generation calls, skipping those three; there were no additional reader-generation failures in the supplied prediction file.

The original metrics (EM 0.1667, F1 0.2673, weighted compression ratio 0.16895) include these three empty failure predictions. They describe this small, balanced smoke subset. No corrected QA metric is inferred from the parser fix.

## Correction and research interpretation

`itercomp.answerable_follow_up_policy: normalize_unused` is now the explicit pipeline default. For a valid `answerable` label and a string follow-up, it retains the label and normalizes the unused follow-up to null. It logs the format violation, parsed decision, and raw response. The reader can then run on the accumulated context as the original answerable branch requires. Strict mode remains available for reproducing the previous validation behavior.

This does not reinterpret the response as unanswerable, infer a label from the follow-up, assert factual sufficiency, or repair arbitrary JSON. Incorrect controller sufficiency judgments remain research failures to diagnose. Non-string/non-null follow-up values, missing fields, invalid labels, duplicate fields and malformed JSON still fail. An unanswerable decision still requires a nonempty follow-up.

No sentence decomposition, title handling, embeddings, relevance equations, percentile, evidence accumulation, stopping rule, generation prompt, model or generation limit changed. Normalization adds no LLM calls. Existing raw usage and latency remain available in new traces, together with `format_warnings`, `parsed_judgment`, per-iteration `controller_format_warnings`, and the sample warning count.

## Validation

A read-only parser audit of all **122 saved controller responses** found:

* **119** decisions identical under the previous strict and new normalization policies.
* **3** formerly rejected replies accepted with an explicit formatting warning, matching exactly the three failed sample IDs.
* No archived prediction, artifact, source response or measured value modified.

Regression fixtures contain the three captured replies and archive identity, not dataset answers or gold evidence. Unit tests exercise these replies as text; integration tests use synthetic QA and mock models. Tests check continued strict rejection, warning/raw-output/usage preservation, no extra calls, reader execution, reader abstention independent of controller judgment, and compatibility with the unchanged benchmark and diagnostics.

The audit reparses saved text only. It does not generate answers or establish new empirical performance. Real-model revalidation remains a manual cloud step.

## Cloud rerun

Update the Kaggle working repository with the corrected package and configs. The original exported run remains a useful record of the prior policy. Use new output directories: `--resume` skips completed failed IDs and configuration/source identities changed. Do not merge the three corrected samples into the old 30-row run; produce a coherent new full-smoke run after the small check.

With the notebook's existing `base`, `INPUTS`, `GOLD`, `OUTPUT_ROOT` and other setup variables, first test just the affected IDs on the GPU:

```python
fixed_root = OUTPUT_ROOT.parent / "itercomp_runs_followup_fix"
failed_out = fixed_root / "failed_three"
subprocess.run(base + [
    "--stage", "both", "--method", "itercomp",
    "--sample-ids", "tests/fixtures/failed_smoke_ids.json",
    "--output-dir", str(failed_out),
], check=True)
```

The saved resolved config may omit the new policy; the corrected loader supplies the explicit `normalize_unused` default. No notebook-only authentication helper update substitutes for updating the package used by experiment subprocesses.

Inspect each new prediction's `error`, `controller_calls`, `controller_format_warning_count`, and `reader_llm_calls`. Then run the full 30-sample smoke split under the same corrected implementation:

```python
full_out = fixed_root / "smoke30"
subprocess.run(base + [
    "--stage", "both", "--method", "itercomp",
    "--output-dir", str(full_out),
], check=True)
subprocess.run([
    sys.executable, "scripts/evaluate_run.py",
    "--gold", str(GOLD),
    "--predictions", str(full_out / "predictions.jsonl"),
    "--out", str(full_out / "reports"),
], check=True)
```

These commands assume `SPLIT='smoke'` and `base` has no `--limit`/`--sample-ids`, matching the provided full-smoke notebook. To score only the three-case check, use `scripts/evaluate_run.py --allow-partial` so it creates a matching gold subset. Preserve the original archive and export the corrected run separately. The normalized controller label alone is not a QA correctness result.
