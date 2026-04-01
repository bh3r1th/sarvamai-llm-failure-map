# Smoke Test Runbook

This smoke run is an operational gate, not a benchmark report. It exists to verify that the first fixed 20-sample batch runs cleanly, writes traceable artifacts, and does not immediately expose prompt or schema failures.

## Dataset

- Dataset file: `data/curated/smoke_20.jsonl`
- Source: fixed subset from `data/golden/golden_set_v1.jsonl`
- Scope: 20 samples covering clear, noisy, ambiguous, adversarial, and conflicting multi-clause instructions

## Execute

Preferred installed-CLI flow:

```bash
csfm smoke-run --config configs/smoke_run.yaml
csfm evaluate-outputs --run-dir data/predictions/<run_id>
```

Script-wrapper flow from repo root:

```bash
python scripts/run_models.py --config configs/smoke_run.yaml
python scripts/evaluate_outputs.py --run-dir data/predictions/<run_id>
```

Notes:

- `configs/smoke_run.yaml` is intentionally stored as JSON-compatible YAML so the repo does not need a YAML dependency yet.
- If you want deterministic reruns in the same directory, pass `--run-id smoke20_manual_a` to the runner.

## Output Layout

Each smoke run writes to one deterministic directory:

- `data/predictions/<run_id>/manifest.json`
- `data/predictions/<run_id>/health_summary.json`
- `data/predictions/<run_id>/<model>__<prompt_language>.jsonl`

Current config expects these prediction files:

- `sarvam_30b__hinglish.jsonl`
- `sarvam_30b__english.jsonl`
- `gpt_4o_mini__hinglish.jsonl`
- `gpt_4o_mini__english.jsonl`

## Inspect First

Inspect these items in order:

1. `manifest.json` to confirm dataset, models, prompt variants, and run id are what you intended.
2. `health_summary.json` to confirm there are no missing files, row-count mismatches, parse failures, or schema failures.
3. One prediction file from each prompt language to spot-check raw responses on difficult examples:
   `raw_083`, `raw_096`, `raw_100`, `raw_036`, `raw_031`.

## Blocking Failure

Any of the following should block a full experiment:

- Missing prediction file for any configured model/prompt pair
- Any prediction file with fewer or more than 20 rows
- Duplicate, missing, or unexpected `sample_id` values
- Any invalid JSONL prediction row
- Any `parse_success = false`
- Any `schema_failure = true`
- Any empty `raw_response`
- Provider/runtime failure that prevents a full artifact set from being written

## Ready For Full Run

Proceed to a larger experiment only when all of the following are true:

- The smoke run wrote all expected prediction files under one run directory.
- `health_summary.json` has zero blocking failures.
- Manual spot-checking of the hardest samples shows the response format is stable across both prompt languages.
- The output directory naming is clear enough that a later evaluator can consume it without guessing.
- The team agrees the prompt wording is not obviously under-specifying the JSON contract.
