# Manual Smoke Summary

## 1. Smoke run status: not ready

No smoke-run prediction artifacts were present under `data/predictions/`, and no evaluator outputs were present under `outputs/metrics/`. Manual inspection of model behavior, prompt contract compliance, parsing stability, and failure-bucket quality is therefore blocked.

## 2. Blockers found

- No prediction JSONL files exist for any of the 4 expected model/prompt pairs.
- No per-sample evaluation outputs exist for the smoke run.
- Prompt contract health cannot be judged because there are no raw responses to inspect.
- Parse/schema stability cannot be judged because there are no model outputs to parse.
- Failure bucket quality cannot be judged on real outputs because no evaluated rows exist.

## 3. Warnings found

- `raw_035`: intent boundary may blur between `purchase_request` and `task_create` because of `add kr`.
- `raw_036`: gold entity type `other` is broad and may reduce interpretability.
- `raw_094`: conditional duration `30` may create unstable extraction expectations.
- `raw_096`: intentionally underspecified dual-time request may produce noisy metrics if overused.
- `raw_100`: name/reference distinction is annotation-sensitive.

## 4. Gold dataset issues

- Most smoke samples are defensible and useful.
- The samples listed in Warnings should be kept only if the team explicitly wants ambiguity/adversarial pressure in the smoke gate.
- No sample looked clearly mislabeled enough to force immediate label-space redesign, but a few need annotation-policy clarification before scaling.

## 5. Prompt contract issues

- Not inspectable yet because no raw responses exist.
- This is itself a blocker because the smoke run was meant to catch non-JSON and extra-prose behavior before a full run.

## 6. Schema issues

- Not inspectable yet because no prediction artifacts exist.
- The absence of smoke outputs means parse/schema reliability is still unproven.

## 7. Failure taxonomy issues

- Not inspectable on real predictions yet.
- High-risk samples to verify first once outputs exist: `raw_083`, `raw_094`, `raw_096`, `raw_100`.

## 8. Recommended fixes before full run

### prompt fixes
- Execute the actual smoke run and inspect whether all 4 model/prompt pairs return strict JSON only.
- If any pair emits prose or code fences, tighten prompt instructions before scaling.

### schema fixes
- Confirm `parse_success` and `schema_failure` remain stable across all 20 samples before full evaluation.
- Reject any model/prompt pair that repeatedly emits unsupported labels or malformed entity objects.

### dataset fixes
- Clarify annotation policy for `raw_035`, `raw_094`, `raw_096`, and `raw_100` before treating their failures as model evidence.
- Consider replacing any sample whose ambiguity is useful for adversarial testing but too unstable for smoke-gate readiness.

### evaluator/failure-rule fixes
- Re-run the manual audit after real outputs exist and verify failure buckets on the high-risk samples first.
- Do not trust bucket counts for blog claims until at least the smoke run has been manually validated.

## Recommendation

**FIX FIRST**
