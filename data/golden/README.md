# Golden Data Artifacts

This folder stores deterministic golden-set artifacts built from `data/raw/seed_hinglish_samples.jsonl`.

## Files

- `golden_candidates.jsonl`: top-ranked hard examples considered for manual review.
- `golden_set_v1.jsonl`: final 50-sample golden set for v1.

## How files are generated

Run from repo root:

```bash
python - <<'PY'
from code_switch_failure_map.data.curate import build_golden_files
build_golden_files()
PY
```

The pipeline is deterministic and does not use model calls or unseeded randomness.

## Manual review workflow

For each selected sample reviewers should:

1. **Verify intent label** against utterance semantics.
2. **Verify entity extraction** (type, text span, normalized value if present).
3. **Verify slice tags** match observed phenomena (ambiguity, transliteration noise, temporal, adversarial, etc.).
4. **Confirm why sample is difficult** in one sentence for failure-analysis traceability.
5. **Add annotation notes** when edge cases or tie-break decisions were needed.

A sample should be revised or removed if confidence is low, if it is a trivial duplicate, or if no defensible ground truth exists.
