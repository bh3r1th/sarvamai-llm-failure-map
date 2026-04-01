# Blog Outline

## Working Title

Code-Switch Failure Map: where extraction pipelines break under Hinglish prompts, noisy transliteration, and adversarial instructions

## Results Structure

1. Setup and methodology
   - Dataset scope
   - Models compared
   - English vs Hinglish prompt variants
   - Exact-match evaluator and deterministic failure buckets
2. Headline comparison
   - Model comparison table
   - Prompt comparison table
   - One short note on schema stability
3. Slice breakdown
   - Code-switching
   - Transliteration noise
   - Ambiguity
   - Adversarial prompts
   - Intent-type breakdown
4. Prompt sensitivity
   - Overall English vs Hinglish deltas
   - Biggest regressions
   - Biggest improvements
5. Failure taxonomy in practice
   - Failure bucket counts
   - Top exemplar per bucket
6. Token-count analysis
   - Descriptive relationship between total tokens and parse / schema / intent / entity errors
7. Caveats
   - Small-sample warnings if applicable
   - Strict exact-match entity scoring
   - No causal claims from token analysis

## Expected Assets

- `outputs/comparisons/<run_id>/model_comparison.{csv,json}`
- `outputs/comparisons/<run_id>/prompt_comparison.{csv,json}`
- `outputs/comparisons/<run_id>/slice_breakdown.{csv,json}`
- `outputs/comparisons/<run_id>/failure_bucket_counts.{csv,json}`
- `outputs/comparisons/<run_id>/token_summary.{csv,json}`
- `outputs/comparisons/<run_id>/exemplar_index.{csv,json}`
- `outputs/comparisons/<run_id>/blog_assets.json`
- `outputs/comparisons/<run_id>/top_findings.txt`
- `outputs/failures/<run_id>/failure_exemplars.json`

## Writing Rule

Every claim in the blog should point back to one exported table, one mined exemplar, or one deterministic evaluator rule.
