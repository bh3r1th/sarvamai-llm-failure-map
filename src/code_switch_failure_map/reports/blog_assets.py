"""Compact assets to support blog writing from analysis outputs."""

from __future__ import annotations

from typing import Any

from code_switch_failure_map.analysis.prompt_sensitivity import compare_prompt_sensitivity
from code_switch_failure_map.schemas.evaluation import EvaluationResult


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_blog_assets(
    *,
    results: list[EvaluationResult],
    tables: dict[str, list[dict[str, Any]]],
    exemplars: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build concise JSON/text-friendly assets for blog drafting."""
    findings: list[str] = []

    model_rows = tables.get("model_comparison", [])
    if model_rows:
        best_model = max(model_rows, key=lambda row: (row["exact_match_rate"], row["entity_f1"]))
        findings.append(
            f"Best overall exact-match rate: {best_model['model_name']} at {_format_pct(best_model['exact_match_rate'])}."
        )

    failure_rows = tables.get("failure_bucket_counts", [])
    if failure_rows:
        dominant_failure = max(failure_rows, key=lambda row: row["samples"])
        findings.append(
            f"Most frequent failure bucket: {dominant_failure['failure_bucket']} affecting {dominant_failure['samples']} evaluated samples."
        )

    boundary_rows = tables.get("boundary_failure_summary", [])
    if boundary_rows:
        dominant_boundary = max(boundary_rows, key=lambda row: row["samples"])
        findings.append(
            f"Most frequent boundary failure: {dominant_boundary['boundary_failure_category']} affecting {dominant_boundary['samples']} evaluated samples."
        )

    language_family_rows = tables.get("language_family_summary", [])
    if language_family_rows:
        weakest_language_family = min(language_family_rows, key=lambda row: (row["exact_match_rate"], row["entity_f1"]))
        findings.append(
            f"Weakest language family slice: {weakest_language_family['language_family']} at {_format_pct(weakest_language_family['exact_match_rate'])} exact match."
        )

    slice_rows = [row for row in tables.get("slice_breakdown", []) if row.get("dimension") == "slice_tag"]
    if slice_rows:
        weakest_slice = min(slice_rows, key=lambda row: (row["exact_match_rate"], row["entity_f1"]))
        findings.append(
            f"Weakest slice: {weakest_slice['slice_tag']} for {weakest_slice['model_name']} / {weakest_slice['prompt_language']} at {_format_pct(weakest_slice['exact_match_rate'])} exact match."
        )

    prompt_analysis = compare_prompt_sensitivity(results)
    if prompt_analysis["biggest_regressions"]:
        regression = prompt_analysis["biggest_regressions"][0]
        findings.append(
            f"Largest prompt regression: Hinglish underperforms English on {regression['slice_tag']} for {regression['model_name']} by {_format_pct(abs(regression['delta_exact_match_rate']))} exact match."
        )
    if prompt_analysis["biggest_improvements"]:
        improvement = prompt_analysis["biggest_improvements"][0]
        findings.append(
            f"Largest prompt improvement: Hinglish outperforms English on {improvement['slice_tag']} for {improvement['model_name']} by {_format_pct(improvement['delta_exact_match_rate'])} exact match."
        )

    if not findings:
        findings.append("No blog-ready findings are available until at least one evaluated run is produced.")

    methodology_summary = [
        "Intent is scored with exact-match accuracy only.",
        "Entities use exact-match tuple comparison on normalized label/value pairs.",
        "Failure buckets are deterministic and assigned from gold-vs-prediction comparison only.",
        "Token analysis is descriptive and should not be read as causal evidence.",
    ]

    caveats = [
        "Prompt sensitivity requires both English and Hinglish runs for the same model to produce meaningful deltas.",
        "Entity exact-match scoring will undercount partially-correct extractions.",
        "Failure bucket counts reflect the current deterministic rules, not human adjudication.",
    ]
    if len(results) <= 50:
        caveats.append("Current run size is small, so individual examples can shift aggregate rates materially.")

    return {
        "top_findings_summary": findings,
        "strongest_examples": exemplars[:10],
        "notable_caveats": caveats,
        "methodology_summary": methodology_summary,
    }
