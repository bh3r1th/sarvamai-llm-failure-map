"""Build report-oriented tables from evaluation outputs."""

from __future__ import annotations

from typing import Any

from code_switch_failure_map.analysis.prompt_sensitivity import compare_prompt_sensitivity
from code_switch_failure_map.analysis.slices import analyze_slice_breakdowns
from code_switch_failure_map.eval.aggregate import summarize_aggregates
from code_switch_failure_map.analysis.token_impact import build_token_summary
from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.prediction import PredictionRecord


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def build_model_comparison_table(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    groups: dict[str, list[EvaluationResult]] = {}
    for result in results:
        groups.setdefault(result.model_name, []).append(result)

    rows: list[dict[str, Any]] = []
    for model_name, model_rows in sorted(groups.items()):
        rows.append(
            {
                "model_name": model_name,
                "samples": len(model_rows),
                "intent_accuracy": _mean([1.0 if row.intent_correct else 0.0 for row in model_rows]),
                "entity_f1": _mean([row.entity_f1 for row in model_rows]),
                "exact_match_rate": _mean([1.0 if row.overall_exact_match else 0.0 for row in model_rows]),
                "schema_failure_rate": _mean([1.0 if row.schema_failure else 0.0 for row in model_rows]),
            }
        )
    return rows


def build_prompt_comparison_table(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    return compare_prompt_sensitivity(results)["overall"]


def build_slice_breakdown_table(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    return analyze_slice_breakdowns(results)


def build_failure_bucket_table(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    bucket_counts: dict[str, int] = {}
    for result in results:
        for bucket in result.assigned_failure_buckets:
            bucket_counts[bucket.value] = bucket_counts.get(bucket.value, 0) + 1
    total = len(results)
    return [
        {
            "failure_bucket": bucket,
            "samples": count,
            "rate": 0.0 if total == 0 else count / total,
        }
        for bucket, count in sorted(bucket_counts.items())
    ]


def build_token_summary_table(results: list[EvaluationResult], predictions: list[PredictionRecord]) -> list[dict[str, Any]]:
    return build_token_summary(results, predictions)


def build_report_tables(results: list[EvaluationResult], predictions: list[PredictionRecord], exemplar_index: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Build the required analysis tables for exports."""
    aggregate_summary = summarize_aggregates(results)
    return {
        "model_comparison": build_model_comparison_table(results),
        "prompt_comparison": build_prompt_comparison_table(results),
        "slice_breakdown": build_slice_breakdown_table(results),
        "language_family_summary": aggregate_summary["by_language_family"],
        "failure_bucket_counts": build_failure_bucket_table(results),
        "boundary_failure_summary": aggregate_summary["boundary_failure_summary"],
        "boundary_failure_by_model": aggregate_summary["boundary_failure_by_model"],
        "boundary_failure_by_model_prompt": aggregate_summary["boundary_failure_by_model_prompt"],
        "boundary_failure_by_language_family": aggregate_summary["boundary_failure_by_language_family"],
        "boundary_failure_by_slice_tag": aggregate_summary["boundary_failure_by_slice_tag"],
        "token_summary": build_token_summary_table(results, predictions),
        "exemplar_index": exemplar_index,
    }
