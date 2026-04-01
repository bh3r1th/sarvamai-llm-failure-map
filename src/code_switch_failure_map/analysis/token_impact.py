"""Descriptive token-count analysis for evaluation outcomes."""

from __future__ import annotations

from typing import Any

from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.prediction import PredictionRecord


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _bucket_label(total_tokens: int | None) -> str:
    if total_tokens is None:
        return "unknown"
    if total_tokens < 64:
        return "lt_64"
    if total_tokens < 128:
        return "64_127"
    if total_tokens < 256:
        return "128_255"
    return "ge_256"


def build_token_summary(
    evaluation_results: list[EvaluationResult],
    predictions: list[PredictionRecord],
) -> list[dict[str, Any]]:
    """Summarize outcome rates by token-count buckets."""
    prediction_by_key = {
        (prediction.sample_id, prediction.model_name, prediction.prompt_language.value): prediction for prediction in predictions
    }
    grouped: dict[tuple[str, str, str], list[tuple[EvaluationResult, PredictionRecord | None]]] = {}

    for result in evaluation_results:
        key = (result.sample_id, result.model_name, result.prompt_language.value)
        bucket = _bucket_label(prediction_by_key.get(key).total_tokens if key in prediction_by_key else None)
        grouped.setdefault((result.model_name, result.prompt_language.value, bucket), []).append((result, prediction_by_key.get(key)))

    summary: list[dict[str, Any]] = []
    for (model_name, prompt_language, token_bucket), rows in sorted(grouped.items()):
        totals = [float(pred.total_tokens) for _, pred in rows if pred is not None and pred.total_tokens is not None]
        summary.append(
            {
                "model_name": model_name,
                "prompt_language": prompt_language,
                "token_bucket": token_bucket,
                "samples": len(rows),
                "avg_total_tokens": _mean(totals),
                "parse_failure_rate": _mean(
                    [
                        1.0
                        if pred is None or pred.parsed_prediction is None or not pred.parse_success
                        else 0.0
                        for _, pred in rows
                    ]
                ),
                "intent_error_rate": _mean([0.0 if result.intent_correct else 1.0 for result, _ in rows]),
                "entity_error_rate": _mean([0.0 if result.entities_exact_match else 1.0 for result, _ in rows]),
                "schema_failure_rate": _mean([1.0 if result.schema_failure else 0.0 for result, _ in rows]),
            }
        )
    return summary
