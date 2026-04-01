"""Failure exemplar mining for blog-usable examples."""

from __future__ import annotations

from typing import Any

from code_switch_failure_map.eval.aggregate import classify_boundary_failure
from code_switch_failure_map.eval.error_buckets import bucket_names
from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.prediction import PredictionRecord
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import FailureCategory


def _short_reason(result: EvaluationResult) -> str:
    if result.schema_failure:
        return "Model output broke the required JSON/schema contract."
    if FailureCategory.TEMPORAL_INVERSION in result.assigned_failure_buckets:
        return "Predicted temporal value diverges from the gold temporal reference."
    if FailureCategory.INTENT_CONFUSION in result.assigned_failure_buckets:
        return "Predicted top-level intent does not match the gold intent."
    if FailureCategory.ENTITY_DRIFT in result.assigned_failure_buckets:
        return "Entity label stayed plausible but the normalized value drifted."
    if FailureCategory.HALLUCINATION in result.assigned_failure_buckets:
        return "Prediction adds unsupported entities beyond the gold annotation."
    if FailureCategory.OMISSION in result.assigned_failure_buckets:
        return "Prediction misses at least one gold entity."
    if FailureCategory.OVER_EXTRACTION in result.assigned_failure_buckets:
        return "Prediction extracts extra entities not supported by the gold record."
    if FailureCategory.SENTIMENT_INVERSION in result.assigned_failure_buckets:
        return "Sentiment-heavy instruction degrades intent/entity handling."
    return "Prediction differs from gold under the deterministic evaluator."


def _signal_score(result: EvaluationResult, prediction: PredictionRecord) -> tuple[float, int, float]:
    return (
        float(len(result.assigned_failure_buckets)),
        prediction.total_tokens or 0,
        1.0 - result.entity_f1 + (0.5 if not result.intent_correct else 0.0) + (1.0 if result.schema_failure else 0.0),
    )


def mine_failure_exemplars(
    *,
    evaluation_results: list[EvaluationResult],
    gold_records: list[SampleRecord],
    predictions: list[PredictionRecord],
    max_per_bucket: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select strong, non-duplicate exemplars for each failure bucket."""
    gold_by_id = {record.sample_id: record for record in gold_records}
    prediction_by_key = {
        (prediction.sample_id, prediction.model_name, prediction.prompt_language.value): prediction for prediction in predictions
    }

    candidate_rows: dict[FailureCategory, list[tuple[tuple[float, int, float], dict[str, Any]]]] = {}
    for result in evaluation_results:
        key = (result.sample_id, result.model_name, result.prompt_language.value)
        prediction = prediction_by_key.get(key)
        gold = gold_by_id.get(result.sample_id)
        if prediction is None or gold is None:
            continue

        exemplar = {
            "sample_id": result.sample_id,
            "model_name": result.model_name,
            "prompt_language": result.prompt_language.value,
            "gold_text": gold.text,
            "gold_intent": result.gold_intent,
            "predicted_intent": result.predicted_intent,
            "gold_entities": [entity.model_dump(mode="json") for entity in gold.gold_entities],
            "predicted_entities": (
                []
                if prediction.parsed_prediction is None
                else [entity.model_dump(mode="json") for entity in prediction.parsed_prediction.entities]
            ),
            "raw_response": prediction.raw_response,
            "assigned_failure_buckets": bucket_names(result.assigned_failure_buckets),
            "boundary_failure_category": classify_boundary_failure(result),
            "reason": _short_reason(result),
            "entity_f1": result.entity_f1,
            "schema_failure": result.schema_failure,
            "notes": result.notes,
        }
        score = _signal_score(result, prediction)

        for bucket in result.assigned_failure_buckets:
            candidate_rows.setdefault(bucket, []).append((score, exemplar))

    selected: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str, str]] = set()

    for bucket in sorted(candidate_rows, key=lambda item: item.value):
        bucket_candidates = sorted(candidate_rows[bucket], key=lambda item: item[0], reverse=True)
        chosen = 0
        for _, exemplar in bucket_candidates:
            exemplar_key = (exemplar["sample_id"], exemplar["model_name"], exemplar["prompt_language"])
            if exemplar_key in used_keys:
                continue
            enriched = dict(exemplar)
            enriched["bucket"] = bucket.value
            selected.append(enriched)
            index_rows.append(
                {
                    "bucket": bucket.value,
                    "sample_id": exemplar["sample_id"],
                    "model_name": exemplar["model_name"],
                    "prompt_language": exemplar["prompt_language"],
                    "reason": exemplar["reason"],
                }
            )
            used_keys.add(exemplar_key)
            chosen += 1
            if chosen >= max_per_bucket:
                break

    return selected, index_rows
