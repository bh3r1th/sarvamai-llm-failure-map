"""Evaluation orchestration and aggregate summary helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Callable

from code_switch_failure_map.eval.entity_metrics import score_entities
from code_switch_failure_map.eval.error_buckets import bucket_names
from code_switch_failure_map.eval.failure_rules import assign_failure_buckets
from code_switch_failure_map.eval.intent_metrics import build_confusion_summary, score_intent
from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.prediction import PredictionRecord
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import SliceTag

_LANGUAGE_FAMILY_BY_TAG: dict[SliceTag, str] = {
    SliceTag.HINGLISH: "hinglish",
    SliceTag.TELUGU_ENGLISH: "telugu_english",
}
_LEGACY_LANGUAGE_FAMILY_BY_TOKEN: dict[str, str] = {
    "hi": "hinglish",
    "te": "telugu_english",
}


def evaluate_prediction_set(
    *,
    gold_records: list[SampleRecord],
    predictions: list[PredictionRecord],
    model_name: str | None = None,
    prompt_language: str | None = None,
) -> tuple[list[EvaluationResult], list[str]]:
    """Evaluate one prediction set against the gold dataset."""
    gold_by_id = {record.sample_id: record for record in gold_records}
    prediction_by_id = {prediction.sample_id: prediction for prediction in predictions}
    duplicate_ids = [sample_id for sample_id, count in Counter(pred.sample_id for pred in predictions).items() if count > 1]

    issues: list[str] = []
    if duplicate_ids:
        issues.append(f"duplicate prediction sample_ids={sorted(duplicate_ids)}")

    unexpected_ids = sorted(sample_id for sample_id in prediction_by_id if sample_id not in gold_by_id)
    if unexpected_ids:
        issues.append(f"unexpected prediction sample_ids={unexpected_ids}")

    results: list[EvaluationResult] = []
    for gold in gold_records:
        prediction = prediction_by_id.get(gold.sample_id)
        if prediction is None:
            issues.append(f"missing prediction for sample_id={gold.sample_id}")
            continue

        intent_score = score_intent(gold, prediction)
        entity_score = score_entities(gold, prediction)
        failure_buckets = assign_failure_buckets(
            sample=gold,
            prediction=prediction,
            intent_score=intent_score,
            entity_score=entity_score,
        )
        notes = prediction.error_message

        results.append(
            EvaluationResult(
                sample_id=gold.sample_id,
                model_name=model_name or prediction.model_name,
                prompt_language=prediction.prompt_language,
                gold_intent=intent_score.gold_intent,
                predicted_intent=intent_score.predicted_intent,
                intent_correct=intent_score.exact_match,
                entity_gold_count=entity_score.gold_count,
                entity_predicted_count=entity_score.predicted_count,
                entity_true_positives=entity_score.true_positives,
                entity_false_positives=entity_score.false_positives,
                entity_false_negatives=entity_score.false_negatives,
                entity_precision=entity_score.precision,
                entity_recall=entity_score.recall,
                entity_f1=entity_score.f1,
                schema_failure=prediction.schema_failure or not prediction.parse_success,
                assigned_failure_buckets=failure_buckets,
                slice_tags=gold.slice_tags,
                notes=notes,
            )
        )

    return results, issues


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _group_metric_row(rows: list[EvaluationResult], key_name: str, key_value: str) -> dict[str, Any]:
    schema_failures = sum(1 for row in rows if row.schema_failure)
    exact_matches = sum(1 for row in rows if row.overall_exact_match)
    return {
        key_name: key_value,
        "samples": len(rows),
        "intent_accuracy": _mean([1.0 if row.intent_correct else 0.0 for row in rows]),
        "entity_precision": _mean([row.entity_precision for row in rows]),
        "entity_recall": _mean([row.entity_recall for row in rows]),
        "entity_f1": _mean([row.entity_f1 for row in rows]),
        "exact_match_rate": _mean([1.0 if row.overall_exact_match else 0.0 for row in rows]),
        "schema_failure_rate": 0.0 if not rows else schema_failures / len(rows),
        "schema_failures": schema_failures,
        "exact_matches": exact_matches,
    }


def language_family_values_for_result(result: EvaluationResult) -> list[str]:
    """Return language-family values from explicit tags or legacy sample-id tokens."""
    explicit = sorted(
        _LANGUAGE_FAMILY_BY_TAG[tag]
        for tag in result.slice_tags
        if tag in _LANGUAGE_FAMILY_BY_TAG
    )
    if explicit:
        return explicit

    sample_id_tokens = {token.lower() for token in re.split(r"[_\-]+", result.sample_id) if token}
    derived = sorted(
        family_name
        for token, family_name in _LEGACY_LANGUAGE_FAMILY_BY_TOKEN.items()
        if token in sample_id_tokens
    )
    return derived


def classify_boundary_failure(result: EvaluationResult) -> str:
    """Assign one inspectable boundary-failure label from existing evaluation fields."""
    if result.schema_failure:
        return "null_or_unparsed_prediction"
    if result.overall_exact_match:
        return "full_exact_match"
    if result.intent_correct and result.entity_false_positives == 0 and result.entity_false_negatives > 0:
        return "intent_correct_entity_missing"
    if result.intent_correct and result.entity_false_positives > 0 and result.entity_false_negatives == 0:
        return "intent_correct_entity_extra"
    if result.intent_correct and (result.entity_false_positives > 0 or result.entity_false_negatives > 0):
        if result.entity_true_positives == 0:
            return "exact_intent_only"
        return "intent_correct_entity_drift"
    if result.entities_exact_match:
        return "exact_entities_only"
    if result.entity_true_positives > 0:
        return "intent_wrong_entity_partially_correct"
    return "intent_wrong_no_entity_match"


def _boundary_failure_rows(
    groups: dict[str, list[EvaluationResult]],
    *,
    key_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(groups.items()):
        total = len(group_rows)
        label_counts = Counter(classify_boundary_failure(row) for row in group_rows)
        for label, count in sorted(label_counts.items()):
            rows.append(
                {
                    key_name: key,
                    "boundary_failure_category": label,
                    "samples": count,
                    "rate": 0.0 if total == 0 else count / total,
                }
            )
    return rows


def _boundary_failure_summary(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    total = len(results)
    counts = Counter(classify_boundary_failure(row) for row in results)
    return [
        {
            "boundary_failure_category": label,
            "samples": count,
            "rate": 0.0 if total == 0 else count / total,
        }
        for label, count in sorted(counts.items())
    ]


def summarize_aggregates(results: list[EvaluationResult]) -> dict[str, list[dict[str, Any]]]:
    """Build aggregate tables for analysis and reporting."""
    by_model_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_prompt_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_model_prompt_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_language_family_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_slice_groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_failure_groups: dict[str, list[EvaluationResult]] = defaultdict(list)

    for row in results:
        by_model_groups[row.model_name].append(row)
        by_prompt_groups[row.prompt_language.value].append(row)
        by_model_prompt_groups[f"{row.model_name}__{row.prompt_language.value}"].append(row)
        for language_family in language_family_values_for_result(row):
            by_language_family_groups[language_family].append(row)

        for slice_tag in row.slice_tags:
            by_slice_groups[slice_tag.value].append(row)

        for bucket in row.assigned_failure_buckets:
            by_failure_groups[bucket.value].append(row)

    summary: dict[str, list[dict[str, Any]]] = {
        "by_model": [_group_metric_row(rows, "model_name", key) for key, rows in sorted(by_model_groups.items())],
        "by_prompt_language": [
            _group_metric_row(rows, "prompt_language", key) for key, rows in sorted(by_prompt_groups.items())
        ],
        "by_model_prompt": [
            _group_metric_row(rows, "model_prompt", key) for key, rows in sorted(by_model_prompt_groups.items())
        ],
        "by_language_family": [
            _group_metric_row(rows, "language_family", key) for key, rows in sorted(by_language_family_groups.items())
        ],
        "by_slice_tag": [_group_metric_row(rows, "slice_tag", key) for key, rows in sorted(by_slice_groups.items())],
        "by_failure_bucket": [
            {
                "failure_bucket": key,
                "samples": len(rows),
                "rate": 0.0 if not results else len(rows) / len(results),
            }
            for key, rows in sorted(by_failure_groups.items())
        ],
        "boundary_failure_summary": _boundary_failure_summary(results),
        "boundary_failure_by_model": _boundary_failure_rows(by_model_groups, key_name="model_name"),
        "boundary_failure_by_model_prompt": _boundary_failure_rows(by_model_prompt_groups, key_name="model_prompt"),
        "boundary_failure_by_language_family": _boundary_failure_rows(
            by_language_family_groups,
            key_name="language_family",
        ),
        "boundary_failure_by_slice_tag": _boundary_failure_rows(by_slice_groups, key_name="slice_tag"),
        "intent_confusion": build_confusion_summary(results),
    }
    return summary


def serialize_evaluation_results(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    """Convert evaluation records into JSONL-compatible rows."""
    rows: list[dict[str, Any]] = []
    for result in results:
        payload = result.model_dump(mode="json")
        payload["assigned_failure_buckets"] = bucket_names(result.assigned_failure_buckets)
        payload["boundary_failure_category"] = classify_boundary_failure(result)
        payload["slice_tags"] = sorted(tag.value for tag in result.slice_tags)
        payload["language_families"] = language_family_values_for_result(result)
        payload["overall_exact_match"] = result.overall_exact_match
        rows.append(payload)
    return rows


def evaluate_run_prediction_files(
    *,
    gold_records: list[SampleRecord],
    prediction_files: list[Path],
    load_prediction_file: Callable[[Path], list[PredictionRecord]],
) -> tuple[list[EvaluationResult], dict[str, list[dict[str, Any]]], list[str]]:
    """Evaluate every prediction file for one run."""
    all_results: list[EvaluationResult] = []
    issues: list[str] = []

    for prediction_file in prediction_files:
        predictions = load_prediction_file(prediction_file)
        file_results, file_issues = evaluate_prediction_set(gold_records=gold_records, predictions=predictions)
        all_results.extend(file_results)
        issues.extend(f"{prediction_file.name}: {issue}" for issue in file_issues)

    return all_results, summarize_aggregates(all_results), issues
