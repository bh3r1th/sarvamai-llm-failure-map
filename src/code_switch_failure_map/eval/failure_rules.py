"""Deterministic failure bucket assignment from gold vs prediction comparison."""

from __future__ import annotations

from code_switch_failure_map.eval.entity_metrics import EntityScore, NormalizedEntity, has_temporal_label
from code_switch_failure_map.eval.intent_metrics import IntentScore
from code_switch_failure_map.schemas.prediction import PredictionRecord
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import FailureCategory, SliceTag


def _has_same_label_value_mismatch(gold: list[NormalizedEntity], predicted: list[NormalizedEntity]) -> bool:
    gold_by_label: dict[str, set[str]] = {}
    predicted_by_label: dict[str, set[str]] = {}

    for entity in gold:
        gold_by_label.setdefault(entity.label, set()).add(entity.value)
    for entity in predicted:
        predicted_by_label.setdefault(entity.label, set()).add(entity.value)

    for label in set(gold_by_label) & set(predicted_by_label):
        if gold_by_label[label] != predicted_by_label[label]:
            return True
    return False


def _has_temporal_inversion(gold: list[NormalizedEntity], predicted: list[NormalizedEntity]) -> bool:
    gold_temporal = [entity for entity in gold if has_temporal_label(entity)]
    predicted_temporal = [entity for entity in predicted if has_temporal_label(entity)]
    if not gold_temporal or not predicted_temporal:
        return False
    return _has_same_label_value_mismatch(gold_temporal, predicted_temporal)


def assign_failure_buckets(
    *,
    sample: SampleRecord,
    prediction: PredictionRecord,
    intent_score: IntentScore,
    entity_score: EntityScore,
) -> set[FailureCategory]:
    """Assign deterministic failure buckets for one evaluated sample."""
    buckets: set[FailureCategory] = set()

    if prediction.schema_failure or not prediction.parse_success or prediction.parsed_prediction is None:
        buckets.add(FailureCategory.SCHEMA_FAILURE)
        return buckets

    if not intent_score.exact_match:
        buckets.add(FailureCategory.INTENT_CONFUSION)

    if _has_temporal_inversion(entity_score.unmatched_gold, entity_score.unmatched_predicted):
        buckets.add(FailureCategory.TEMPORAL_INVERSION)

    if SliceTag.SENTIMENT_LOAD in sample.slice_tags and (not intent_score.exact_match or entity_score.f1 < 1.0):
        buckets.add(FailureCategory.SENTIMENT_INVERSION)

    if _has_same_label_value_mismatch(entity_score.unmatched_gold, entity_score.unmatched_predicted):
        buckets.add(FailureCategory.ENTITY_DRIFT)

    gold_labels = {entity.label for entity in entity_score.matched + entity_score.unmatched_gold}
    hallucinated = [entity for entity in entity_score.unmatched_predicted if entity.label not in gold_labels]
    if hallucinated or (entity_score.gold_count == 0 and entity_score.predicted_count > 0):
        buckets.add(FailureCategory.HALLUCINATION)

    if entity_score.false_negatives > 0:
        buckets.add(FailureCategory.OMISSION)

    if entity_score.false_positives > 0:
        buckets.add(FailureCategory.OVER_EXTRACTION)

    if not buckets and (not intent_score.exact_match or entity_score.f1 < 1.0):
        buckets.add(FailureCategory.OTHER)

    return buckets
