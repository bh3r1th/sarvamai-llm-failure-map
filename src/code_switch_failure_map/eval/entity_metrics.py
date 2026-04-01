"""Entity scoring helpers for extraction evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from code_switch_failure_map.schemas.prediction import ParsedEntity, PredictionRecord
from code_switch_failure_map.schemas.sample import EntityMention, SampleRecord

_TEMPORAL_LABELS = {"time", "date", "datetime", "duration"}


@dataclass(frozen=True)
class NormalizedEntity:
    """Canonical entity tuple used for exact matching."""

    label: str
    value: str


@dataclass(frozen=True)
class EntityScore:
    """Entity scoring summary for one sample."""

    gold_count: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    exact_match: bool
    matched: list[NormalizedEntity]
    unmatched_gold: list[NormalizedEntity]
    unmatched_predicted: list[NormalizedEntity]


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().lower().split())


def normalize_gold_entity(entity: EntityMention) -> NormalizedEntity:
    """Normalize a gold entity into the exact-match comparison form."""
    canonical_value = entity.normalized_value if entity.normalized_value else entity.text
    return NormalizedEntity(label=entity.type.value, value=_normalize_text(canonical_value))


def normalize_predicted_entity(entity: ParsedEntity) -> NormalizedEntity:
    """Normalize a predicted entity into the exact-match comparison form."""
    canonical_value = entity.value if entity.value is not None else ""
    return NormalizedEntity(label=_normalize_text(entity.label), value=_normalize_text(canonical_value))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def score_entities(sample: SampleRecord, prediction: PredictionRecord) -> EntityScore:
    """Compute exact-match entity precision/recall/F1."""
    gold_entities = [normalize_gold_entity(entity) for entity in sample.gold_entities]
    predicted_entities = (
        [normalize_predicted_entity(entity) for entity in prediction.parsed_prediction.entities]
        if prediction.parsed_prediction is not None
        else []
    )

    gold_counter = Counter(gold_entities)
    predicted_counter = Counter(predicted_entities)
    matched_counter = gold_counter & predicted_counter
    unmatched_gold_counter = gold_counter - matched_counter
    unmatched_predicted_counter = predicted_counter - matched_counter

    matched = list(matched_counter.elements())
    unmatched_gold = list(unmatched_gold_counter.elements())
    unmatched_predicted = list(unmatched_predicted_counter.elements())

    true_positives = sum(matched_counter.values())
    false_positives = sum(unmatched_predicted_counter.values())
    false_negatives = sum(unmatched_gold_counter.values())
    precision = _safe_ratio(true_positives, len(predicted_entities))
    recall = _safe_ratio(true_positives, len(gold_entities))
    f1 = 1.0 if precision == 1.0 and recall == 1.0 else (0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall))

    return EntityScore(
        gold_count=len(gold_entities),
        predicted_count=len(predicted_entities),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=false_positives == 0 and false_negatives == 0,
        matched=matched,
        unmatched_gold=unmatched_gold,
        unmatched_predicted=unmatched_predicted,
    )


def has_temporal_label(entity: NormalizedEntity) -> bool:
    """Return whether the entity label is temporal."""
    return entity.label in _TEMPORAL_LABELS
