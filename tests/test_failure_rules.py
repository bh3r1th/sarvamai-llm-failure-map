"""Behavioral tests for deterministic failure bucket assignment."""

from __future__ import annotations

from code_switch_failure_map.eval.entity_metrics import score_entities
from code_switch_failure_map.eval.failure_rules import assign_failure_buckets
from code_switch_failure_map.eval.intent_metrics import score_intent
from code_switch_failure_map.schemas.prediction import ParsedEntity, ParsedPrediction, PredictionRecord
from code_switch_failure_map.schemas.sample import EntityMention, MetadataFlags, SampleRecord
from code_switch_failure_map.schemas.taxonomy import (
    EntityType,
    FailureCategory,
    IntentLabel,
    PromptLanguage,
    SliceTag,
    SourceSplit,
)


def _sample(*, sample_id: str = "s1", intent: IntentLabel = IntentLabel.REMINDER_CREATE, entities: list[EntityMention] | None = None, slice_tags: set[SliceTag] | None = None) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        source_split=SourceSplit.GOLDEN,
        text="kal subah 8 baje yaad dila",
        gold_intent=intent,
        gold_entities=entities or [],
        metadata_flags=MetadataFlags(),
        slice_tags=slice_tags or {SliceTag.PROMPT_LANGUAGE_HINGLISH},
        prompt_variant="baseline_v1",
        prompt_language=PromptLanguage.HINGLISH,
    )


def _prediction(*, sample_id: str = "s1", intent: str | None = None, entities: list[ParsedEntity] | None = None, parse_success: bool = True, schema_failure: bool = False) -> PredictionRecord:
    parsed_prediction = ParsedPrediction(intent=intent, entities=entities or []) if parse_success else None
    return PredictionRecord(
        sample_id=sample_id,
        model_name="demo-model",
        prompt_language=PromptLanguage.HINGLISH,
        prompt_text="prompt",
        raw_response='{"intent": null, "entities": []}' if parse_success else "bad json",
        parsed_prediction=parsed_prediction,
        parse_success=parse_success,
        schema_failure=schema_failure,
    )


def test_entity_drift_detection() -> None:
    sample = _sample(
        entities=[EntityMention(type=EntityType.DATE, text="tomorrow")],
        slice_tags={SliceTag.TEMPORAL_REFERENCE, SliceTag.PROMPT_LANGUAGE_HINGLISH},
    )
    prediction = _prediction(intent="reminder_create", entities=[ParsedEntity(label="date", value="yesterday")])

    buckets = assign_failure_buckets(
        sample=sample,
        prediction=prediction,
        intent_score=score_intent(sample, prediction),
        entity_score=score_entities(sample, prediction),
    )

    assert FailureCategory.ENTITY_DRIFT in buckets
    assert FailureCategory.TEMPORAL_INVERSION in buckets
    assert FailureCategory.OMISSION in buckets
    assert FailureCategory.OVER_EXTRACTION in buckets


def test_hallucination_detection() -> None:
    sample = _sample(intent=IntentLabel.INFORMATION_QUERY, entities=[])
    prediction = _prediction(intent="information_query", entities=[ParsedEntity(label="person", value="rahul")])

    buckets = assign_failure_buckets(
        sample=sample,
        prediction=prediction,
        intent_score=score_intent(sample, prediction),
        entity_score=score_entities(sample, prediction),
    )

    assert FailureCategory.HALLUCINATION in buckets
    assert FailureCategory.OVER_EXTRACTION in buckets


def test_schema_failure_propagation() -> None:
    sample = _sample()
    prediction = _prediction(parse_success=False, schema_failure=True)

    buckets = assign_failure_buckets(
        sample=sample,
        prediction=prediction,
        intent_score=score_intent(sample, prediction),
        entity_score=score_entities(sample, prediction),
    )

    assert buckets == {FailureCategory.SCHEMA_FAILURE}
