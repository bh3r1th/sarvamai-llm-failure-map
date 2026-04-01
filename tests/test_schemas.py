"""Schema validation tests for foundational typed surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from code_switch_failure_map.config import ExperimentConfig, ModelConfig
from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.prediction import ParsedPrediction, PredictionRecord
from code_switch_failure_map.schemas.sample import EntityMention, MetadataFlags, SampleRecord
from code_switch_failure_map.schemas.taxonomy import (
    EntityType,
    FailureCategory,
    IntentLabel,
    PromptLanguage,
    SliceTag,
    SourceSplit,
)
from code_switch_failure_map.utils.paths import build_experiment_paths


def test_valid_intent_and_entity_creation() -> None:
    sample = SampleRecord(
        sample_id="s1",
        source_split=SourceSplit.GOLDEN,
        text="kal subah 8 baje mummy ko call karna",
        normalized_text=None,
        gold_intent=IntentLabel.CALL_REQUEST,
        gold_entities=[
            EntityMention(
                type=EntityType.TIME,
                text="subah 8 baje",
                normalized_value="08:00",
                confidence=0.95,
                start_char=4,
                end_char=16,
            ),
            EntityMention(type=EntityType.PERSON, text="mummy"),
        ],
        metadata_flags=MetadataFlags(code_switching=True),
        slice_tags={SliceTag.CODE_SWITCHING, SliceTag.PROMPT_LANGUAGE_HINGLISH},
        prompt_variant="baseline_v1",
        prompt_language=PromptLanguage.HINGLISH,
    )

    prediction = PredictionRecord(
        sample_id="s1",
        model_name="demo-model",
        prompt_language=PromptLanguage.HINGLISH,
        prompt_text="prompt",
        raw_response='{"intent": "call_request", "entities": []}',
        parsed_prediction=ParsedPrediction(intent="call_request", entities=[]),
        parse_success=True,
        schema_failure=False,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )

    evaluation = EvaluationResult(
        sample_id="s1",
        model_name="demo-model",
        prompt_language=PromptLanguage.HINGLISH,
        gold_intent="call_request",
        predicted_intent="call_request",
        intent_correct=True,
        entity_gold_count=0,
        entity_predicted_count=0,
        entity_true_positives=0,
        entity_false_positives=0,
        entity_false_negatives=0,
        entity_precision=1.0,
        entity_recall=1.0,
        entity_f1=1.0,
        schema_failure=False,
        assigned_failure_buckets=set(),
    )

    cfg = ExperimentConfig.from_repo_root(
        experiment_name="hinglish-baseline",
        repo_root=Path("/tmp/repo"),
        models=[
            ModelConfig(
                model_name="demo-model",
                prompt_variant="baseline_v1",
                prompt_language=PromptLanguage.HINGLISH,
            )
        ],
    )

    assert sample.gold_intent == IntentLabel.CALL_REQUEST
    assert prediction.parse_success is True
    assert evaluation.overall_exact_match is True
    assert cfg.paths.data_golden == Path("/tmp/repo/data/golden")


def test_invalid_label_rejection() -> None:
    with pytest.raises(ValidationError):
        SampleRecord(
            sample_id="s1",
            source_split=SourceSplit.RAW,
            text="hello",
            gold_intent="weather_query",
            gold_entities=[],
            metadata_flags=MetadataFlags(),
            slice_tags={SliceTag.PROMPT_LANGUAGE_EN},
            prompt_variant="v1",
            prompt_language=PromptLanguage.ENGLISH,
        )

    with pytest.raises(ValidationError):
        EntityMention(type="food", text="pizza")


def test_required_field_enforcement() -> None:
    with pytest.raises(ValidationError):
        PredictionRecord(
            sample_id="s1",
            model_name="demo-model",
            prompt_language=PromptLanguage.ENGLISH,
            prompt_text="prompt",
            raw_response="{bad json}",
            parse_success=True,
            schema_failure=False,
        )


def test_parse_success_schema_failure_consistency() -> None:
    with pytest.raises(ValidationError):
        PredictionRecord(
            sample_id="s1",
            model_name="demo-model",
            prompt_language=PromptLanguage.ENGLISH,
            prompt_text="prompt",
            raw_response='{"intent": null, "entities": []}',
            parse_success=True,
            schema_failure=True,
            parsed_prediction=ParsedPrediction(intent=None, entities=[]),
        )


def test_failure_category_parsing() -> None:
    result = EvaluationResult(
        sample_id="s3",
        model_name="demo-model",
        prompt_language=PromptLanguage.ENGLISH,
        gold_intent="call_request",
        predicted_intent="message_send",
        intent_correct=False,
        entity_gold_count=1,
        entity_predicted_count=2,
        entity_true_positives=0,
        entity_false_positives=2,
        entity_false_negatives=1,
        entity_precision=0.0,
        entity_recall=0.0,
        entity_f1=0.0,
        schema_failure=False,
        assigned_failure_buckets=["schema_failure", FailureCategory.ENTITY_DRIFT, "omission"],
    )

    assert FailureCategory.SCHEMA_FAILURE in result.assigned_failure_buckets
    assert FailureCategory.ENTITY_DRIFT in result.assigned_failure_buckets
    assert FailureCategory.OMISSION in result.assigned_failure_buckets


def test_optional_span_and_normalized_values() -> None:
    mention = EntityMention(type=EntityType.DATE, text="kal", normalized_value=None)
    assert mention.normalized_value is None
    assert mention.start_char is None

    with pytest.raises(ValidationError):
        EntityMention(type=EntityType.DATE, text="kal", start_char=2)


def test_paths_surface() -> None:
    paths = build_experiment_paths(Path("/repo"))

    assert paths.data_raw == Path("/repo/data/raw")
    assert paths.outputs_comparisons == Path("/repo/outputs/comparisons")
