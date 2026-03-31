"""Schema validation tests for foundational typed surfaces."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from code_switch_failure_map.config import ExperimentConfig, ModelConfig
from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.prediction import ParsedPrediction, PredictionRecord, TokenCounts
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
        prompt_variant="baseline_v1",
        model_name="demo-model",
        raw_model_response='{"intent": "call_request"}',
        parsed_prediction=ParsedPrediction(intent=IntentLabel.CALL_REQUEST, entities=[]),
        parse_success=True,
        token_counts=TokenCounts(input_tokens=10, output_tokens=5, total_tokens=15),
    )

    evaluation = EvaluationResult(
        sample_id="s1",
        intent_exact_match=True,
        entities_exact_match=True,
        overall_exact_match=True,
        failure_categories=set(),
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
            prompt_variant="v1",
            model_name="demo-model",
            raw_model_response="{bad json}",
            parse_success=True,
        )


def test_token_count_sanity() -> None:
    with pytest.raises(ValidationError):
        TokenCounts(input_tokens=7, output_tokens=2, total_tokens=20)


def test_failure_category_parsing() -> None:
    result = EvaluationResult(
        sample_id="s3",
        intent_exact_match=False,
        entities_exact_match=False,
        overall_exact_match=False,
        failure_categories=["schema_failure", FailureCategory.ENTITY_DRIFT, "omission"],
    )

    assert FailureCategory.SCHEMA_FAILURE in result.failure_categories
    assert FailureCategory.ENTITY_DRIFT in result.failure_categories
    assert FailureCategory.OMISSION in result.failure_categories


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
