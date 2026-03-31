"""Tests for model runner base interfaces and prediction persistence contract."""

from __future__ import annotations

from code_switch_failure_map.models.base import BaseModelRunner, ProviderResponse
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import IntentLabel, PromptLanguage, SliceTag, SourceSplit


class StubRunner(BaseModelRunner):
    """Stub runner that returns deterministic raw text for interface testing."""

    def __init__(self, *, raw_text: str) -> None:
        super().__init__(model_name="stub-model", prompt_language=PromptLanguage.ENGLISH)
        self._raw_text = raw_text

    def invoke_model(self, *, prompt_text: str, sample: SampleRecord) -> ProviderResponse:
        return ProviderResponse(raw_text=self._raw_text, latency_ms=12.5)


def _sample() -> SampleRecord:
    return SampleRecord(
        sample_id="s-1",
        source_split=SourceSplit.CURATED,
        text="book cab tomorrow",
        gold_intent=IntentLabel.NAVIGATION_REQUEST,
        prompt_variant="baseline",
        prompt_language=PromptLanguage.ENGLISH,
        slice_tags={SliceTag.PROMPT_LANGUAGE_EN},
    )


def test_parse_response_success() -> None:
    runner = StubRunner(raw_text='{"intent": "task_create", "entities": [{"label": "date", "value": "tomorrow"}]}')

    result = runner.parse_response(runner._raw_text)

    assert result.parse_success is True
    assert result.schema_failure is False
    assert result.parsed_prediction is not None
    assert result.parsed_prediction.intent == "task_create"
    assert len(result.parsed_prediction.entities) == 1


def test_parse_response_invalid_json_marks_parse_failure() -> None:
    runner = StubRunner(raw_text='{"intent": "task_create"')

    result = runner.parse_response(runner._raw_text)

    assert result.parse_success is False
    assert result.schema_failure is False
    assert "Invalid JSON" in (result.error_message or "")


def test_parse_response_schema_failure_for_wrong_structure() -> None:
    runner = StubRunner(raw_text='{"intent": "task_create", "entities": "not_a_list"}')

    result = runner.parse_response(runner._raw_text)

    assert result.parse_success is False
    assert result.schema_failure is True
    assert "entities must be a list" in (result.error_message or "")


def test_prediction_record_generation_contains_contract_fields() -> None:
    runner = StubRunner(raw_text='{"intent": null, "entities": []}')

    prediction = runner.run_one(_sample())

    assert prediction.sample_id == "s-1"
    assert prediction.model_name == "stub-model"
    assert prediction.prompt_language == PromptLanguage.ENGLISH
    assert prediction.prompt_text
    assert prediction.raw_response == '{"intent": null, "entities": []}'
    assert prediction.parse_success is True
    assert prediction.schema_failure is False
    assert prediction.parsed_prediction is not None
    assert prediction.input_tokens is not None
    assert prediction.output_tokens is not None
    assert prediction.total_tokens == prediction.input_tokens + prediction.output_tokens
    assert prediction.latency_ms == 12.5


def test_run_batch_uses_runner_interface() -> None:
    runner = StubRunner(raw_text='{"intent": null, "entities": []}')

    predictions = runner.run_batch([_sample(), _sample()])

    assert len(predictions) == 2
    assert all(pred.model_name == "stub-model" for pred in predictions)
