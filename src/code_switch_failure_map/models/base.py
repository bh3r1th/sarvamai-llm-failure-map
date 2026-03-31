"""Base runner contract for model execution and response parsing."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from code_switch_failure_map.models.tokenizer_stats import TokenUsage, merge_token_usage
from code_switch_failure_map.prompts.render import render_extraction_prompt
from code_switch_failure_map.schemas.prediction import ParsedEntity, ParsedPrediction, PredictionRecord
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import PromptLanguage

_EXPECTED_TOP_LEVEL_KEYS = {"intent", "entities"}
_EXPECTED_ENTITY_KEYS = {"label", "value"}


@dataclass(frozen=True)
class ParseResult:
    """Outcome of parsing one raw model response."""

    parsed_prediction: ParsedPrediction | None
    parse_success: bool
    schema_failure: bool
    error_message: str | None = None


@dataclass(frozen=True)
class ProviderResponse:
    """Raw provider output before parsing/normalization."""

    raw_text: str
    token_usage: TokenUsage | None = None
    latency_ms: float | None = None


class BaseModelRunner(ABC):
    """Abstract model runner with strict JSON parsing and record creation."""

    def __init__(self, *, model_name: str, prompt_language: PromptLanguage) -> None:
        self.model_name = model_name
        self.prompt_language = prompt_language

    @abstractmethod
    def invoke_model(self, *, prompt_text: str, sample: SampleRecord) -> ProviderResponse:
        """Execute one model request and return raw provider output."""

    def parse_response(self, raw_text: str) -> ParseResult:
        """Strictly parse JSON output with expected key contract."""
        try:
            decoded = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            return ParseResult(
                parsed_prediction=None,
                parse_success=False,
                schema_failure=False,
                error_message=f"Invalid JSON response: {exc.msg}",
            )

        if not isinstance(decoded, dict):
            return ParseResult(
                parsed_prediction=None,
                parse_success=False,
                schema_failure=True,
                error_message="Response JSON must be an object",
            )

        if set(decoded.keys()) != _EXPECTED_TOP_LEVEL_KEYS:
            return ParseResult(
                parsed_prediction=None,
                parse_success=False,
                schema_failure=True,
                error_message="Response JSON must contain exactly keys: intent, entities",
            )

        intent = decoded["intent"]
        entities = decoded["entities"]

        if intent is not None and not isinstance(intent, str):
            return ParseResult(None, False, True, "intent must be string or null")
        if not isinstance(entities, list):
            return ParseResult(None, False, True, "entities must be a list")

        normalized_entities: list[ParsedEntity] = []
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                return ParseResult(None, False, True, f"entities[{index}] must be an object")
            if set(entity.keys()) != _EXPECTED_ENTITY_KEYS:
                return ParseResult(None, False, True, f"entities[{index}] must contain exactly keys: label, value")
            label = entity["label"]
            value = entity["value"]
            if not isinstance(label, str) or not label.strip():
                return ParseResult(None, False, True, f"entities[{index}].label must be a non-empty string")
            if value is not None and not isinstance(value, str):
                return ParseResult(None, False, True, f"entities[{index}].value must be string or null")
            normalized_entities.append(ParsedEntity(label=label, value=value))

        return ParseResult(
            parsed_prediction=ParsedPrediction(intent=intent, entities=normalized_entities),
            parse_success=True,
            schema_failure=False,
        )

    def make_prediction_record(
        self,
        *,
        sample: SampleRecord,
        prompt_text: str,
        provider_response: ProviderResponse,
        parse_result: ParseResult,
    ) -> PredictionRecord:
        """Build final persisted prediction object."""
        token_usage = merge_token_usage(
            prompt_text=prompt_text,
            raw_response=provider_response.raw_text,
            usage=provider_response.token_usage,
        )

        return PredictionRecord(
            sample_id=sample.sample_id,
            model_name=self.model_name,
            prompt_language=self.prompt_language,
            prompt_text=prompt_text,
            raw_response=provider_response.raw_text,
            parsed_prediction=parse_result.parsed_prediction,
            parse_success=parse_result.parse_success,
            schema_failure=parse_result.schema_failure,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            total_tokens=token_usage.total_tokens,
            latency_ms=provider_response.latency_ms,
            error_message=parse_result.error_message,
        )

    def run_one(self, sample: SampleRecord) -> PredictionRecord:
        """Run one sample end-to-end and return persisted prediction payload."""
        prompt_text = render_extraction_prompt(text=sample.text, prompt_language=self.prompt_language)
        provider_response = self.invoke_model(prompt_text=prompt_text, sample=sample)
        parse_result = self.parse_response(provider_response.raw_text)
        return self.make_prediction_record(
            sample=sample,
            prompt_text=prompt_text,
            provider_response=provider_response,
            parse_result=parse_result,
        )

    def run_batch(self, samples: list[SampleRecord]) -> list[PredictionRecord]:
        """Run a list of samples sequentially."""
        return [self.run_one(sample) for sample in samples]


def ensure_prompt_language(value: str) -> PromptLanguage:
    """Parse prompt language string into enum with explicit failure."""
    try:
        return PromptLanguage(value)
    except ValueError as exc:
        supported = ", ".join(language.value for language in PromptLanguage)
        raise ValueError(f"Unsupported prompt_language={value!r}. Expected one of: {supported}") from exc
