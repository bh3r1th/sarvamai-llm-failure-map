"""Schema definitions for model predictions and parse artifacts."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from code_switch_failure_map.schemas.taxonomy import PromptLanguage


class ParsedEntity(BaseModel):
    """Normalized entity emitted by model response parsing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    value: str | None = None


class ParsedPrediction(BaseModel):
    """Normalized parse result extracted from raw model text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: str | None = None
    entities: list[ParsedEntity] = Field(default_factory=list)


class PredictionRecord(BaseModel):
    """Prediction object for one model run against one sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    model_name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    prompt_language: PromptLanguage
    prompt_text: str
    raw_response: str
    parsed_prediction: ParsedPrediction | None = None
    parse_success: bool
    schema_failure: bool
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_parse_fields(self) -> "PredictionRecord":
        if self.parse_success and self.parsed_prediction is None:
            raise ValueError("parsed_prediction is required when parse_success is true")
        if self.parse_success and self.schema_failure:
            raise ValueError("schema_failure cannot be true when parse_success is true")
        return self
