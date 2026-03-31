"""Schema definitions for model predictions and parse artifacts."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from code_switch_failure_map.schemas.sample import EntityMention


class TokenCounts(BaseModel):
    """Token usage accounting for one model response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_token_totals(self) -> "TokenCounts":
        if self.total_tokens is None:
            return self

        if self.input_tokens is not None and self.output_tokens is not None:
            expected = self.input_tokens + self.output_tokens
            if self.total_tokens != expected:
                raise ValueError("total_tokens must equal input_tokens + output_tokens")
        return self


class ParsedPrediction(BaseModel):
    """Normalized parse result extracted from raw model text."""

    model_config = ConfigDict(extra="forbid")

    intent: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    entities: list[EntityMention] | dict[str, str | list[str]] = Field(default_factory=list)


class PredictionRecord(BaseModel):
    """Prediction object for one model run against one sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    prompt_variant: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    model_name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    raw_model_response: Annotated[str, StringConstraints(min_length=1)]
    parsed_prediction: ParsedPrediction | None = None
    parse_success: bool
    schema_failure: Annotated[str, StringConstraints(min_length=1)] | None = None
    token_counts: TokenCounts | None = None

    @model_validator(mode="after")
    def validate_parse_fields(self) -> "PredictionRecord":
        if self.parse_success and self.parsed_prediction is None:
            raise ValueError("parsed_prediction is required when parse_success is true")
        if not self.parse_success and self.schema_failure is None:
            raise ValueError("schema_failure is required when parse_success is false")
        return self
