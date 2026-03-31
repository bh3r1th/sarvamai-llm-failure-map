"""Schema definitions for exact-match style evaluation outputs."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from code_switch_failure_map.schemas.taxonomy import FailureCategory


class EvaluationResult(BaseModel):
    """Evaluation result for one prediction relative to one gold sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    intent_exact_match: bool
    entities_exact_match: bool
    overall_exact_match: bool
    failure_categories: set[FailureCategory] = Field(default_factory=set)

    @model_validator(mode="after")
    def validate_overall_exact_match(self) -> "EvaluationResult":
        expected = self.intent_exact_match and self.entities_exact_match
        if self.overall_exact_match != expected:
            raise ValueError("overall_exact_match must be intent_exact_match AND entities_exact_match")

        if self.overall_exact_match and self.failure_categories:
            raise ValueError("failure_categories must be empty when overall_exact_match is true")
        return self
