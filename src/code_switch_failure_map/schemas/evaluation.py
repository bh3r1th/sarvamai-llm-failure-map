"""Schema definitions for per-sample evaluation and aggregate outputs."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from code_switch_failure_map.schemas.taxonomy import FailureCategory, PromptLanguage, SliceTag


class EvaluationResult(BaseModel):
    """Evaluation result for one prediction relative to one gold sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    model_name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)] = "unknown-model"
    prompt_language: PromptLanguage = PromptLanguage.ENGLISH
    gold_intent: str | None = None
    predicted_intent: str | None = None
    intent_correct: bool
    entity_gold_count: int = Field(default=0, ge=0)
    entity_predicted_count: int = Field(default=0, ge=0)
    entity_true_positives: int = Field(default=0, ge=0)
    entity_false_positives: int = Field(default=0, ge=0)
    entity_false_negatives: int = Field(default=0, ge=0)
    entity_precision: float = Field(ge=0.0, le=1.0)
    entity_recall: float = Field(ge=0.0, le=1.0)
    entity_f1: float = Field(ge=0.0, le=1.0)
    schema_failure: bool = False
    assigned_failure_buckets: set[FailureCategory] = Field(default_factory=set)
    slice_tags: set[SliceTag] = Field(default_factory=set)
    notes: str | None = None

    @property
    def intent_exact_match(self) -> bool:
        return self.intent_correct

    @property
    def entities_exact_match(self) -> bool:
        return self.entity_false_positives == 0 and self.entity_false_negatives == 0

    @property
    def overall_exact_match(self) -> bool:
        return self.intent_correct and self.entities_exact_match and not self.schema_failure

    @property
    def failure_categories(self) -> set[FailureCategory]:
        return self.assigned_failure_buckets

    @model_validator(mode="after")
    def validate_entity_counts(self) -> "EvaluationResult":
        if self.entity_true_positives > self.entity_gold_count:
            raise ValueError("entity_true_positives cannot exceed entity_gold_count")
        if self.entity_true_positives > self.entity_predicted_count:
            raise ValueError("entity_true_positives cannot exceed entity_predicted_count")

        expected_fp = self.entity_predicted_count - self.entity_true_positives
        expected_fn = self.entity_gold_count - self.entity_true_positives
        if self.entity_false_positives != expected_fp:
            raise ValueError("entity_false_positives must equal entity_predicted_count - entity_true_positives")
        if self.entity_false_negatives != expected_fn:
            raise ValueError("entity_false_negatives must equal entity_gold_count - entity_true_positives")

        if self.overall_exact_match and self.assigned_failure_buckets:
            raise ValueError("assigned_failure_buckets must be empty when overall_exact_match is true")
        return self


class AggregateTable(BaseModel):
    """Named aggregate table with row dictionaries."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    rows: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationArtifacts(BaseModel):
    """Collection of evaluator outputs written for one run."""

    model_config = ConfigDict(extra="forbid")

    run_id: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    dataset_path: str
    prediction_run_dir: str
    per_sample_path: str
    summary_json_path: str
    aggregate_tables: list[AggregateTable] = Field(default_factory=list)
