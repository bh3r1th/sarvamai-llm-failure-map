"""Typed configuration surface for code-switching failure-map experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from code_switch_failure_map.schemas.taxonomy import PromptLanguage
from code_switch_failure_map.utils.paths import ExperimentPaths, build_experiment_paths


class ModelConfig(BaseModel):
    """Model-facing configuration for a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    prompt_variant: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    prompt_language: PromptLanguage


class ExperimentConfig(BaseModel):
    """Top-level typed config for current experiment scope."""

    model_config = ConfigDict(extra="forbid")

    experiment_name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    repo_root: Path
    models: list[ModelConfig] = Field(default_factory=list)
    paths: ExperimentPaths

    @classmethod
    def from_repo_root(
        cls,
        *,
        experiment_name: str,
        repo_root: Path,
        models: list[ModelConfig] | None = None,
    ) -> "ExperimentConfig":
        """Construct config with canonical path defaults."""

        return cls(
            experiment_name=experiment_name,
            repo_root=repo_root,
            models=models or [],
            paths=build_experiment_paths(repo_root),
        )


class SmokePromptVariantConfig(BaseModel):
    """Configured prompt surface for a smoke run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
    prompt_language: PromptLanguage


class SmokeRunConfig(BaseModel):
    """Small reproducible manifest for smoke runs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_path: Path
    models: list[Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]] = Field(default_factory=list)
    prompt_variants: list[SmokePromptVariantConfig] = Field(default_factory=list)
    output_dir: Path
    run_id: str | None = None
    run_id_prefix: Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)] = "smoke"
    notes: str = ""

    @model_validator(mode="after")
    def validate_lists(self) -> "SmokeRunConfig":
        if not self.models:
            raise ValueError("models must contain at least one model")
        if not self.prompt_variants:
            raise ValueError("prompt_variants must contain at least one prompt variant")

        languages = [variant.prompt_language for variant in self.prompt_variants]
        if len(set(languages)) != len(languages):
            raise ValueError("prompt_variants must use unique prompt_language values")

        return self
