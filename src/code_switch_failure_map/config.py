"""Typed configuration surface for code-switching failure-map experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

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
