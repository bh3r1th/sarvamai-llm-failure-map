"""Path helpers for dataset and output directory conventions."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ExperimentPaths(BaseModel):
    """Computed repository-relative paths used across experiments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_root: Path
    data_raw: Path
    data_curated: Path
    data_golden: Path
    data_predictions: Path
    outputs_metrics: Path
    outputs_failures: Path
    outputs_comparisons: Path


def build_experiment_paths(repo_root: Path) -> ExperimentPaths:
    """Build canonical path surface from a repository root."""

    data_dir = repo_root / "data"
    outputs_dir = repo_root / "outputs"

    return ExperimentPaths(
        repo_root=repo_root,
        data_raw=data_dir / "raw",
        data_curated=data_dir / "curated",
        data_golden=data_dir / "golden",
        data_predictions=data_dir / "predictions",
        outputs_metrics=outputs_dir / "metrics",
        outputs_failures=outputs_dir / "failures",
        outputs_comparisons=outputs_dir / "comparisons",
    )
