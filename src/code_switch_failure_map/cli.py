"""CLI entrypoint scaffolding for the code-switch failure map project."""

from __future__ import annotations

import typer

app = typer.Typer(help="Code-switching failure map command-line interface.")


@app.command("make-dataset")
def make_dataset() -> None:
    """Create and/or curate dataset artifacts."""
    # TODO: Wire dataset generation and curation pipeline.
    raise typer.Exit(code=0)


@app.command("run-models")
def run_models() -> None:
    """Run model backends on curated evaluation inputs."""
    # TODO: Wire model runner orchestration across providers.
    raise typer.Exit(code=0)


@app.command("evaluate")
def evaluate() -> None:
    """Evaluate predictions against golden labels."""
    # TODO: Wire evaluation metrics and failure bucketing.
    raise typer.Exit(code=0)


@app.command("build-report")
def build_report() -> None:
    """Build report tables and blog-ready assets."""
    # TODO: Wire report artifact generation for publication.
    raise typer.Exit(code=0)


@app.command("mine-failures")
def mine_failures() -> None:
    """Mine representative failure exemplars for analysis."""
    # TODO: Wire failure exemplar mining and slice analysis.
    raise typer.Exit(code=0)
