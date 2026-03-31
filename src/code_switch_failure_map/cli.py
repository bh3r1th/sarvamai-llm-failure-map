"""CLI entrypoint for the code-switch failure map project."""

from __future__ import annotations

from pathlib import Path

import typer

from code_switch_failure_map.data.load import dumpable_records, load_dataset
from code_switch_failure_map.data.validate import assert_valid_records
from code_switch_failure_map.utils.io import write_jsonl

app = typer.Typer(help="Code-switching failure map command-line interface.")


@app.command("make-dataset")
def make_dataset(
    input_path: Path = typer.Option(..., "--input", exists=True, file_okay=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(..., "--output", file_okay=True, dir_okay=False),
    validate: bool = typer.Option(True, "--validate/--no-validate"),
) -> None:
    """Read, validate, normalize, and write dataset JSONL records."""
    records = load_dataset(input_path)
    if validate:
        assert_valid_records(records)

    serialized = dumpable_records(records)
    write_jsonl(output_path, serialized)

    typer.echo(f"Loaded records: {len(records)}")
    typer.echo(f"Wrote normalized records: {len(serialized)}")
    typer.echo(f"Output path: {output_path}")


@app.command("run-models")
def run_models() -> None:
    """Placeholder for model runner orchestration."""
    typer.echo("Not implemented yet: run-models")
    raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate() -> None:
    """Placeholder for evaluation execution."""
    typer.echo("Not implemented yet: evaluate")
    raise typer.Exit(code=1)


@app.command("build-report")
def build_report() -> None:
    """Placeholder for report build pipeline."""
    typer.echo("Not implemented yet: build-report")
    raise typer.Exit(code=1)


@app.command("mine-failures")
def mine_failures() -> None:
    """Placeholder for failure mining pipeline."""
    typer.echo("Not implemented yet: mine-failures")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
