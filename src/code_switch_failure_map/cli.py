"""CLI entrypoint for the code-switch failure map project."""

from __future__ import annotations

from pathlib import Path

import typer

from code_switch_failure_map.data.load import dumpable_records, load_dataset
from code_switch_failure_map.data.validate import assert_valid_records
from code_switch_failure_map.models.base import BaseModelRunner, ensure_prompt_language
from code_switch_failure_map.models.openai_runner import OpenAIRunner
from code_switch_failure_map.models.sarvam import SarvamRunner
from code_switch_failure_map.schemas.prediction import PredictionRecord
from code_switch_failure_map.utils.io import write_jsonl

app = typer.Typer(help="Code-switching failure map command-line interface.")


def _build_runner(model_name: str, prompt_language_raw: str) -> BaseModelRunner:
    prompt_language = ensure_prompt_language(prompt_language_raw)
    normalized = model_name.strip().lower()

    if normalized == "sarvam-30b":
        return SarvamRunner(prompt_language=prompt_language)
    if normalized == "gpt-4o-mini":
        return OpenAIRunner(prompt_language=prompt_language)

    raise ValueError("Unsupported model_name. Expected one of: sarvam-30b, gpt-4o-mini")


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
def run_models(
    dataset_path: Path = typer.Option(..., "--dataset", exists=True, file_okay=True, dir_okay=False, readable=True),
    output_path: Path = typer.Option(..., "--output", file_okay=True, dir_okay=False),
    model_name: str = typer.Option(..., "--model"),
    prompt_language: str = typer.Option(..., "--prompt-language"),
) -> None:
    """Run one model against one dataset and persist predictions to JSONL."""
    records = load_dataset(dataset_path)

    try:
        runner = _build_runner(model_name=model_name, prompt_language_raw=prompt_language)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    predictions: list[PredictionRecord] = runner.run_batch(records)
    rows = [prediction.model_dump(mode="json") for prediction in predictions]
    write_jsonl(output_path, rows)

    typer.echo(f"Ran model: {runner.model_name}")
    typer.echo(f"Prompt language: {runner.prompt_language.value}")
    typer.echo(f"Samples processed: {len(predictions)}")
    typer.echo(f"Predictions path: {output_path}")


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
