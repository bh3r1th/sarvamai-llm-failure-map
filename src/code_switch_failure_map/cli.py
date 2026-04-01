"""CLI entrypoint for the code-switch failure map project."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Any

import typer

from code_switch_failure_map.config import SmokeRunConfig
from code_switch_failure_map.data.load import load_dataset
from code_switch_failure_map.data.curate import count_by_intent, count_by_slice
from code_switch_failure_map.data.load import dumpable_records
from code_switch_failure_map.data.validate import assert_valid_records
from code_switch_failure_map.analysis.exemplar_mining import mine_failure_exemplars
from code_switch_failure_map.analysis.prompt_sensitivity import compare_prompt_sensitivity
from code_switch_failure_map.reports.blog_assets import build_blog_assets
from code_switch_failure_map.reports.export_json import write_table_bundle
from code_switch_failure_map.reports.tables import build_report_tables
from code_switch_failure_map.eval.aggregate import evaluate_run_prediction_files, serialize_evaluation_results
from code_switch_failure_map.models.base import BaseModelRunner, ensure_prompt_language
from code_switch_failure_map.models.openai_runner import OpenAIRunner
from code_switch_failure_map.models.sarvam import SarvamRunner
from code_switch_failure_map.schemas.evaluation import EvaluationArtifacts
from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.prediction import PredictionRecord
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.utils.io import (
    make_run_id,
    read_json,
    read_json_compatible_yaml,
    read_jsonl,
    slugify_filename,
    write_json,
    write_jsonl,
)

app = typer.Typer(help="Code-switching failure map command-line interface.")


def _build_runner(model_name: str, prompt_language_raw: str) -> BaseModelRunner:
    prompt_language = ensure_prompt_language(prompt_language_raw)
    normalized = model_name.strip().lower()

    if normalized == "sarvam-30b":
        return SarvamRunner(prompt_language=prompt_language)
    if normalized == "gpt-4.1-nano":
        return OpenAIRunner(prompt_language=prompt_language, model_name=model_name.strip())

    raise ValueError("Unsupported model_name. Expected one of: sarvam-30b, gpt-4.1-nano")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _load_smoke_run_config(config_path: Path) -> tuple[SmokeRunConfig, Path, Path]:
    repo_root = _repo_root()
    raw = read_json_compatible_yaml(config_path)
    config = SmokeRunConfig.model_validate(raw)
    dataset_path = _resolve_repo_path(config.dataset_path, repo_root=repo_root)
    output_dir = _resolve_repo_path(config.output_dir, repo_root=repo_root)
    return config, dataset_path, output_dir


def _sample_composition(records: list[SampleRecord]) -> dict[str, Any]:
    conflicting_instruction_ids = {"raw_083", "raw_096", "raw_100"}

    return {
        "total_samples": len(records),
        "clear": sum(
            1
            for record in records
            if "ambiguity" not in {tag.value for tag in record.slice_tags}
            and "adversarial" not in {tag.value for tag in record.slice_tags}
            and "transliteration_noise" not in {tag.value for tag in record.slice_tags}
        ),
        "noisy": sum(1 for record in records if record.metadata_flags.transliteration_noise),
        "ambiguous": sum(1 for record in records if record.metadata_flags.ambiguity),
        "adversarial": sum(1 for record in records if "adversarial" in {tag.value for tag in record.slice_tags}),
        "conflicting_or_multi_clause": sum(1 for record in records if record.sample_id in conflicting_instruction_ids),
        "intent_distribution": count_by_intent(records),
        "slice_distribution": count_by_slice(records),
    }


def _prediction_output_path(*, output_dir: Path, run_id: str, model_name: str, prompt_language: str) -> Path:
    filename = f"{slugify_filename(model_name)}__{slugify_filename(prompt_language)}.jsonl"
    return output_dir / run_id / filename


def _metrics_output_dir(run_id: str) -> Path:
    return _repo_root() / "outputs" / "metrics" / run_id


def _comparisons_output_dir(run_id: str) -> Path:
    return _repo_root() / "outputs" / "comparisons" / run_id


def _failures_output_dir(run_id: str) -> Path:
    return _repo_root() / "outputs" / "failures" / run_id


def run_experiment(config_path: Path, *, run_id_override: str | None = None) -> Path:
    """Execute a configured run without smoke-only dataset size constraints."""
    config, dataset_path, output_dir = _load_smoke_run_config(config_path)
    records = load_dataset(dataset_path)
    assert_valid_records(records)

    run_id = run_id_override or config.run_id or make_run_id(config.run_id_prefix)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_outputs: list[dict[str, Any]] = []
    for model_name in config.models:
        for prompt_variant in config.prompt_variants:
            runner = _build_runner(model_name=model_name, prompt_language_raw=prompt_variant.prompt_language.value)
            predictions: list[PredictionRecord] = runner.run_batch(records)
            rows = [prediction.model_dump(mode="json") for prediction in predictions]
            output_path = _prediction_output_path(
                output_dir=output_dir,
                run_id=run_id,
                model_name=model_name,
                prompt_language=prompt_variant.prompt_language.value,
            )
            write_jsonl(output_path, rows)

            manifest_outputs.append(
                {
                    "model_name": model_name,
                    "prompt_variant": prompt_variant.name,
                    "prompt_language": prompt_variant.prompt_language.value,
                    "path": str(output_path.relative_to(_repo_root())),
                    "rows_written": len(rows),
                }
            )

    manifest = {
        "run_id": run_id,
        "config_path": str(config_path),
        "dataset_path": str(dataset_path.relative_to(_repo_root())),
        "output_dir": str(run_dir.relative_to(_repo_root())),
        "models": config.models,
        "prompt_variants": [
            {"name": variant.name, "prompt_language": variant.prompt_language.value} for variant in config.prompt_variants
        ],
        "notes": config.notes,
        "dataset_summary": _sample_composition(records),
        "prediction_files": manifest_outputs,
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def run_smoke_experiment(config_path: Path, *, run_id_override: str | None = None) -> Path:
    """Execute the configured smoke run and persist deterministic artifacts."""
    config, dataset_path, output_dir = _load_smoke_run_config(config_path)
    records = load_dataset(dataset_path)
    assert_valid_records(records)

    if len(records) != 20:
        raise ValueError(f"Smoke dataset must contain exactly 20 samples, found {len(records)} in {dataset_path}")

    run_id = run_id_override or config.run_id or make_run_id(config.run_id_prefix)
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_outputs: list[dict[str, Any]] = []
    for model_name in config.models:
        for prompt_variant in config.prompt_variants:
            runner = _build_runner(model_name=model_name, prompt_language_raw=prompt_variant.prompt_language.value)
            predictions: list[PredictionRecord] = runner.run_batch(records)
            rows = [prediction.model_dump(mode="json") for prediction in predictions]
            output_path = _prediction_output_path(
                output_dir=output_dir,
                run_id=run_id,
                model_name=model_name,
                prompt_language=prompt_variant.prompt_language.value,
            )
            write_jsonl(output_path, rows)

            manifest_outputs.append(
                {
                    "model_name": model_name,
                    "prompt_variant": prompt_variant.name,
                    "prompt_language": prompt_variant.prompt_language.value,
                    "path": str(output_path.relative_to(_repo_root())),
                    "rows_written": len(rows),
                }
            )

    manifest = {
        "run_id": run_id,
        "config_path": str(config_path),
        "dataset_path": str(dataset_path.relative_to(_repo_root())),
        "output_dir": str(run_dir.relative_to(_repo_root())),
        "models": config.models,
        "prompt_variants": [
            {"name": variant.name, "prompt_language": variant.prompt_language.value} for variant in config.prompt_variants
        ],
        "notes": config.notes,
        "dataset_summary": _sample_composition(records),
        "prediction_files": manifest_outputs,
    }
    write_json(run_dir / "manifest.json", manifest)
    return run_dir


def run_configured_experiment(config_path: Path, *, run_id_override: str | None = None) -> tuple[Path, str, int]:
    """Route config execution to smoke or full run based on dataset size."""
    config, dataset_path, _output_dir = _load_smoke_run_config(config_path)
    records = load_dataset(dataset_path)
    dataset_size = len(records)

    if dataset_size == 20:
        return run_smoke_experiment(config_path, run_id_override=run_id_override), "smoke", dataset_size
    return run_experiment(config_path, run_id_override=run_id_override), "full", dataset_size


def _load_prediction_rows(path: Path) -> tuple[list[PredictionRecord], list[str]]:
    issues: list[str] = []
    predictions: list[PredictionRecord] = []
    for index, row in enumerate(read_jsonl(path), start=1):
        try:
            predictions.append(PredictionRecord.model_validate(row))
        except Exception as exc:
            issues.append(f"{path.name}: invalid prediction row at line {index}: {exc}")
    return predictions, issues


def _load_prediction_records_or_raise(path: Path) -> list[PredictionRecord]:
    predictions, issues = _load_prediction_rows(path)
    if issues:
        message = "\n".join(f"- {issue}" for issue in issues)
        raise ValueError(f"Prediction file validation failed for {path}:\n{message}")
    return predictions


def _prediction_files_for_run(run_dir: Path) -> tuple[list[Path], dict[str, Any] | None]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        files = [run_dir / Path(str(item["path"])).name for item in manifest.get("prediction_files", [])]
        return files, manifest

    files = sorted(path for path in run_dir.glob("*.jsonl") if path.is_file())
    return files, None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_evaluation_results(path: Path) -> list[EvaluationResult]:
    return [EvaluationResult.model_validate(row) for row in read_jsonl(path)]


def _load_run_context(run_dir: Path) -> tuple[str, Path, list[SampleRecord], list[PredictionRecord], list[EvaluationResult]]:
    prediction_files, manifest = _prediction_files_for_run(run_dir)
    if manifest is None or "dataset_path" not in manifest:
        raise ValueError(f"Run directory {run_dir} must contain manifest.json with dataset_path")

    run_id = str(manifest.get("run_id", run_dir.name))
    dataset_path = _resolve_repo_path(Path(str(manifest["dataset_path"])), repo_root=_repo_root())
    gold_records = load_dataset(dataset_path)

    evaluation_path = _metrics_output_dir(run_id) / "per_sample_evaluation.jsonl"
    if not evaluation_path.exists():
        raise ValueError(
            f"Missing evaluation artifact {evaluation_path}. Run `csfm evaluate --run-dir {run_dir}` first."
        )

    predictions: list[PredictionRecord] = []
    for prediction_file in prediction_files:
        predictions.extend(_load_prediction_records_or_raise(prediction_file))

    evaluation_results = _load_evaluation_results(evaluation_path)
    return run_id, dataset_path, gold_records, predictions, evaluation_results


def evaluate_prediction_outputs(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Evaluate prediction outputs and write per-sample plus aggregate summaries."""
    prediction_files, manifest = _prediction_files_for_run(run_dir)
    if not prediction_files:
        raise ValueError(f"No prediction JSONL files found in {run_dir}")

    if manifest is None or "dataset_path" not in manifest:
        raise ValueError(f"Run directory {run_dir} must contain manifest.json with dataset_path")

    dataset_path = _resolve_repo_path(Path(str(manifest["dataset_path"])), repo_root=_repo_root())
    gold_records = load_dataset(dataset_path)
    run_id = str(manifest.get("run_id", run_dir.name))

    results, aggregate_tables, issues = evaluate_run_prediction_files(
        gold_records=gold_records,
        prediction_files=prediction_files,
        load_prediction_file=_load_prediction_records_or_raise,
    )

    metrics_dir = _metrics_output_dir(run_id)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = metrics_dir / "per_sample_evaluation.jsonl"
    write_jsonl(per_sample_path, serialize_evaluation_results(results))

    summary_json = {
        "run_id": run_id,
        "dataset_path": str(dataset_path.relative_to(_repo_root())),
        "prediction_run_dir": str(run_dir.relative_to(_repo_root())),
        "issues": issues,
        "aggregate_tables": aggregate_tables,
    }
    summary_json_path = metrics_dir / "aggregate_summary.json"
    write_json(summary_json_path, summary_json)

    for table_name, rows in aggregate_tables.items():
        _write_csv(metrics_dir / f"{table_name}.csv", rows)

    artifacts = EvaluationArtifacts(
        run_id=run_id,
        dataset_path=str(dataset_path.relative_to(_repo_root())),
        prediction_run_dir=str(run_dir.relative_to(_repo_root())),
        per_sample_path=str(per_sample_path.relative_to(_repo_root())),
        summary_json_path=str(summary_json_path.relative_to(_repo_root())),
        aggregate_tables=[{"name": name, "rows": rows} for name, rows in aggregate_tables.items()],
    )
    return artifacts.model_dump(mode="json"), issues


def mine_failure_assets(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Mine strong failure exemplars for one evaluated run."""
    run_id, dataset_path, _gold_records, predictions, evaluation_results = _load_run_context(run_dir)
    output_dir = _failures_output_dir(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    exemplars, exemplar_index = mine_failure_exemplars(
        evaluation_results=evaluation_results,
        gold_records=gold_records,
        predictions=predictions,
    )
    exemplars_path = output_dir / "failure_exemplars.json"
    exemplar_index_path = output_dir / "exemplar_index.json"
    write_json(exemplars_path, {"run_id": run_id, "dataset_path": str(dataset_path.relative_to(_repo_root())), "rows": exemplars})
    write_json(exemplar_index_path, {"run_id": run_id, "rows": exemplar_index})

    _write_csv(output_dir / "exemplar_index.csv", exemplar_index)
    return {
        "run_id": run_id,
        "failure_exemplars_path": str(exemplars_path.relative_to(_repo_root())),
        "exemplar_index_json_path": str(exemplar_index_path.relative_to(_repo_root())),
        "exemplar_index_csv_path": str((output_dir / "exemplar_index.csv").relative_to(_repo_root())),
        "exemplar_count": len(exemplars),
    }, []


def build_report_assets_for_run(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Build comparison tables and compact blog assets for one evaluated run."""
    run_id, dataset_path, gold_records, predictions, evaluation_results = _load_run_context(run_dir)
    comparisons_dir = _comparisons_output_dir(run_id)
    failures_artifacts, _ = mine_failure_assets(run_dir)
    exemplars_payload = read_json(_repo_root() / str(failures_artifacts["failure_exemplars_path"]))
    exemplar_index_payload = read_json(_repo_root() / str(failures_artifacts["exemplar_index_json_path"]))
    exemplars = list(exemplars_payload.get("rows", []))
    exemplar_index = list(exemplar_index_payload.get("rows", []))

    tables = build_report_tables(evaluation_results, predictions, exemplar_index)
    created_files = write_table_bundle(comparisons_dir, tables)
    prompt_sensitivity = compare_prompt_sensitivity(evaluation_results)
    prompt_sensitivity_path = comparisons_dir / "prompt_sensitivity.json"
    write_json(prompt_sensitivity_path, {"run_id": run_id, "rows": prompt_sensitivity})

    blog_assets = build_blog_assets(results=evaluation_results, tables=tables, exemplars=exemplars)
    blog_assets_path = comparisons_dir / "blog_assets.json"
    findings_path = comparisons_dir / "top_findings.txt"
    write_json(
        blog_assets_path,
        {
            "run_id": run_id,
            "dataset_path": str(dataset_path.relative_to(_repo_root())),
            "assets": blog_assets,
        },
    )
    findings_path.write_text("\n".join(blog_assets["top_findings_summary"]), encoding="utf-8")
    created_files.extend([str(prompt_sensitivity_path), str(blog_assets_path), str(findings_path)])

    return {
        "run_id": run_id,
        "dataset_path": str(dataset_path.relative_to(_repo_root())),
        "comparison_output_dir": str(comparisons_dir.relative_to(_repo_root())),
        "created_files": [str(Path(path).relative_to(_repo_root())) if Path(path).is_absolute() else path for path in created_files],
        "top_findings_summary": blog_assets["top_findings_summary"],
        "strongest_examples_count": len(blog_assets["strongest_examples"]),
    }, []


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


@app.command("smoke-run")
def smoke_run(
    config_path: Path = typer.Option(
        Path("configs/smoke_run.yaml"),
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
) -> None:
    """Run the first controlled 20-sample smoke experiment."""
    run_dir = run_smoke_experiment(config_path, run_id_override=run_id)
    typer.echo(f"Smoke run complete: {run_dir}")
    typer.echo(f"Manifest path: {run_dir / 'manifest.json'}")


def _run_evaluate_outputs(run_dir: Path) -> None:
    artifacts, issues = evaluate_prediction_outputs(run_dir)

    typer.echo(f"Run id: {artifacts['run_id']}")
    typer.echo(f"Per-sample evaluations: {artifacts['per_sample_path']}")
    typer.echo(f"Aggregate summary JSON: {artifacts['summary_json_path']}")
    for table in artifacts["aggregate_tables"]:
        typer.echo(f"Aggregate table: {table['name']} ({len(table['rows'])} rows)")

    if issues:
        typer.echo("Evaluator issues detected:")
        for issue in issues:
            typer.echo(f"- {issue}")
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True, readable=True),
) -> None:
    """Evaluate prediction outputs against gold data and write summaries."""
    _run_evaluate_outputs(run_dir)


@app.command("evaluate-outputs")
def evaluate_outputs(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True, readable=True),
) -> None:
    """Evaluate prediction outputs against gold data and write summaries."""
    _run_evaluate_outputs(run_dir)


@app.command("build-report")
def build_report(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True, readable=True),
) -> None:
    """Build report tables and blog assets from evaluated outputs."""
    artifacts, _ = build_report_assets_for_run(run_dir)
    typer.echo(f"Run id: {artifacts['run_id']}")
    typer.echo(f"Comparison output dir: {artifacts['comparison_output_dir']}")
    for path in artifacts["created_files"]:
        typer.echo(f"Created: {path}")


@app.command("mine-failures")
def mine_failures(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False, dir_okay=True, readable=True),
) -> None:
    """Mine blog-usable failure exemplars from evaluated outputs."""
    artifacts, _ = mine_failure_assets(run_dir)
    typer.echo(f"Run id: {artifacts['run_id']}")
    typer.echo(f"Failure exemplars: {artifacts['failure_exemplars_path']}")
    typer.echo(f"Exemplar index JSON: {artifacts['exemplar_index_json_path']}")
    typer.echo(f"Exemplar index CSV: {artifacts['exemplar_index_csv_path']}")


if __name__ == "__main__":
    app()
