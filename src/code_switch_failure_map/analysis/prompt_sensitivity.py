"""English vs Hinglish prompt sensitivity analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from code_switch_failure_map.analysis.slices import analyze_slice_breakdowns
from code_switch_failure_map.schemas.evaluation import EvaluationResult


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _overall_metrics(rows: list[EvaluationResult]) -> dict[str, float]:
    return {
        "intent_accuracy": _mean([1.0 if row.intent_correct else 0.0 for row in rows]),
        "entity_f1": _mean([row.entity_f1 for row in rows]),
        "exact_match_rate": _mean([1.0 if row.overall_exact_match else 0.0 for row in rows]),
        "schema_failure_rate": _mean([1.0 if row.schema_failure else 0.0 for row in rows]),
    }


def compare_prompt_sensitivity(results: list[EvaluationResult]) -> dict[str, list[dict[str, Any]]]:
    """Compare English and Hinglish prompts by model and slice."""
    by_model_language: dict[tuple[str, str], list[EvaluationResult]] = defaultdict(list)
    for row in results:
        by_model_language[(row.model_name, row.prompt_language.value)].append(row)

    overall: list[dict[str, Any]] = []
    for model_name in sorted({row.model_name for row in results}):
        english = by_model_language.get((model_name, "english"), [])
        hinglish = by_model_language.get((model_name, "hinglish"), [])
        if not english or not hinglish:
            continue

        en_metrics = _overall_metrics(english)
        hi_metrics = _overall_metrics(hinglish)
        overall.append(
            {
                "model_name": model_name,
                "english_samples": len(english),
                "hinglish_samples": len(hinglish),
                "english_intent_accuracy": en_metrics["intent_accuracy"],
                "hinglish_intent_accuracy": hi_metrics["intent_accuracy"],
                "delta_intent_accuracy": hi_metrics["intent_accuracy"] - en_metrics["intent_accuracy"],
                "english_entity_f1": en_metrics["entity_f1"],
                "hinglish_entity_f1": hi_metrics["entity_f1"],
                "delta_entity_f1": hi_metrics["entity_f1"] - en_metrics["entity_f1"],
                "english_exact_match_rate": en_metrics["exact_match_rate"],
                "hinglish_exact_match_rate": hi_metrics["exact_match_rate"],
                "delta_exact_match_rate": hi_metrics["exact_match_rate"] - en_metrics["exact_match_rate"],
                "english_schema_failure_rate": en_metrics["schema_failure_rate"],
                "hinglish_schema_failure_rate": hi_metrics["schema_failure_rate"],
                "delta_schema_failure_rate": hi_metrics["schema_failure_rate"] - en_metrics["schema_failure_rate"],
            }
        )

    slice_rows = analyze_slice_breakdowns(results)
    slice_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in slice_rows:
        dimension = row.get("dimension")
        if dimension != "slice_tag":
            continue
        slice_groups[(row["model_name"], row["slice_tag"])][row["prompt_language"]] = row

    per_slice_deltas: list[dict[str, Any]] = []
    for (model_name, slice_tag), by_language in sorted(slice_groups.items()):
        english = by_language.get("english")
        hinglish = by_language.get("hinglish")
        if not english or not hinglish:
            continue
        per_slice_deltas.append(
            {
                "model_name": model_name,
                "slice_tag": slice_tag,
                "english_samples": english["samples"],
                "hinglish_samples": hinglish["samples"],
                "english_exact_match_rate": english["exact_match_rate"],
                "hinglish_exact_match_rate": hinglish["exact_match_rate"],
                "delta_exact_match_rate": hinglish["exact_match_rate"] - english["exact_match_rate"],
                "english_entity_f1": english["entity_f1"],
                "hinglish_entity_f1": hinglish["entity_f1"],
                "delta_entity_f1": hinglish["entity_f1"] - english["entity_f1"],
            }
        )

    regression_candidates = [row for row in per_slice_deltas if row["delta_exact_match_rate"] < 0]
    improvement_candidates = [row for row in per_slice_deltas if row["delta_exact_match_rate"] > 0]
    biggest_regressions = sorted(
        regression_candidates,
        key=lambda row: (row["delta_exact_match_rate"], row["delta_entity_f1"]),
    )[:5]
    biggest_improvements = sorted(
        improvement_candidates,
        key=lambda row: (row["delta_exact_match_rate"], row["delta_entity_f1"]),
        reverse=True,
    )[:5]

    return {
        "overall": overall,
        "per_slice_deltas": per_slice_deltas,
        "biggest_regressions": biggest_regressions,
        "biggest_improvements": biggest_improvements,
    }
