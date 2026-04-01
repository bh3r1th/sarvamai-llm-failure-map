"""Slice-level comparative analysis helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from code_switch_failure_map.schemas.evaluation import EvaluationResult
from code_switch_failure_map.schemas.taxonomy import SliceTag

LANGUAGE_FAMILY_TAGS: tuple[SliceTag, ...] = (
    SliceTag.HINGLISH,
    SliceTag.TELUGU_ENGLISH,
)


def _mean(values: list[float]) -> float:
    return 0.0 if not values else sum(values) / len(values)


def _metric_row(rows: list[EvaluationResult], *, dimension: str, value: str) -> dict[str, Any]:
    return {
        "dimension": dimension,
        dimension: value,
        "samples": len(rows),
        "intent_accuracy": _mean([1.0 if row.intent_correct else 0.0 for row in rows]),
        "entity_precision": _mean([row.entity_precision for row in rows]),
        "entity_recall": _mean([row.entity_recall for row in rows]),
        "entity_f1": _mean([row.entity_f1 for row in rows]),
        "exact_match_rate": _mean([1.0 if row.overall_exact_match else 0.0 for row in rows]),
        "schema_failure_rate": _mean([1.0 if row.schema_failure else 0.0 for row in rows]),
    }


def language_family_values(slice_tags: set[SliceTag]) -> list[str]:
    """Return explicit language-family tags only; no silent inference."""
    return sorted(tag.value for tag in slice_tags if tag in LANGUAGE_FAMILY_TAGS)


def analyze_slice_breakdowns(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    """Summarize evaluation metrics by required slice dimensions."""
    groups: dict[tuple[str, str, str, str], list[EvaluationResult]] = defaultdict(list)

    for row in results:
        for slice_tag in row.slice_tags:
            groups[("slice_tag", slice_tag.value, row.model_name, row.prompt_language.value)].append(row)
        for language_family in language_family_values(row.slice_tags):
            groups[("language_family", language_family, row.model_name, row.prompt_language.value)].append(row)
        if row.gold_intent:
            groups[("intent_type", row.gold_intent, row.model_name, row.prompt_language.value)].append(row)

    summary: list[dict[str, Any]] = []
    for (dimension, value, model_name, prompt_language), rows in sorted(groups.items()):
        payload = _metric_row(rows, dimension=dimension, value=value)
        payload["model_name"] = model_name
        payload["prompt_language"] = prompt_language
        summary.append(payload)
    return summary


def slice_focus_summary(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    """Return only the blog-relevant slice tags requested in the prompt."""
    required = {
        "code_switching",
        "hinglish",
        "telugu_english",
        "transliteration_noise",
        "ambiguity",
        "adversarial",
        "prompt_language_en",
        "prompt_language_hinglish",
    }
    rows = analyze_slice_breakdowns(results)
    return [
        row
        for row in rows
        if (row.get("dimension") == "slice_tag" and row.get("slice_tag") in required)
        or row.get("dimension") == "intent_type"
    ]
