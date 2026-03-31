"""Validation helpers for dataset records."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import ValidationError

from code_switch_failure_map.schemas.sample import EntityMention, SampleRecord
from code_switch_failure_map.schemas.taxonomy import PromptLanguage, SliceTag, SourceSplit

ALLOWED_PROMPT_VARIANTS: set[str] = {"baseline_v1", "english_v1", "hinglish_v1"}


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation issue detected in dataset rows."""

    sample_id: str
    message: str


def normalized_text_key(text: str) -> str:
    """Canonical text key used to detect obvious near-duplicate samples."""
    compact = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def _entities_valid(entities: list[EntityMention]) -> bool:
    return all(isinstance(entity, EntityMention) for entity in entities)


def validate_records(records: list[SampleRecord], allowed_prompt_variants: set[str] | None = None) -> list[ValidationIssue]:
    """Validate records with cross-row and semantic checks."""
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    seen_normalized_text: dict[str, str] = {}
    variants = allowed_prompt_variants or ALLOWED_PROMPT_VARIANTS

    for record in records:
        if record.sample_id in seen_ids:
            issues.append(ValidationIssue(sample_id=record.sample_id, message="duplicate sample_id"))
        seen_ids.add(record.sample_id)

        if not record.text.strip():
            issues.append(ValidationIssue(sample_id=record.sample_id, message="text must be non-empty"))

        if not record.gold_intent.strip():
            issues.append(ValidationIssue(sample_id=record.sample_id, message="gold_intent must be present"))

        if not _entities_valid(record.gold_entities):
            issues.append(ValidationIssue(sample_id=record.sample_id, message="gold_entities structure is invalid"))

        if record.prompt_variant not in variants:
            issues.append(ValidationIssue(sample_id=record.sample_id, message=f"prompt_variant '{record.prompt_variant}' is not allowed"))

        if record.source_split not in set(SourceSplit):
            issues.append(ValidationIssue(sample_id=record.sample_id, message=f"source_split '{record.source_split}' is not allowed"))

        if record.prompt_language not in set(PromptLanguage):
            issues.append(ValidationIssue(sample_id=record.sample_id, message=f"prompt_language '{record.prompt_language}' is not allowed"))

        for tag in record.slice_tags:
            if tag not in set(SliceTag):
                issues.append(ValidationIssue(sample_id=record.sample_id, message=f"slice_tag '{tag}' is not allowed"))

        normalization_source = record.normalized_text if record.normalized_text else record.text
        normalized_key = normalized_text_key(normalization_source)
        if normalized_key:
            first_seen_id = seen_normalized_text.get(normalized_key)
            if first_seen_id and first_seen_id != record.sample_id:
                issues.append(
                    ValidationIssue(
                        sample_id=record.sample_id,
                        message=f"normalized text duplicates sample '{first_seen_id}'",
                    )
                )
            else:
                seen_normalized_text[normalized_key] = record.sample_id

    return issues


def validate_raw_rows(rows: list[dict[str, object]]) -> tuple[list[SampleRecord], list[ValidationIssue]]:
    """Parse raw dictionaries to records and report schema-level issues deterministically."""
    records: list[SampleRecord] = []
    issues: list[ValidationIssue] = []

    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", f"<row_{index}>"))
        try:
            records.append(SampleRecord.model_validate(row))
        except ValidationError as exc:
            for error in exc.errors():
                location = ".".join(str(part) for part in error.get("loc", []))
                message = error.get("msg", "unknown validation error")
                issues.append(ValidationIssue(sample_id=sample_id, message=f"{location}: {message}"))

    issues.extend(validate_records(records))
    return records, issues


def assert_valid_records(records: list[SampleRecord], allowed_prompt_variants: set[str] | None = None) -> None:
    """Raise ValueError when any dataset validation issues are found."""
    issues = validate_records(records, allowed_prompt_variants=allowed_prompt_variants)
    if not issues:
        return

    summary = "\n".join(f"- {issue.sample_id}: {issue.message}" for issue in issues)
    raise ValueError(f"Dataset validation failed with {len(issues)} issue(s):\n{summary}")
