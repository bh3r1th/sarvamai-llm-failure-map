"""Validation helpers for dataset records."""

from __future__ import annotations

from dataclasses import dataclass

from code_switch_failure_map.schemas.sample import EntityMention, SampleRecord
from code_switch_failure_map.schemas.taxonomy import PromptLanguage, SourceSplit

ALLOWED_PROMPT_VARIANTS: set[str] = {"baseline_v1", "english_v1", "hinglish_v1"}


@dataclass(frozen=True)
class ValidationIssue:
    """Single validation issue detected in dataset rows."""

    sample_id: str
    message: str


def _entities_valid(entities: list[EntityMention] | dict[str, str | list[str]]) -> bool:
    if isinstance(entities, list):
        return all(isinstance(entity, EntityMention) for entity in entities)

    if not isinstance(entities, dict):
        return False

    for value in entities.values():
        if isinstance(value, str):
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            continue
        return False
    return True


def validate_records(records: list[SampleRecord], allowed_prompt_variants: set[str] | None = None) -> list[ValidationIssue]:
    """Validate records with cross-row and semantic checks."""
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
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

    return issues


def assert_valid_records(records: list[SampleRecord], allowed_prompt_variants: set[str] | None = None) -> None:
    """Raise ValueError when any dataset validation issues are found."""
    issues = validate_records(records, allowed_prompt_variants=allowed_prompt_variants)
    if not issues:
        return

    summary = "\n".join(f"- {issue.sample_id}: {issue.message}" for issue in issues)
    raise ValueError(f"Dataset validation failed with {len(issues)} issue(s):\n{summary}")
