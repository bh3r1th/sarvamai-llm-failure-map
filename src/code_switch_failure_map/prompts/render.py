"""Prompt renderer for language-specific extraction templates."""

from __future__ import annotations

from collections.abc import Iterable

from code_switch_failure_map.prompts.english import ENGLISH_EXTRACTION_TEMPLATE
from code_switch_failure_map.prompts.hinglish import HINGLISH_EXTRACTION_TEMPLATE
from code_switch_failure_map.schemas.taxonomy import EntityType
from code_switch_failure_map.schemas.taxonomy import IntentLabel
from code_switch_failure_map.schemas.taxonomy import PromptLanguage


DEFAULT_OUTPUT_SCHEMA_DESCRIPTION = (
    '{"intent": "<one allowed intent>", "entities": [{"label": "<one allowed entity label>", "value": "<string or null>"}]}'
)


def render_extraction_prompt(
    text: str,
    prompt_language: PromptLanguage,
    expected_output_schema_description: str = DEFAULT_OUTPUT_SCHEMA_DESCRIPTION,
) -> str:
    """Render prompt instructions for the selected prompt language."""
    template = {
        PromptLanguage.ENGLISH: ENGLISH_EXTRACTION_TEMPLATE,
        PromptLanguage.HINGLISH: HINGLISH_EXTRACTION_TEMPLATE,
    }[prompt_language]

    return template.format(
        text=text.strip(),
        schema=expected_output_schema_description.strip(),
        allowed_intents=_format_allowed_values(label.value for label in IntentLabel),
        allowed_entity_labels=_format_allowed_values(label.value for label in EntityType),
    ).strip()


def _format_allowed_values(values: Iterable[str]) -> str:
    entries = [f"- {value}" for value in values]
    return "\n".join(entries)
