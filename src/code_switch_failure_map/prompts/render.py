"""Prompt renderer for language-specific extraction templates."""

from __future__ import annotations

from code_switch_failure_map.prompts.english import ENGLISH_EXTRACTION_TEMPLATE
from code_switch_failure_map.prompts.hinglish import HINGLISH_EXTRACTION_TEMPLATE
from code_switch_failure_map.schemas.taxonomy import PromptLanguage


_DEFAULT_SCHEMA_DESCRIPTION = '{"intent": "string | null", "entities": [{"label": "string", "value": "string | null"}]}'


def render_extraction_prompt(
    text: str,
    prompt_language: PromptLanguage,
    expected_output_schema_description: str = _DEFAULT_SCHEMA_DESCRIPTION,
) -> str:
    """Render prompt instructions for the selected prompt language."""
    template = {
        PromptLanguage.ENGLISH: ENGLISH_EXTRACTION_TEMPLATE,
        PromptLanguage.HINGLISH: HINGLISH_EXTRACTION_TEMPLATE,
    }[prompt_language]

    return template.format(text=text.strip(), schema=expected_output_schema_description.strip())
