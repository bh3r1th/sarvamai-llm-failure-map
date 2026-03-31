"""Behavioral tests for prompt rendering."""

from __future__ import annotations

from code_switch_failure_map.prompts.render import render_extraction_prompt
from code_switch_failure_map.schemas.taxonomy import PromptLanguage


def test_prompt_render_english_and_hinglish() -> None:
    english = render_extraction_prompt(
        text="book me a cab",
        prompt_language=PromptLanguage.ENGLISH,
        expected_output_schema_description='{"intent": "string|null", "entities": []}',
    )
    hinglish = render_extraction_prompt(
        text="mere liye cab book karo",
        prompt_language=PromptLanguage.HINGLISH,
        expected_output_schema_description='{"intent": "string|null", "entities": []}',
    )

    assert "Extract the user's intent" in english
    assert "Aap ek information extraction system ho" in hinglish


def test_prompt_render_includes_strict_json_contract_wording() -> None:
    rendered = render_extraction_prompt(
        text="weather kal",
        prompt_language=PromptLanguage.ENGLISH,
        expected_output_schema_description='{"intent": "string|null", "entities": []}',
    )

    assert "strict JSON" in rendered
    assert "intent" in rendered
    assert "entities" in rendered
    assert "null" in rendered
    assert "no extra prose" in rendered
