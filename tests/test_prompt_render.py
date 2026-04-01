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

    assert "Extract the user's intent and entities" in english
    assert "Input text se user ka intent aur entities extract karo" in hinglish
    assert "book me a cab" in english
    assert "mere liye cab book karo" in hinglish


def test_prompt_render_includes_strict_json_contract_wording() -> None:
    rendered = render_extraction_prompt(
        text="weather kal",
        prompt_language=PromptLanguage.ENGLISH,
        expected_output_schema_description='{"intent": "string|null", "entities": []}',
    )

    assert "Return ONLY a valid JSON object" in rendered
    assert "Allowed intents" in rendered
    assert "Do NOT include any explanation" in rendered
    assert "First character of response must be '{'" in rendered
    assert "Do NOT create new intent labels" in rendered
    assert "- reminder_create" in rendered
    assert "- purchase_request" in rendered
    assert "- other" in rendered


def test_prompt_render_lists_allowed_entity_labels_and_has_no_extra_padding() -> None:
    rendered = render_extraction_prompt(
        text="kal mom ko call karna",
        prompt_language=PromptLanguage.HINGLISH,
    )

    assert rendered.startswith("Input text se user ka intent aur entities extract karo.")
    assert "Allowed entity labels" in rendered
    assert "- time" in rendered
    assert "- destination" in rendered
    assert "- other" in rendered
    assert "Start directly with JSON." in rendered
