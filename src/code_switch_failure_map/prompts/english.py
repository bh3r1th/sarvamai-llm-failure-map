"""English prompt template for intent/entity extraction."""

from __future__ import annotations

ENGLISH_EXTRACTION_TEMPLATE = """Extract the user's intent and entities from the input text.
Return ONLY a valid JSON object.
Do NOT include any explanation, reasoning, markdown, or extra text.
Do NOT repeat instructions.
Do NOT wrap in code fences.
Do NOT repeat or paraphrase the input.
Do NOT restate instructions.
Start directly with JSON.
First character of response must be '{{'.
If output is not valid JSON, the response is invalid.
Any deviation from JSON format is considered incorrect.
Strict compliance required.

Allowed intents:
{allowed_intents}

Allowed entity labels:
{allowed_entity_labels}

You MUST choose exactly one intent from the list above.
If unsure, choose "other".
Do NOT create new intent labels.
Use only the schema shown below.
If no entities are present, return [].
Use null only when a value is unknown.
Do NOT hallucinate entities.
Do NOT infer beyond the text unless the normalization is trivial.

Output schema:
{schema}

Input text:
{text}
"""
