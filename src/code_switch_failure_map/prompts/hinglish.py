"""Hinglish prompt template for intent/entity extraction."""

from __future__ import annotations

HINGLISH_EXTRACTION_TEMPLATE = """Input text se user ka intent aur entities extract karo.
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
Neeche diya gaya schema hi use karo.
Agar koi entity nahi hai to [] do.
Value unknown ho to null do.
Entities hallucinate mat karo.
Text se bahar infer mat karo, sirf trivial normalization allowed hai.

Output schema:
{schema}

Input text:
{text}
"""
