"""English prompt template for intent/entity extraction."""

from __future__ import annotations

ENGLISH_EXTRACTION_TEMPLATE = """You are an information extraction system.
Extract the user's intent and entities from the input text.
Return strict JSON only (no markdown, no commentary, no extra prose).
If intent is unknown, set intent to null.
If an entity is unknown or missing, set it to null.
Expected output schema:
{schema}
Input text:
{text}
"""
