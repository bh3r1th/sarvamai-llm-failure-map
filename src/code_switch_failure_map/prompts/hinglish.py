"""Hinglish prompt template for intent/entity extraction."""

from __future__ import annotations

HINGLISH_EXTRACTION_TEMPLATE = """Aap ek information extraction system ho.
Input text se user ka intent aur entities extract karo.
Sirf strict JSON do (no markdown, no commentary, no extra prose).
Agar intent clear nahi hai to intent = null do.
Agar entity unknown/missing ho to null do.
Expected output schema:
{schema}
Input text:
{text}
"""
