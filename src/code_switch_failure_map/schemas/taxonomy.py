"""Shared enums and constrained types for experiment schemas."""

from __future__ import annotations

from enum import Enum


class SourceSplit(str, Enum):
    """Origin split for a sample."""

    RAW = "raw"
    CURATED = "curated"
    GOLDEN = "golden"


class SliceTag(str, Enum):
    """Supported analysis slice tags for this experiment."""

    CODE_SWITCHING = "code_switching"
    TRANSLITERATION_NOISE = "transliteration_noise"
    AMBIGUITY = "ambiguity"
    PROMPT_LANGUAGE = "prompt_language"


class PromptLanguage(str, Enum):
    """Prompt language variant used for inference."""

    HINGLISH = "hinglish"
    ENGLISH = "english"


class FailureCategory(str, Enum):
    """Failure buckets for error analysis."""

    TEMPORAL_INVERSION = "temporal_inversion"
    INTENT_CONFUSION = "intent_confusion"
    SENTIMENT_INVERSION = "sentiment_inversion"
    ENTITY_DRIFT = "entity_drift"
    HALLUCINATION = "hallucination"
    SCHEMA_FAILURE = "schema_failure"
    OTHER = "other"
