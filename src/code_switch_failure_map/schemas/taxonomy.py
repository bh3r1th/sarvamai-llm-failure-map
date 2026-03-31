"""Shared enums and constrained types for experiment schemas."""

from __future__ import annotations

from enum import Enum


class SourceSplit(str, Enum):
    """Origin split for a sample."""

    RAW = "raw"
    CURATED = "curated"
    GOLDEN = "golden"


class IntentLabel(str, Enum):
    """Frozen intent labels for extraction tasks."""

    REMINDER_CREATE = "reminder_create"
    REMINDER_UPDATE = "reminder_update"
    REMINDER_CANCEL = "reminder_cancel"
    MESSAGE_SEND = "message_send"
    CALL_REQUEST = "call_request"
    MEETING_SCHEDULE = "meeting_schedule"
    NOTE_CREATE = "note_create"
    TASK_CREATE = "task_create"
    INFORMATION_QUERY = "information_query"
    MEDIA_CONTROL = "media_control"
    NAVIGATION_REQUEST = "navigation_request"
    PURCHASE_REQUEST = "purchase_request"
    OTHER = "other"


class EntityType(str, Enum):
    """Frozen entity types for extraction outputs."""

    TIME = "time"
    DATE = "date"
    DATETIME = "datetime"
    DURATION = "duration"
    PERSON = "person"
    CONTACT = "contact"
    PLACE = "place"
    MESSAGE_CONTENT = "message_content"
    TASK_OBJECT = "task_object"
    EVENT_NAME = "event_name"
    MEDIA_NAME = "media_name"
    DESTINATION = "destination"
    QUANTITY = "quantity"
    SENTIMENT_TARGET = "sentiment_target"
    OTHER = "other"


class SliceTag(str, Enum):
    """Supported analysis slice tags for this experiment."""

    CODE_SWITCHING = "code_switching"
    TRANSLITERATION_NOISE = "transliteration_noise"
    AMBIGUITY = "ambiguity"
    TEMPORAL_REFERENCE = "temporal_reference"
    SENTIMENT_LOAD = "sentiment_load"
    SHORT_UTTERANCE = "short_utterance"
    LONG_CONTEXT = "long_context"
    ADVERSARIAL = "adversarial"
    PROMPT_LANGUAGE_EN = "prompt_language_en"
    PROMPT_LANGUAGE_HINGLISH = "prompt_language_hinglish"


class PromptLanguage(str, Enum):
    """Prompt language variant used for inference."""

    HINGLISH = "hinglish"
    ENGLISH = "english"


class FailureCategory(str, Enum):
    """Frozen failure buckets for error analysis."""

    TEMPORAL_INVERSION = "temporal_inversion"
    INTENT_CONFUSION = "intent_confusion"
    SENTIMENT_INVERSION = "sentiment_inversion"
    ENTITY_DRIFT = "entity_drift"
    HALLUCINATION = "hallucination"
    SCHEMA_FAILURE = "schema_failure"
    OMISSION = "omission"
    OVER_EXTRACTION = "over_extraction"
    OTHER = "other"
