# Dataset Specification (Frozen Contract)

This document defines the **locked label space and record schema** for the Hinglish failure-map experiment. Do not run model experiments against data that violates this contract.

## 1) Record Schema

Each sample must map to `SampleRecord` in `src/code_switch_failure_map/schemas/sample.py`.

Required fields:
- `sample_id` (string, non-empty)
- `source_split` (`raw | curated | golden`)
- `text` (string, non-empty)
- `gold_intent` (one value from the frozen intent enum)
- `gold_entities` (list of `EntityMention`)
- `slice_tags` (set of frozen slice tags)
- `prompt_variant` (string, non-empty)
- `prompt_language` (`english | hinglish`)

Optional fields:
- `normalized_text` (string or null)
- `metadata_flags` (booleans; default false)

## 2) Frozen Label Spaces

### 2.1 Intents
- `reminder_create`
- `reminder_update`
- `reminder_cancel`
- `message_send`
- `call_request`
- `meeting_schedule`
- `note_create`
- `task_create`
- `information_query`
- `media_control`
- `navigation_request`
- `purchase_request`
- `other`

### 2.2 Entity Types
- `time`
- `date`
- `datetime`
- `duration`
- `person`
- `contact`
- `place`
- `message_content`
- `task_object`
- `event_name`
- `media_name`
- `destination`
- `quantity`
- `sentiment_target`
- `other`

### 2.3 Slice Tags
- `code_switching`
- `transliteration_noise`
- `ambiguity`
- `temporal_reference`
- `sentiment_load`
- `short_utterance`
- `long_context`
- `adversarial`
- `prompt_language_en`
- `prompt_language_hinglish`

### 2.4 Failure Categories
- `temporal_inversion`
- `intent_confusion`
- `sentiment_inversion`
- `entity_drift`
- `hallucination`
- `schema_failure`
- `omission`
- `over_extraction`
- `other`

## 3) Entity Value Structure

Each entity is represented as:
- `type`: frozen enum above
- `text`: exact extracted span text (required)
- `normalized_value`: canonicalized value (optional, nullable)
- `confidence`: optional float in `[0.0, 1.0]`
- `start_char`, `end_char`: optional char offsets; if one exists, both must exist, and `end_char > start_char`

## 4) Invariants for Stable Evaluation

- `gold_entities` must be a **list**, not dict-shaped free-form structures.
- `prompt_language` must match language slice tags:
  - `english` -> include `prompt_language_en`
  - `hinglish` -> include `prompt_language_hinglish`
- The opposite prompt-language tag must not be present.
- If metadata flags are true, corresponding tags must be present:
  - `code_switching` flag -> `code_switching` tag
  - `transliteration_noise` flag -> `transliteration_noise` tag
  - `ambiguity` flag -> `ambiguity` tag

## 5) Null and Inference Rules

- Use `null` when normalization is unknown.
- Do not infer beyond utterance text unless normalization is trivial and deterministic.
- For unresolved ambiguity, prefer conservative labels and include `ambiguity` slice tag.
