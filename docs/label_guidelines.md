# Label Guidelines

This guide is annotation-facing and uses the frozen schema contract.

## Intent Definitions

- `reminder_create`: User asks to set a new reminder.
- `reminder_update`: User modifies an existing reminder.
- `reminder_cancel`: User cancels/deletes a reminder.
- `message_send`: User asks to send or draft a message.
- `call_request`: User asks to place/call someone.
- `meeting_schedule`: User asks to schedule/reschedule a meeting.
- `note_create`: User asks to write/save a note.
- `task_create`: User asks to create a to-do/task item.
- `information_query`: User asks for information (fact/status/weather/general lookup intent).
- `media_control`: User controls playback/media operations.
- `navigation_request`: User asks for directions/navigation.
- `purchase_request`: User asks to buy/order/reorder something.
- `other`: None of the above applies confidently.

## Entity Annotation Rules

Each extracted entity must include:
- `type`
- `text`

Optional fields:
- `normalized_value` only when normalization is straightforward.
- `confidence` when your annotation workflow captures confidence.
- `start_char` and `end_char` only if source offsets are available.

### Valid Extraction Principles

1. Extract text-supported spans only.
2. Keep span minimal but complete.
3. Do not add implied entities unless explicitly stated.
4. Use `other` type only when no specific entity type applies.

## Ambiguous Hinglish Handling

- If utterance mixes intent cues (e.g., "kal call ya msg kar dena"), choose the best-supported intent and add `ambiguity` slice tag.
- If temporal words are underspecified (e.g., "baad mein", "kal" without context), keep raw text in entity and avoid over-normalizing.
- For slang or mixed-language shorthand, annotate literal text and normalize only if deterministic.

## Edge-Case Rules

- **Null when unknown:** set `normalized_value: null` when unsure.
- **No over-inference:** do not resolve hidden assumptions from world knowledge.
- **Mixed-language slang:** preserve user wording in `text`; avoid semantic rewriting.
- **Transliteration variance:** treat variants (e.g., "kal", "kal", "kl") as same concept only when context is clear.
- **Tense ambiguity:** when tense does not fix time, prefer `date`/`time` text with null normalization.
