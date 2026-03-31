# Failure Taxonomy (Frozen Buckets)

Use exactly these buckets for model error analysis:

- `temporal_inversion`
- `intent_confusion`
- `sentiment_inversion`
- `entity_drift`
- `hallucination`
- `schema_failure`
- `omission`
- `over_extraction`
- `other`

## Bucket Definitions

- `temporal_inversion`: Time/date meaning is reversed or shifted (past vs future, wrong day relation, etc.).
- `intent_confusion`: Predicted intent differs materially from gold intent.
- `sentiment_inversion`: Sentiment polarity/stance is flipped against source meaning.
- `entity_drift`: Entity content/type is wrong despite extraction attempt.
- `hallucination`: Prediction introduces unsupported content absent from input.
- `schema_failure`: Output is malformed/unparseable or violates schema constraints.
- `omission`: Required intent/entity signal present in source is missing in prediction.
- `over_extraction`: Extra entities/arguments extracted beyond what text supports.
- `other`: Error exists but does not fit above definitions.

## Assignment Rules

1. Assign the **most specific** applicable bucket.
2. Prefer semantic buckets (`intent_confusion`, `entity_drift`) over `other`.
3. If parse/format breaks prevent interpretation, assign `schema_failure`.
4. If both missing and extra entities occur, allow both `omission` and `over_extraction`.
5. For ambiguous utterances, judge against annotation guidelines and avoid speculative penalties.

## Ambiguous Hinglish Consistency

- Transliteration or slang alone is not a failure category.
- If transliteration causes wrong entity value, use `entity_drift`.
- If tense ambiguity was unresolved in gold and model matches gold conservatively, do not count as failure.
