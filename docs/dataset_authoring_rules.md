# Hinglish Raw Dataset Authoring Rules

This guide defines how to author high-quality **raw** examples for structured extraction tasks.

## 1. Language and Style

- Prefer natural Romanized Hindi (e.g., `kal`, `subah`, `bhej do`, `yaad dila`).
- Mix in English naturally where real users do (`set reminder`, `schedule`, `ping`, `mute`).
- Keep varied spellings realistic (`baje`, `bje`, `bjhe`; `message`, `msg`, `txt`).
- Allow imperfect grammar and shorthand phrasing; avoid polished textbook sentences.

## 2. Real-World Noise to Include

- Transliteration variation: missing vowels, phonetic spellings, and abbreviations.
- Device/chat shorthand: `pls`, `tmrw`, `mtg`, `krdo`, `jaldi`.
- Partial sentences and command fragments.
- Ambiguous temporal references (`shaam tak`, `thodi der me`, `usual time`).
- Sentiment-heavy phrasing when relevant (`mood off`, `panic na kare`).

## 3. Diversity Expectations

- Avoid template churn (same sentence with only name/time swapped).
- Vary intents, entity combinations, sentence lengths, and difficulty.
- Include easy, noisy, code-switched, ambiguous, and adversarial examples.
- Keep entities grounded in text; avoid speculative labels not supported by utterance.

## 4. Annotation Consistency

- `source_split` for seed authoring must be `raw`.
- `gold_intent` must use only frozen intent labels from taxonomy.
- `gold_entities[].type` must use only frozen entity labels.
- `slice_tags` must come from the frozen slice tag set.
- If metadata flag is true, include the corresponding slice tag:
  - `code_switching` -> `code_switching`
  - `transliteration_noise` -> `transliteration_noise`
  - `ambiguity` -> `ambiguity`

## 5. Quality Bar

- Each sample should feel like a message someone would genuinely send or say.
- Prefer semantic variety over micro-variations.
- If two samples normalize to near-identical text, keep only one.
- Preserve hard/adversarial samples that stress extraction logic without becoming gibberish.
