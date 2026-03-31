# Golden Set Rules (v1)

A sample is considered **golden** only when all conditions below are true:

1. **High label confidence**
   - Intent is unambiguous after close reading.
   - Entity spans/types are internally consistent with the utterance.
2. **Hard but answerable**
   - The sample may include ambiguity, transliteration noise, temporal reasoning, or adversarial phrasing.
   - There is still a clear best gold label and extractable target entities.
3. **Useful for failure analysis**
   - The sample stresses known weak spots (intent confusion, entity drift, code-switch parsing, etc.).
   - At least one explicit difficulty reason can be articulated.
4. **Not a trivial duplicate**
   - Avoid near-copy paraphrases that add no new phenomenon.
5. **Not impossible / no-ground-truth**
   - Exclude requests that lack a determinate answer or are annotation-invalid.

## Selection priorities (deterministic)

The ranking logic prefers samples with:
- adversarial slice tags
- ambiguity
- transliteration noise
- temporal expressions
- code-switching density
- richer entity structure
- short and long utterance extremes

## Balance targets

- Final golden set size is exactly **50**.
- No single intent should dominate the set.
- Include broad slice coverage and multiple entity types.
- Ensure both short and long utterances are represented.
