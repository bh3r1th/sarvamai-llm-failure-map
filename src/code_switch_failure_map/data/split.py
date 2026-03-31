"""Deterministic utilities for assigning curated/golden data buckets."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import EntityType, SliceTag, SourceSplit


BUCKET_CURATED = "curated"
BUCKET_GOLDEN = "golden"


@dataclass(frozen=True)
class GoldenSelectionResult:
    """Artifacts emitted by deterministic golden-set selection."""

    candidates: list[SampleRecord]
    golden_set: list[SampleRecord]
    intent_distribution: dict[str, int]
    slice_distribution: dict[str, int]
    imbalance_warnings: list[str]


def assign_bucket(sample_id: str, golden_ratio: float = 0.2) -> str:
    """Assign a stable bucket based on sample id hashing."""
    if not 0.0 < golden_ratio < 1.0:
        raise ValueError("golden_ratio must be between 0 and 1")

    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    slot = int(digest[:8], 16) / 0xFFFFFFFF
    return BUCKET_GOLDEN if slot < golden_ratio else BUCKET_CURATED


def split_records(records: list[SampleRecord], golden_ratio: float = 0.2) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Split records into curated and golden sets deterministically."""
    curated: list[SampleRecord] = []
    golden: list[SampleRecord] = []
    for record in records:
        bucket = assign_bucket(record.sample_id, golden_ratio=golden_ratio)
        if bucket == BUCKET_GOLDEN:
            golden.append(record)
        else:
            curated.append(record)
    return curated, golden


def _stable_tie_breaker(sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _code_switch_density_score(text: str) -> int:
    tokens = [token for token in text.split() if token]
    if not tokens:
        return 0
    latin_tokens = sum(1 for token in tokens if any("a" <= char.lower() <= "z" for char in token))
    ratio = latin_tokens / len(tokens)
    if 0.2 <= ratio <= 0.8:
        return 4
    if 0.1 <= ratio < 0.2 or 0.8 < ratio <= 0.9:
        return 2
    return 0


def _ambiguity_score(record: SampleRecord) -> int:
    score = 0
    if record.metadata_flags.ambiguity or SliceTag.AMBIGUITY in record.slice_tags:
        score += 7
    if len(record.gold_entities) == 0:
        score += 1
    return score


def _difficulty_score(record: SampleRecord) -> int:
    score = 0
    tags = record.slice_tags

    if SliceTag.ADVERSARIAL in tags:
        score += 9
    if record.metadata_flags.transliteration_noise or SliceTag.TRANSLITERATION_NOISE in tags:
        score += 6
    if SliceTag.TEMPORAL_REFERENCE in tags:
        score += 5
    if record.metadata_flags.code_switching or SliceTag.CODE_SWITCHING in tags:
        score += 5

    score += _ambiguity_score(record)
    score += _code_switch_density_score(record.text)

    token_count = len(record.text.split())
    if token_count <= 4:
        score += 2
    if token_count >= 10:
        score += 2

    if len(record.gold_entities) >= 2:
        score += 2

    return score


def _sorted_by_priority(records: list[SampleRecord]) -> list[SampleRecord]:
    return sorted(
        records,
        key=lambda record: (
            -_difficulty_score(record),
            -len(record.slice_tags),
            -len(record.gold_entities),
            -len(record.text),
            _stable_tie_breaker(record.sample_id),
            record.sample_id,
        ),
    )


def _with_golden_split(records: list[SampleRecord]) -> list[SampleRecord]:
    return [record.model_copy(update={"source_split": SourceSplit.GOLDEN}) for record in records]


def _summarize_slices(records: list[SampleRecord]) -> dict[str, int]:
    slice_counts: Counter[str] = Counter()
    for record in records:
        for tag in record.slice_tags:
            slice_counts[tag.value] += 1
    return dict(sorted(slice_counts.items()))


def _collect_entity_types(record: SampleRecord) -> set[EntityType]:
    return {entity.type for entity in record.gold_entities}


def _has_minimum_diversity(intent_counts: Counter[str], entity_types: set[EntityType]) -> bool:
    return len(intent_counts) >= 8 and len(entity_types) >= 6


def select_golden_set(
    records: list[SampleRecord],
    size: int = 50,
    candidate_size: int = 80,
    per_intent_cap: int = 6,
) -> GoldenSelectionResult:
    """Select deterministic golden candidates + final set with adversarial diversity bias."""
    if size <= 0:
        raise ValueError("size must be > 0")
    if len(records) < size:
        raise ValueError(f"need at least {size} records, found {len(records)}")

    ranked = _sorted_by_priority(records)
    candidates = _with_golden_split(ranked[: min(len(ranked), candidate_size)])

    selected: list[SampleRecord] = []
    intent_counts: Counter[str] = Counter()
    slice_counts: Counter[str] = Counter()
    entity_types: set[EntityType] = set()

    must_have_tags = {
        SliceTag.ADVERSARIAL,
        SliceTag.AMBIGUITY,
        SliceTag.TRANSLITERATION_NOISE,
        SliceTag.TEMPORAL_REFERENCE,
        SliceTag.CODE_SWITCHING,
    }

    # First pass: enforce strong coverage of hard slices.
    for tag in sorted(must_have_tags, key=lambda value: value.value):
        for record in ranked:
            if len(selected) >= size:
                break
            if record.sample_id in {item.sample_id for item in selected}:
                continue
            if tag not in record.slice_tags:
                continue
            intent_key = record.gold_intent.value
            if intent_counts[intent_key] >= per_intent_cap:
                continue
            selected.append(record)
            intent_counts[intent_key] += 1
            for slice_tag in record.slice_tags:
                slice_counts[slice_tag.value] += 1
            entity_types.update(_collect_entity_types(record))
            break

    # Second pass: greedy fill by ranked difficulty with intent caps.
    for record in ranked:
        if len(selected) >= size:
            break
        if record.sample_id in {item.sample_id for item in selected}:
            continue
        intent_key = record.gold_intent.value
        if intent_counts[intent_key] >= per_intent_cap:
            continue
        selected.append(record)
        intent_counts[intent_key] += 1
        for slice_tag in record.slice_tags:
            slice_counts[slice_tag.value] += 1
        entity_types.update(_collect_entity_types(record))

    # Final pass: if caps blocked completion, fill deterministically without cap.
    if len(selected) < size:
        for record in ranked:
            if len(selected) >= size:
                break
            if record.sample_id in {item.sample_id for item in selected}:
                continue
            selected.append(record)
            intent_counts[record.gold_intent.value] += 1
            for slice_tag in record.slice_tags:
                slice_counts[slice_tag.value] += 1
            entity_types.update(_collect_entity_types(record))

    if len(selected) != size:
        raise ValueError(f"unable to select exactly {size} samples; selected {len(selected)}")

    selected = _with_golden_split(_sorted_by_priority(selected))
    intent_distribution = dict(sorted(intent_counts.items()))
    slice_distribution = dict(sorted(slice_counts.items()))

    warnings: list[str] = []
    if max(intent_counts.values()) > 8:
        warnings.append("single-intent concentration is high (>8 samples in one intent)")
    if not _has_minimum_diversity(intent_counts, entity_types):
        warnings.append("minimum diversity threshold not met for intents/entity types")
    if slice_counts[SliceTag.SHORT_UTTERANCE.value] == 0:
        warnings.append("no short_utterance sample present")
    if sum(1 for record in selected if len(record.text.split()) >= 10) < 4:
        warnings.append("few long utterances (>=10 tokens) in golden set")

    return GoldenSelectionResult(
        candidates=candidates,
        golden_set=selected,
        intent_distribution=intent_distribution,
        slice_distribution=slice_distribution,
        imbalance_warnings=warnings,
    )
