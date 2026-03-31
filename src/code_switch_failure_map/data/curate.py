"""Helpers for selecting and summarizing curated dataset slices."""

from __future__ import annotations

from collections import Counter

from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import SliceTag


def filter_by_slice_tags(records: list[SampleRecord], required_tags: set[SliceTag]) -> list[SampleRecord]:
    """Return records that contain all required slice tags."""
    return [record for record in records if required_tags.issubset(record.slice_tags)]


def select_adversarial_candidates(records: list[SampleRecord]) -> list[SampleRecord]:
    """Heuristic adversarial candidate selector from metadata flags."""
    return [
        record
        for record in records
        if record.metadata_flags.ambiguity
        or record.metadata_flags.transliteration_noise
        or (record.metadata_flags.code_switching and len(record.text.split()) <= 6)
    ]


def summary_counts_by_slice(records: list[SampleRecord]) -> dict[str, int]:
    """Compute counts grouped by each slice tag value."""
    counter: Counter[str] = Counter()
    for record in records:
        for tag in record.slice_tags:
            counter[tag.value] += 1
    return dict(sorted(counter.items()))
