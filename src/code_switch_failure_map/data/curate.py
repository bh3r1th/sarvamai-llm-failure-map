"""Helpers for selecting and summarizing curated dataset slices."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from code_switch_failure_map.data.load import dumpable_records, load_dataset
from code_switch_failure_map.data.split import GoldenSelectionResult, select_golden_set
from code_switch_failure_map.data.validate import normalized_text_key
from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.schemas.taxonomy import SliceTag
from code_switch_failure_map.utils.io import write_jsonl


def select_candidates_by_slice(records: list[SampleRecord], required_tags: set[SliceTag]) -> list[SampleRecord]:
    """Return records that contain all required slice tags."""
    return [record for record in records if required_tags.issubset(record.slice_tags)]


def filter_by_slice_tags(records: list[SampleRecord], required_tags: set[SliceTag]) -> list[SampleRecord]:
    """Backward-compatible alias for slice filtering."""
    return select_candidates_by_slice(records, required_tags)


def select_adversarial_candidates(records: list[SampleRecord]) -> list[SampleRecord]:
    """Heuristic adversarial candidate selector from metadata flags."""
    return [
        record
        for record in records
        if record.metadata_flags.ambiguity
        or record.metadata_flags.transliteration_noise
        or (record.metadata_flags.code_switching and len(record.text.split()) <= 6)
    ]


def count_by_intent(records: list[SampleRecord]) -> dict[str, int]:
    """Compute counts grouped by intent label."""
    counter: Counter[str] = Counter(record.gold_intent.value for record in records)
    return dict(sorted(counter.items()))


def count_by_slice(records: list[SampleRecord]) -> dict[str, int]:
    """Compute counts grouped by each slice tag value."""
    counter: Counter[str] = Counter()
    for record in records:
        for tag in record.slice_tags:
            counter[tag.value] += 1
    return dict(sorted(counter.items()))


def summary_counts_by_slice(records: list[SampleRecord]) -> dict[str, int]:
    """Backward-compatible alias for slice summary counts."""
    return count_by_slice(records)


def export_subset_by_ids(records: list[SampleRecord], sample_ids: list[str]) -> list[SampleRecord]:
    """Return subset in the same order as ``sample_ids`` while skipping missing ids."""
    by_id = {record.sample_id: record for record in records}
    return [by_id[sample_id] for sample_id in sample_ids if sample_id in by_id]


def identify_low_diversity_samples(records: list[SampleRecord], min_group_size: int = 2) -> dict[str, list[str]]:
    """Group sample IDs by identical canonical text keys to flag low-diversity variants."""
    grouped: dict[str, list[str]] = {}
    for record in records:
        key = normalized_text_key(record.normalized_text or record.text)
        grouped.setdefault(key, []).append(record.sample_id)

    return {key: ids for key, ids in grouped.items() if len(ids) >= min_group_size}


def build_golden_files(
    source_path: str | Path = "data/raw/seed_hinglish_samples.jsonl",
    candidates_path: str | Path = "data/golden/golden_candidates.jsonl",
    golden_set_path: str | Path = "data/golden/golden_set_v1.jsonl",
    golden_size: int = 50,
) -> GoldenSelectionResult:
    """Build deterministic golden candidate + final set files and print a concise summary."""
    records = load_dataset(source_path)
    result = select_golden_set(records, size=golden_size)

    write_jsonl(candidates_path, dumpable_records(result.candidates))
    write_jsonl(golden_set_path, dumpable_records(result.golden_set))

    print(f"golden set summary: size={len(result.golden_set)} candidates={len(result.candidates)}")
    print(f"intent distribution: {result.intent_distribution}")
    print(f"slice distribution: {result.slice_distribution}")
    if result.imbalance_warnings:
        print(f"imbalance warnings: {result.imbalance_warnings}")
    else:
        print("imbalance warnings: none")

    return result
