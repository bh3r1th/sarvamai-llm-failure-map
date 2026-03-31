"""Behavioral tests for dataset I/O, validation, curation, and splitting."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from code_switch_failure_map.data.curate import count_by_slice, export_subset_by_ids
from code_switch_failure_map.data.load import load_dataset
from code_switch_failure_map.data.split import select_golden_set, split_records
from code_switch_failure_map.data.validate import assert_valid_records, validate_records
from code_switch_failure_map.schemas.sample import MetadataFlags, SampleRecord
from code_switch_failure_map.schemas.taxonomy import PromptLanguage, SliceTag, SourceSplit
from code_switch_failure_map.utils.io import read_jsonl, write_jsonl


def _base_record(sample_id: str = "s1", text: str = "kal ka weather kya hai?", normalized_text: str | None = None) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        source_split=SourceSplit.RAW,
        text=text,
        normalized_text=normalized_text,
        gold_intent="information_query",
        gold_entities=[],
        metadata_flags=MetadataFlags(code_switching=True),
        slice_tags={SliceTag.CODE_SWITCHING, SliceTag.PROMPT_LANGUAGE_HINGLISH},
        prompt_variant="baseline_v1",
        prompt_language=PromptLanguage.HINGLISH,
    )


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    rows = [_base_record().model_dump(mode="json")]

    write_jsonl(path, rows)
    loaded_rows = read_jsonl(path)
    loaded = load_dataset(path)

    assert loaded_rows == rows
    assert loaded[0].sample_id == "s1"


def test_dataset_validation_passes_on_good_examples() -> None:
    records = [_base_record(sample_id="s1"), _base_record(sample_id="s2", text="mujhe kal 9 baje call yaad dila")]
    assert validate_records(records) == []


def test_duplicate_sample_id_detection() -> None:
    records = [_base_record(sample_id="dup"), _base_record(sample_id="dup")]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        assert_valid_records(records)


def test_duplicate_normalized_text_detection() -> None:
    records = [
        _base_record(sample_id="s1", text="Call Mom!", normalized_text="call mom"),
        _base_record(sample_id="s2", text="call    mom", normalized_text="call mom"),
    ]

    issues = validate_records(records)

    assert any("normalized text duplicates" in issue.message for issue in issues)


def test_missing_gold_intent_detection() -> None:
    payload = _base_record().model_dump(mode="json")
    payload["gold_intent"] = "   "

    with pytest.raises(ValidationError):
        SampleRecord.model_validate(payload)


def test_slice_summary_generation() -> None:
    records = [
        _base_record(sample_id="s1"),
        _base_record(sample_id="s2", text="noise", normalized_text="noise").model_copy(
            update={
                "metadata_flags": MetadataFlags(code_switching=True, transliteration_noise=True),
                "slice_tags": {
                    SliceTag.CODE_SWITCHING,
                    SliceTag.TRANSLITERATION_NOISE,
                    SliceTag.PROMPT_LANGUAGE_HINGLISH,
                },
            }
        ),
    ]

    summary = count_by_slice(records)

    assert summary["code_switching"] == 2
    assert summary["transliteration_noise"] == 1


def test_export_subset_behavior() -> None:
    records = [
        _base_record(sample_id="s1"),
        _base_record(sample_id="s2", text="second"),
        _base_record(sample_id="s3", text="third"),
    ]

    subset = export_subset_by_ids(records, ["s3", "missing", "s1"])

    assert [record.sample_id for record in subset] == ["s3", "s1"]


def test_deterministic_split_behavior() -> None:
    records = [_base_record(sample_id=f"s{i}") for i in range(20)]

    curated_a, golden_a = split_records(records, golden_ratio=0.3)
    curated_b, golden_b = split_records(records, golden_ratio=0.3)

    assert [record.sample_id for record in curated_a] == [record.sample_id for record in curated_b]
    assert [record.sample_id for record in golden_a] == [record.sample_id for record in golden_b]


def test_golden_selection_is_deterministic_and_size_bound() -> None:
    records = load_dataset("data/raw/seed_hinglish_samples.jsonl")

    first = select_golden_set(records, size=50)
    second = select_golden_set(records, size=50)

    first_ids = [record.sample_id for record in first.golden_set]
    second_ids = [record.sample_id for record in second.golden_set]

    assert len(first_ids) == 50
    assert first_ids == second_ids


def test_golden_selection_has_no_duplicate_ids() -> None:
    records = load_dataset("data/raw/seed_hinglish_samples.jsonl")
    selected = select_golden_set(records, size=50).golden_set

    ids = [record.sample_id for record in selected]
    assert len(ids) == len(set(ids))


def test_golden_selection_meets_minimum_diversity() -> None:
    records = load_dataset("data/raw/seed_hinglish_samples.jsonl")
    result = select_golden_set(records, size=50)

    assert len(result.intent_distribution) >= 8
    assert max(result.intent_distribution.values()) <= 8
    assert result.slice_distribution.get(SliceTag.ADVERSARIAL.value, 0) >= 10
    assert result.slice_distribution.get(SliceTag.AMBIGUITY.value, 0) >= 10
    assert result.slice_distribution.get(SliceTag.TEMPORAL_REFERENCE.value, 0) >= 10
