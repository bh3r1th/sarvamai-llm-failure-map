"""Behavioral tests for dataset I/O, validation, and splitting."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from code_switch_failure_map.data.load import load_dataset
from code_switch_failure_map.data.split import split_records
from code_switch_failure_map.data.validate import assert_valid_records
from code_switch_failure_map.schemas.sample import MetadataFlags, SampleRecord
from code_switch_failure_map.schemas.taxonomy import PromptLanguage, SliceTag, SourceSplit
from code_switch_failure_map.utils.io import read_jsonl, write_jsonl


def _base_record(sample_id: str = "s1") -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        source_split=SourceSplit.RAW,
        text="kal ka weather kya hai?",
        normalized_text="kal ka weather kya hai?",
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


def test_duplicate_sample_id_detection() -> None:
    records = [_base_record(sample_id="dup"), _base_record(sample_id="dup")]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        assert_valid_records(records)


def test_missing_gold_intent_detection() -> None:
    payload = _base_record().model_dump(mode="json")
    payload["gold_intent"] = "   "

    with pytest.raises(ValidationError):
        SampleRecord.model_validate(payload)


def test_deterministic_split_behavior() -> None:
    records = [_base_record(sample_id=f"s{i}") for i in range(20)]

    curated_a, golden_a = split_records(records, golden_ratio=0.3)
    curated_b, golden_b = split_records(records, golden_ratio=0.3)

    assert [record.sample_id for record in curated_a] == [record.sample_id for record in curated_b]
    assert [record.sample_id for record in golden_a] == [record.sample_id for record in golden_b]
