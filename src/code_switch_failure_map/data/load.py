"""Dataset loading helpers for typed sample records."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.utils.io import read_jsonl


def load_raw_rows(path: str | Path) -> list[dict[str, object]]:
    """Load raw JSONL rows without schema parsing."""
    return read_jsonl(path)


def load_dataset(path: str | Path) -> list[SampleRecord]:
    """Load and parse dataset JSONL rows as ``SampleRecord`` objects."""
    rows = read_jsonl(path)
    records: list[SampleRecord] = []
    for line_number, row in enumerate(rows, start=1):
        try:
            records.append(SampleRecord.model_validate(row))
        except ValidationError as exc:
            raise ValueError(f"Invalid record at line {line_number} in {path}: {exc}") from exc
    return records


def dumpable_records(records: list[SampleRecord]) -> list[dict[str, object]]:
    """Serialize typed records to JSON-compatible dictionaries."""
    return [record.model_dump(mode="json") for record in records]
