"""Dataset loading helpers for typed sample records."""

from __future__ import annotations

from pathlib import Path

from code_switch_failure_map.schemas.sample import SampleRecord
from code_switch_failure_map.utils.io import read_jsonl


def load_dataset(path: str | Path) -> list[SampleRecord]:
    """Load and parse dataset JSONL rows as ``SampleRecord`` objects."""
    rows = read_jsonl(path)
    return [SampleRecord.model_validate(row) for row in rows]


def dumpable_records(records: list[SampleRecord]) -> list[dict[str, object]]:
    """Serialize typed records to JSON-compatible dictionaries."""
    return [record.model_dump(mode="json") for record in records]
