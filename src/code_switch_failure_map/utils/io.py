"""I/O helpers for JSONL dataset artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


JsonDict = dict[str, Any]


def read_jsonl(path: str | Path) -> list[JsonDict]:
    """Read a UTF-8 JSONL file into a list of dictionaries."""
    input_path = Path(path)
    records: list[JsonDict] = []
    with input_path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parsed = orjson.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object at line {line_no} in {input_path}")
            records.append(parsed)
    return records


def write_jsonl(path: str | Path, rows: list[JsonDict]) -> None:
    """Write dictionaries to JSONL using compact canonical encoding."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS))
            handle.write(b"\n")
