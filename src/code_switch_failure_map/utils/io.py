"""I/O helpers for JSON, JSONL, and run artifact paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
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


def read_json(path: str | Path) -> JsonDict:
    """Read one JSON document from disk."""
    input_path = Path(path)
    payload = orjson.loads(input_path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    return payload


def write_json(path: str | Path, payload: JsonDict) -> None:
    """Write one JSON document using deterministic key ordering."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def read_json_compatible_yaml(path: str | Path) -> JsonDict:
    """Read a JSON-compatible YAML file.

    This loader intentionally accepts the JSON subset of YAML so the repo can
    keep a ``.yaml`` config surface without adding a YAML dependency yet.
    """

    input_path = Path(path)
    try:
        payload = orjson.loads(input_path.read_bytes())
    except orjson.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid config file {input_path}. Use JSON-compatible YAML for now."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in config file {input_path}")
    return payload


def slugify_filename(value: str) -> str:
    """Convert a free-form label into a stable filename token."""
    compact = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return compact or "item"


def make_run_id(prefix: str) -> str:
    """Build a UTC timestamped run identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{slugify_filename(prefix)}_{timestamp}"
