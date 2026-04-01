"""Export helpers for report JSON and CSV assets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from code_switch_failure_map.utils.io import write_json


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one flat row table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_table_bundle(output_dir: Path, tables: dict[str, list[dict[str, Any]]]) -> list[str]:
    """Write JSON and CSV variants of each report table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    for name, rows in tables.items():
        json_path = output_dir / f"{name}.json"
        csv_path = output_dir / f"{name}.csv"
        write_json(json_path, {"name": name, "rows": rows})
        write_csv(csv_path, rows)
        created.extend([str(json_path), str(csv_path)])
    return created
