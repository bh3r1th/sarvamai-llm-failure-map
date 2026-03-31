#!/usr/bin/env python3
"""Build curated dataset artifacts from raw Hinglish seed samples."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from code_switch_failure_map.data.curate import count_by_intent, count_by_slice, identify_low_diversity_samples
from code_switch_failure_map.data.load import dumpable_records, load_raw_rows
from code_switch_failure_map.data.validate import validate_raw_rows
from code_switch_failure_map.utils.io import write_jsonl

RAW_PATH = Path("data/raw/seed_hinglish_samples.jsonl")
CURATED_PATH = Path("data/curated/seed_hinglish_curated.jsonl")


def _print_table(title: str, rows: dict[str, int]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in rows.items():
        print(f"{key:30} {value:>4}")


def main() -> int:
    raw_rows = load_raw_rows(RAW_PATH)
    records, issues = validate_raw_rows(raw_rows)

    if issues:
        print(f"Validation failed with {len(issues)} issue(s):")
        for issue in issues:
            print(f"- {issue.sample_id}: {issue.message}")
        return 1

    curated_rows = dumpable_records(records)
    write_jsonl(CURATED_PATH, curated_rows)

    slice_counts = count_by_slice(records)
    intent_counts = count_by_intent(records)
    low_diversity = identify_low_diversity_samples(records)

    print(f"total sample count: {len(records)}")
    _print_table("coverage by slice", slice_counts)
    _print_table("coverage by intent", intent_counts)

    print("\nweak spots in dataset diversity")
    print("-------------------------------")
    if not low_diversity:
        print("none detected")
    else:
        for key, sample_ids in sorted(low_diversity.items()):
            print(f"{sample_ids} -> {key}")

    print(f"\nWrote curated dataset to: {CURATED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
