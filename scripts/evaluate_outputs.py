#!/usr/bin/env python3
"""Evaluate prediction outputs and write aggregate artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from code_switch_failure_map.cli import evaluate_prediction_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction artifacts against the gold dataset.")
    parser.add_argument("--run-dir", required=True, help="Path to one smoke run directory under data/predictions.")
    args = parser.parse_args()

    artifacts, issues = evaluate_prediction_outputs(Path(args.run_dir))
    print(f"Run id: {artifacts['run_id']}")
    print(f"Per-sample evaluations: {artifacts['per_sample_path']}")
    print(f"Aggregate summary JSON: {artifacts['summary_json_path']}")
    for table in artifacts["aggregate_tables"]:
        print(f"Aggregate table: {table['name']} ({len(table['rows'])} rows)")

    if issues:
        print("Evaluator issues detected:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("Evaluation artifacts written.")


if __name__ == "__main__":
    main()
