#!/usr/bin/env python3
"""Run the configured experiment from a config manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from code_switch_failure_map.cli import run_configured_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured experiment from a config manifest.")
    parser.add_argument("--config", default="configs/smoke_run.yaml", help="Path to the smoke-run config file.")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id to avoid timestamp-based naming.")
    args = parser.parse_args()

    run_dir, run_kind, dataset_size = run_configured_experiment(Path(args.config), run_id_override=args.run_id)
    print(f"{run_kind.title()} run artifacts written to: {run_dir}")
    print(f"Dataset size: {dataset_size}")


if __name__ == "__main__":
    main()
