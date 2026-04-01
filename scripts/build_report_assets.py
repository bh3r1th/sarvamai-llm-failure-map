#!/usr/bin/env python3
"""Build analysis tables and compact blog assets for one evaluated run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from code_switch_failure_map.cli import build_report_assets_for_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report tables and blog assets from evaluated outputs.")
    parser.add_argument("--run-dir", required=True, help="Path to a prediction run directory under data/predictions.")
    args = parser.parse_args()

    artifacts, _ = build_report_assets_for_run(Path(args.run_dir))
    print(f"Run id: {artifacts['run_id']}")
    print(f"Comparison output dir: {artifacts['comparison_output_dir']}")
    for path in artifacts["created_files"]:
        print(f"Created: {path}")


if __name__ == "__main__":
    main()
