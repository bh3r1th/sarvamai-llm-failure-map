#!/usr/bin/env python3
"""Mine failure exemplars from an evaluated run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from code_switch_failure_map.cli import mine_failure_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine high-signal failure exemplars for one evaluated run.")
    parser.add_argument("--run-dir", required=True, help="Path to a prediction run directory under data/predictions.")
    args = parser.parse_args()

    artifacts, _ = mine_failure_assets(Path(args.run_dir))
    print(f"Run id: {artifacts['run_id']}")
    print(f"Failure exemplars: {artifacts['failure_exemplars_path']}")
    print(f"Exemplar index JSON: {artifacts['exemplar_index_json_path']}")
    print(f"Exemplar index CSV: {artifacts['exemplar_index_csv_path']}")


if __name__ == "__main__":
    main()
