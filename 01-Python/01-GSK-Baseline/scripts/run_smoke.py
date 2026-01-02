#!/usr/bin/env python3
"""Quick smoke test for GSK implementation.

Usage:
    python scripts/run_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gsk.config import Config
from gsk.experiment import run_cec2017_experiments


def main() -> None:
    cfg = Config(
        project_root=PROJECT_ROOT,
        runs=3,
        dims=(10,),
        funcs=(1, 3),
        exclude_funcs=(2,),
        enable_curves_10=True,
        verbose=True,
    )

    print("Running GSK smoke test...")
    run_cec2017_experiments(cfg=cfg, smoke=False, verbose=True)
    print("\nSmoke test completed successfully.")


if __name__ == "__main__":
    main()
