#!/usr/bin/env python3
"""Validate GSK results against bundled reference summaries.

Usage:
    python scripts/validate_gsk.py
    python scripts/validate_gsk.py --abs-tol 0 --rel-tol 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gsk.constants import DEFAULT_DIMS, DEFAULT_EXCLUDE_FUNCS, REPORT_ZERO_TOL
from gsk.validation import validate_summary_pair


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate GSK results.")
    ap.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_DIMS),
        help="Dimensions to validate.",
    )
    ap.add_argument(
        "--abs-tol",
        type=float,
        default=0.0,
        help="Absolute tolerance.",
    )
    ap.add_argument(
        "--rel-tol",
        type=float,
        default=0.0,
        help="Relative tolerance.",
    )
    ap.add_argument(
        "--zero-tol",
        type=float,
        default=REPORT_ZERO_TOL,
        help="Zero tolerance for small values.",
    )
    args = ap.parse_args()

    results_dir = PROJECT_ROOT / "results" / "gsk" / "summary"
    baseline_dir = PROJECT_ROOT / "previous_results" / "gsk"

    all_ok = True

    for D in args.dims:
        new_csv = results_dir / f"Summary_All_Results_D{D}.csv"
        ref_csv = baseline_dir / f"Summary_All_Results_D{D}.csv"

        if not new_csv.exists():
            print(f"[SKIP] D={D}: No results found at {new_csv}")
            continue

        if not ref_csv.exists():
            print(f"[SKIP] D={D}: No reference found at {ref_csv}")
            continue

        ok, report = validate_summary_pair(
            ref_csv=ref_csv,
            new_csv=new_csv,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
            excluded_funcs=DEFAULT_EXCLUDE_FUNCS,
            zero_tol=args.zero_tol,
        )

        if ok:
            print(f"[PASS] D={D}")
        else:
            print(f"[FAIL] D={D}")
            print(report)
            all_ok = False

    if all_ok:
        print("\nAll validations passed.")
        sys.exit(0)
    else:
        print("\nSome validations failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
