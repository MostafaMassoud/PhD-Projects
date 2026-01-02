#!/usr/bin/env python3
"""CLI entry point: run baseline GSK and validate against reference CSVs.

This script:
1) executes baseline GSK under the configured experiment protocol,
2) writes summary CSVs under ``results/<alg_name>/summary``,
3) compares them to reference results under ``previous_results/gsk``,
4) writes detailed comparison artifacts under ``results/<alg_name>/summary``, and
5) exits with non-zero status if mismatches exceed tolerances.

Examples
--------
Strict (bit-for-bit) validation:

    python scripts/validate_gsk.py --abs-tol 0 --rel-tol 0

Quick smoke validation:

    python scripts/validate_gsk.py --smoke --abs-tol 1e-8 --rel-tol 1e-8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    root = _project_root()

    # Allow running without installing the package.
    sys.path.insert(0, str(root / "src"))

    from gsk_baseline.experiment import ExperimentConfig, run_cec2017_experiments
    from gsk_baseline.validation import validate_against_reference
    from gsk_baseline.utils import parse_int_set, set_single_thread_env_if_requested

    p = argparse.ArgumentParser(description="Run baseline GSK and validate against reference results")

    p.add_argument("--dims", nargs="*", type=int, default=None, help="Dimensions to validate")
    p.add_argument("--funcs", type=str, default=None, help="Functions to run (default: 1-30)")
    p.add_argument("--exclude", type=str, default="2", help="Excluded functions (default: 2)")
    p.add_argument("--runs", type=int, default=51, help="Number of runs (default: 51)")

    p.add_argument("--alg-name", type=str, default="GSK-Baseline", help="Results subfolder name under ./results (default: GSK-Baseline)")

    p.add_argument("--abs-tol", type=float, default=0.0, help="Absolute tolerance per metric")
    p.add_argument("--rel-tol", type=float, default=0.0, help="Relative tolerance per metric")

    p.add_argument("--cec-root", type=str, default=None, help="Override path to external CEC2017 folder")
    p.add_argument("--reference-dir", type=str, default=None, help="Override reference results directory")
    p.add_argument("--artifacts-dir", type=str, default=None, help="Override output directory for comparison artifacts")

    p.add_argument("--smoke", action="store_true", help="Reduced suite for quick validation")
    p.add_argument("--quiet", action="store_true", help="Reduce console output")
    p.add_argument(
        "--single-thread",
        action="store_true",
        help="Set common BLAS thread environment variables to 1 (helps determinism of timing)",
    )

    args = p.parse_args(argv)

    set_single_thread_env_if_requested(force_single_thread=bool(args.single_thread))

    dims = tuple(args.dims) if args.dims else (10, 30, 50, 100)
    funcs = tuple(parse_int_set(args.funcs)) if args.funcs else tuple(range(1, 31))
    exclude = tuple(parse_int_set(args.exclude)) if args.exclude else (2,)

    cfg = ExperimentConfig(
        dims=tuple(int(d) for d in dims),
        funcs=tuple(int(f) for f in funcs),
        exclude_funcs=tuple(int(f) for f in exclude),
        runs=int(args.runs),
        project_root=root,
        alg_name=str(args.alg_name),
    )

    cec_root = Path(args.cec_root).resolve() if args.cec_root else None

    # 1) Run experiments
    run_cec2017_experiments(cfg=cfg, cec_root=cec_root, smoke=bool(args.smoke), verbose=not bool(args.quiet))

    # 2) Validate summaries
    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None
    artifacts_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else cfg.summary_dir()

    res = validate_against_reference(
        project_root=root,
        results_dir=cfg.summary_dir(),
        reference_dir=reference_dir,
        dims=dims if not bool(args.smoke) else (10,),
        abs_tol=float(args.abs_tol),
        rel_tol=float(args.rel_tol),
        artifacts_dir=artifacts_dir,
        verbose=not bool(args.quiet),
    )

    if not res.ok:
        if not bool(args.quiet):
            print(f"\nVALIDATION FAILED: mismatched functions={res.n_mismatched}")
            print(f"Artifacts written to: {res.artifacts_dir}")
        return 1

    if not bool(args.quiet):
        print("\nVALIDATION PASSED")
        print(f"Artifacts written to: {res.artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
