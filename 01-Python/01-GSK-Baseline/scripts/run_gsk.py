#!/usr/bin/env python3
"""CLI entry point: run baseline GSK experiments on CEC2017.

This script is intentionally lightweight and delegates the heavy lifting to
:func:`gsk_baseline.experiment.run_cec2017_experiments`.

Examples
--------
Full protocol (29 functions, 4 dimensions, 51 runs):

    python scripts/run_gsk.py --runs 51 --dims 10 30 50 100

Smoke test (quick sanity check):

    python scripts/run_gsk.py --smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _project_root() -> Path:
    # scripts/ is one level below project root
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    root = _project_root()

    # Allow running without installing the package.
    sys.path.insert(0, str(root / "src"))

    from gsk_baseline.experiment import ExperimentConfig, run_cec2017_experiments
    from gsk_baseline.utils import parse_int_set, set_single_thread_env_if_requested

    p = argparse.ArgumentParser(description="Run baseline GSK experiments on CEC2017")

    p.add_argument("--dims", nargs="*", type=int, default=None, help="Problem dimensions (default: 10 30 50 100)")
    p.add_argument("--funcs", type=str, default=None, help="Function ids (e.g. '1-30' or '1,3,4')")
    p.add_argument("--exclude", type=str, default="2", help="Excluded function ids (default: '2')")
    p.add_argument("--runs", type=int, default=51, help="Number of independent runs (default: 51)")
    p.add_argument("--pop-size", type=int, default=100, help="Population size (default: 100)")
    p.add_argument("--base-seed", type=int, default=123456, help="Base seed (default: 123456)")
    p.add_argument("--stride-run", type=int, default=9973, help="Seed stride per function (default: 9973)")

    p.add_argument("--cec-root", type=str, default=None, help="Override path to external CEC2017 folder")

    p.add_argument("--smoke", action="store_true", help="Run a reduced suite for quick verification")
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
        pop_size=int(args.pop_size),
        base_seed=int(args.base_seed),
        stride_run=int(args.stride_run),
        project_root=root,
    )

    cec_root = Path(args.cec_root).resolve() if args.cec_root else None

    run_cec2017_experiments(cfg=cfg, cec_root=cec_root, smoke=bool(args.smoke), verbose=not bool(args.quiet))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
