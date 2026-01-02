#!/usr/bin/env python3
"""CLI entrypoint: run baseline GSK on CEC2017.

Usage:
    python scripts/run_gsk.py --runs 51 --dims 10 30 50 100
    python scripts/run_gsk.py --smoke  # Quick sanity check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gsk.config import Config
from gsk.experiment import run_cec2017_experiments
from gsk.utils import apply_thread_env, parse_int_set


def _build_parser(defaults: Config) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run baseline GSK on CEC2017 benchmark."
    )

    ap.add_argument(
        "--runs",
        type=int,
        default=defaults.runs,
        help="Runs per (function, dimension).",
    )
    ap.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=list(defaults.dims),
        help="Dimensions to run.",
    )
    ap.add_argument(
        "--funcs",
        type=str,
        default="1-30",
        help="Functions to run as compact spec (e.g., '1-30' or '1,3,4').",
    )
    ap.add_argument(
        "--exclude-funcs",
        type=str,
        default=",".join(str(i) for i in defaults.exclude_funcs),
        help="Functions to exclude as compact spec.",
    )
    ap.add_argument(
        "--cec-root",
        type=str,
        default=None,
        help="Path to external CEC2017 folder (optional).",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Small quick run for sanity check.",
    )

    # Curves export flags
    ap.add_argument(
        "--enable-curves-10",
        default=defaults.enable_curves_10,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable curves for D=10 (median run only).",
    )
    ap.add_argument(
        "--enable-curves-30",
        default=defaults.enable_curves_30,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable curves for D=30 (median run only).",
    )
    ap.add_argument(
        "--enable-curves-50",
        default=defaults.enable_curves_50,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable curves for D=50 (median run only).",
    )
    ap.add_argument(
        "--enable-curves-100",
        default=defaults.enable_curves_100,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable curves for D=100 (median run only).",
    )
    ap.add_argument(
        "--enable-curves-all",
        default=defaults.enable_curves_all,
        action=argparse.BooleanOptionalAction,
        help="Export curves for all requested dims.",
    )

    ap.add_argument(
        "--baseline-compare-alg",
        type=str,
        default=defaults.baseline_compare_alg,
        help="Folder name under previous_results/ for baseline comparison.",
    )

    # Determinism / logging
    ap.add_argument(
        "--single-thread",
        default=defaults.force_single_thread,
        action=argparse.BooleanOptionalAction,
        help="Force single-threaded BLAS (recommended for reproducibility).",
    )
    ap.add_argument(
        "--override-thread-env",
        default=defaults.override_thread_env,
        action=argparse.BooleanOptionalAction,
        help="Override thread env vars even if already set.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print progress to stdout.",
    )

    # GSK hyperparameters
    ap.add_argument("--pop-size", type=int, default=defaults.pop_size)
    ap.add_argument("--KF", type=float, default=defaults.KF)
    ap.add_argument("--KR", type=float, default=defaults.KR)
    ap.add_argument("--Kexp", type=float, default=defaults.Kexp)

    ap.add_argument("--base-seed", type=int, default=defaults.base_seed)
    ap.add_argument("--stride-run", type=int, default=defaults.stride_run)

    ap.add_argument(
        "--max-nfes",
        type=int,
        default=defaults.max_nfes_override if defaults.max_nfes_override else 0,
        help="Optional absolute NFE budget (0 = use CEC2017 default: 10000*D).",
    )

    ap.add_argument(
        "--report-zero-tol",
        type=float,
        default=defaults.report_zero_tol,
        help="Reporting tolerance: treat |x|<=tol as 0.",
    )
    ap.add_argument(
        "--val-to-reach",
        type=float,
        default=defaults.val_to_reach,
        help="Curve/log10 clamp value.",
    )

    return ap


def main() -> None:
    defaults = Config(project_root=PROJECT_ROOT)
    ap = _build_parser(defaults)
    args = ap.parse_args()

    apply_thread_env(
        force_single_thread=bool(args.single_thread),
        override=bool(args.override_thread_env),
    )

    funcs = tuple(sorted(parse_int_set(str(args.funcs))))
    exclude = tuple(sorted(parse_int_set(str(args.exclude_funcs))))

    cfg = Config(
        alg_name="gsk",
        project_root=PROJECT_ROOT,
        cec_root=Path(args.cec_root).resolve() if args.cec_root else None,
        runs=int(args.runs),
        dims=tuple(int(d) for d in args.dims),
        funcs=tuple(int(f) for f in funcs),
        exclude_funcs=tuple(int(f) for f in exclude),
        pop_size=int(args.pop_size),
        KF=float(args.KF),
        KR=float(args.KR),
        Kexp=float(args.Kexp),
        base_seed=int(args.base_seed),
        stride_run=int(args.stride_run),
        max_nfes_override=(int(args.max_nfes) if int(args.max_nfes) > 0 else None),
        report_zero_tol=float(args.report_zero_tol),
        val_to_reach=float(args.val_to_reach),
        baseline_compare_alg=str(args.baseline_compare_alg),
        enable_curves_10=bool(args.enable_curves_10),
        enable_curves_30=bool(args.enable_curves_30),
        enable_curves_50=bool(args.enable_curves_50),
        enable_curves_100=bool(args.enable_curves_100),
        enable_curves_all=bool(args.enable_curves_all),
        verbose=not bool(args.quiet),
        force_single_thread=bool(args.single_thread),
        override_thread_env=bool(args.override_thread_env),
    )

    run_cec2017_experiments(cfg=cfg, smoke=bool(args.smoke), verbose=cfg.verbose)


if __name__ == "__main__":
    main()
