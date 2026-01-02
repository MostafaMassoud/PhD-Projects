from __future__ import annotations

"""gsk_baseline.experiment

CEC2017 experiment protocol for baseline GSK
===========================================

This module orchestrates experiments in a way that is typical for
Q1-journal-level metaheuristic benchmarking:

- Multiple independent runs with controlled seeding.
- Fixed evaluation budget: ``max_nfes = 10000 * D``.
- Aggregate performance measures: Best, Median, Mean, Worst, SD.
- Outputs written in a standard CSV schema to enable downstream analysis.

No algorithmic logic lives here: the optimizer itself is in
:func:`gsk_baseline.gsk.gsk_optimize`.
"""

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cec2017_adapter import cec2017_function, ensure_cec2017_importable
from .gsk import GSKConfig, gsk_optimize
from .utils import (
    EnvironmentMetadata,
    as_pretty_dict,
    collect_environment_metadata,
    ensure_dir,
    format_environment_metadata,
    seed_for_run,
    write_json,
)


SUMMARY_HEADER: Tuple[str, ...] = ("Function", "Best", "Median", "Mean", "Worst", "SD")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a CEC2017 experiment suite.

    This configuration is *experiment-level* (functions, dimensions, number of
    runs, output locations) and distinct from :class:`gsk_baseline.gsk.GSKConfig`,
    which describes a single optimizer run.
    """

    # CEC2017 experiment design
    dims: Tuple[int, ...] = (10, 30, 50, 100)
    funcs: Tuple[int, ...] = tuple(range(1, 31))
    exclude_funcs: Tuple[int, ...] = (2,)
    runs: int = 51

    # Baseline optimizer settings
    pop_size: int = 100
    bounds: Tuple[float, float] = (-100.0, 100.0)
    KF: float = 0.5
    KR: float = 0.9
    Kexp: float = 10.0

    # Reproducibility
    base_seed: int = 123456
    stride_run: int = 9973

    # Output
    project_root: Path = Path(".")
    results_dirname: str = "results"
    logs_dirname: str = "logs"

    def results_dir(self) -> Path:
        return ensure_dir(self.project_root / self.results_dirname)

    def logs_dir(self) -> Path:
        return ensure_dir(self.project_root / self.logs_dirname)


def _format_scientific(x: float) -> str:
    """Format numbers in a compact scientific notation similar to many CEC tables."""

    # The provided reference CSVs use 2 decimals and uppercase E (e.g., 1.41E+01).
    return f"{x:.2E}"


def _write_summary_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    """Write a dimension-level summary CSV with the required schema."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        for r in rows:
            w.writerow(
                [
                    int(r["Function"]),
                    _format_scientific(r["Best"]),
                    _format_scientific(r["Median"]),
                    _format_scientific(r["Mean"]),
                    _format_scientific(r["Worst"]),
                    _format_scientific(r["SD"]),
                ]
            )


def run_cec2017_experiments(
    *,
    cfg: ExperimentConfig,
    cec_root: Optional[Path] = None,
    smoke: bool = False,
    verbose: bool = True,
) -> Dict[int, Path]:
    """Run baseline GSK on the CEC2017 benchmark suite.

    Parameters
    ----------
    cfg:
        Experiment configuration.
    cec_root:
        Optional override path to the external CEC2017 folder. If not provided,
        the code expects ``../00-CEC2017`` relative to ``cfg.project_root``.
    smoke:
        If True, run a reduced suite intended for quick verification.
        The budget rule remains unchanged.
    verbose:
        Print progress to stdout.

    Returns
    -------
    dict[int, Path]
        Mapping dimension -> path to generated summary CSV.

    Notes
    -----
    **Budget rule:** Each run uses ``max_nfes = 10000 * D``.

    **Objective error reporting:** For CEC2017, the global optimum is known to be
    ``f_opt = 100 * func_id`` for functions 1..30 in the commonly used suite.
    We report error as ``best_f - f_opt``.
    """

    project_root = cfg.project_root.resolve()

    # Ensure CEC2017 import works and log where it was loaded from.
    cec_source = ensure_cec2017_importable(project_root, override=cec_root)

    # Environment + configuration logs
    env_meta = collect_environment_metadata()
    ensure_dir(cfg.logs_dir())
    write_json(cfg.logs_dir() / "environment.json", as_pretty_dict(env_meta))
    write_json(cfg.logs_dir() / "experiment_config.json", as_pretty_dict(cfg))

    if verbose:
        print(format_environment_metadata(env_meta))
        print("CEC2017 source:")
        print(f"  {cec_source}")
        print("Experiment configuration:")
        for k, v in as_pretty_dict(cfg).items():
            print(f"  {k}: {v}")

    # Smoke test reductions (protocol-level only, not algorithmic changes)
    if smoke:
        dims = (10,)
        funcs = (1, 3, 4, 5)
        runs = min(3, int(cfg.runs))
        exclude = cfg.exclude_funcs
    else:
        dims = cfg.dims
        funcs = cfg.funcs
        runs = int(cfg.runs)
        exclude = cfg.exclude_funcs

    results_dir = cfg.results_dir()

    generated: Dict[int, Path] = {}

    for D in dims:
        D = int(D)
        if verbose:
            print("\n" + "=" * 72)
            print(f"Dimension D={D} | max_nfes={10000*D} | pop_size={cfg.pop_size} | runs={runs}")
            print("=" * 72)

        dim_rows: List[Dict[str, float]] = []

        for func_id in funcs:
            func_id = int(func_id)
            if func_id in set(int(f) for f in exclude):
                continue

            objective = cec2017_function(func_id)
            optimum = float(func_id * 100.0)

            outcomes: List[float] = []

            for run_id in range(1, runs + 1):
                seed = seed_for_run(
                    base_seed=int(cfg.base_seed),
                    stride_run=int(cfg.stride_run),
                    func_id=func_id,
                    run_id=run_id,
                )

                run_cfg = GSKConfig(
                    dim=D,
                    pop_size=int(cfg.pop_size),
                    bounds=cfg.bounds,
                    max_nfes=10000 * D,
                    seed=seed,
                    KF=float(cfg.KF),
                    KR=float(cfg.KR),
                    Kexp=float(cfg.Kexp),
                )

                res = gsk_optimize(objective=objective, config=run_cfg)

                # Error (CEC2017 convention): f_best - f_opt
                err = float(res.best_f - optimum)
                outcomes.append(err)

                if verbose and (run_id == 1 or run_id == runs or (run_id % 10 == 0)):
                    print(
                        f"F{func_id:02d} Run {run_id:02d}/{runs} | seed={seed} | "
                        f"nfes={res.nfes_used}/{res.max_nfes} | stop={res.stop_reason} | best_err={err:.3e}"
                    )

            out = np.asarray(outcomes, dtype=np.float64)

            row = {
                "Function": float(func_id),
                "Best": float(np.min(out)),
                "Median": float(np.median(out)),
                "Mean": float(np.mean(out)),
                "Worst": float(np.max(out)),
                "SD": float(np.std(out, ddof=1)) if out.size > 1 else 0.0,
            }
            dim_rows.append(row)

        summary_path = results_dir / f"Summary_All_Results_D{D}.csv"
        _write_summary_csv(summary_path, dim_rows)
        generated[D] = summary_path

        if verbose:
            print(f"Saved: {summary_path}")

    return generated
