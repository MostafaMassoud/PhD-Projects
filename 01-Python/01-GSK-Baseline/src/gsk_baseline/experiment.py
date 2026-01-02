from __future__ import annotations

"""gsk_baseline.experiment

CEC2017 experiment protocol for baseline GSK
===========================================

This module contains the *experimental harness* for evaluating the baseline
Gaining–Sharing Knowledge (GSK) optimizer on the CEC2017 benchmark suite.

Why keep the experiment driver separate from the optimizer?
-----------------------------------------------------------
For rigorous, reproducible metaheuristic research, it is best practice to
separate:

1) **Algorithm logic** (the optimizer):
   implemented in :func:`gsk_baseline.gsk.gsk_optimize`

2) **Experimental protocol** (what to run, how many runs, where to store
   results, how to summarize):
   implemented here.

This separation helps:
- prevent accidental "leakage" of benchmark-specific decisions into the algorithm,
- keep the baseline clean (no RL/MB/hybrids),
- make results reproducible and easy to validate.

CEC2017 protocol implemented here
---------------------------------
- Dimensions supported: D ∈ {10, 30, 50, 100} (configurable).
- Functions supported: F1..F30, with default exclusion of F2
  (CEC2017 F2 is sometimes excluded in downstream repos; the default here matches
  the provided reference results).
- Independent runs: default 51 runs per function and dimension (configurable).
- Evaluation budget (hard constraint): ``max_nfes = 10000 * D`` per run.
- Reported metric: **error** defined as ``best_f - f_opt``, where ``f_opt`` is
  the known CEC2017 optimum shift (commonly ``100 * func_id``).

Output layout (user requirement)
--------------------------------
All outputs are written under the project root as:

- ``<project_root>/results/<alg_name>/gen_logs``:
  runtime logs, environment/config dumps, and *per-run* final outcomes.

- ``<project_root>/results/<alg_name>/summary``:
  aggregated summary CSVs in the exact schema required by the validation tool.

No additional output folders (e.g., curves/, figures/, results/) are created.

Notes on determinism
--------------------
This harness is deterministic given the same configuration and base seed.
The optimizer itself uses a local NumPy RandomState derived solely from the
run seed. For strict reproducibility of timing measurements you may also set
single-thread BLAS environment variables via the CLI flag ``--single-thread``.
"""

from dataclasses import dataclass
import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cec2017_adapter import cec2017_function, ensure_cec2017_importable
from .gsk import GSKConfig, gsk_optimize
from .utils import (
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

    This configuration controls the *benchmarking protocol* (dimensions, function
    ids, number of runs, output folders, seeding strategy) and provides default
    baseline hyperparameters for GSK.

    Parameters
    ----------
    dims:
        Problem dimensions to evaluate.
    funcs:
        Candidate function ids (typically 1..30). This is filtered by
        :attr:`exclude_funcs` before running.
    exclude_funcs:
        Function ids to exclude. Default excludes F2 to match the provided
        reference results in ``previous_results/gsk``.
    runs:
        Number of independent runs per (function, dimension).
    pop_size:
        Population size (baseline often uses 100 for CEC2017).
    bounds:
        Box constraints (lower, upper) used for all dimensions.
    KF, KR, Kexp:
        Baseline GSK parameters (see :class:`gsk_baseline.gsk.GSKConfig`).
    base_seed:
        Base seed used to derive deterministic per-run seeds.
    stride_run:
        Seed stride separating functions (helps avoid overlapping sequences).
    project_root:
        Project root directory (default is current working directory when invoked
        via CLI scripts, but the scripts pass their own resolved project root).
    results_root_dirname:
        Root results directory name under :attr:`project_root` (default: ``results``).
    alg_name:
        Algorithm name used for the results subfolder:
        ``<project_root>/results/<alg_name>/...``.
    summary_dirname, gen_logs_dirname:
        Subfolder names for summary CSVs and logs.
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

    # Output (required structure)
    project_root: Path = Path(".")
    results_root_dirname: str = "results"
    alg_name: str = "GSK-Baseline"
    summary_dirname: str = "summary"
    gen_logs_dirname: str = "gen_logs"

    def algo_results_dir(self) -> Path:
        """Return ``<project_root>/results/<alg_name>`` (created if missing)."""
        return ensure_dir(self.project_root / self.results_root_dirname / self.alg_name)

    def summary_dir(self) -> Path:
        """Return ``<project_root>/results/<alg_name>/summary`` (created if missing)."""
        return ensure_dir(self.algo_results_dir() / self.summary_dirname)

    def gen_logs_dir(self) -> Path:
        """Return ``<project_root>/results/<alg_name>/gen_logs`` (created if missing)."""
        return ensure_dir(self.algo_results_dir() / self.gen_logs_dirname)


def _format_scientific(x: float) -> str:
    """Format a number as scientific notation consistent with the reference CSVs."""
    # Reference files use 2 decimals and uppercase E (e.g., 1.41E+01).
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
        Experiment configuration (dimensions, runs, algorithm hyperparameters,
        output layout).
    cec_root:
        Optional override path to the external CEC2017 folder. If not provided,
        the code expects ``../00-CEC2017`` relative to ``cfg.project_root``.
    smoke:
        If True, run a small subset intended for quick verification. The budget
        rule remains unchanged.
    verbose:
        If True, print progress to stdout. Regardless of this flag, a full log is
        written to ``gen_logs``.

    Returns
    -------
    dict[int, Path]
        Mapping dimension -> path to the generated summary CSV.

    Budget rule (non-negotiable)
    ----------------------------
    Each run uses ``max_nfes = 10000 * D`` evaluations, enforced internally by the
    optimizer's :class:`gsk_baseline.budget.BudgetController`.

    Reported metric
    ---------------
    We report error as ``best_f - f_opt``.
    """

    project_root = cfg.project_root.resolve()

    # Ensure CEC2017 import works and record where it was loaded from.
    cec_source = ensure_cec2017_importable(project_root, override=cec_root)

    # Resolve output folders.
    algo_dir = cfg.algo_results_dir()
    summary_dir = cfg.summary_dir()
    gen_logs_dir = cfg.gen_logs_dir()

    # Persist environment + configuration (machine-readable).
    env_meta = collect_environment_metadata()
    write_json(gen_logs_dir / "environment.json", as_pretty_dict(env_meta))
    write_json(gen_logs_dir / "experiment_config.json", as_pretty_dict(cfg))

    # Create a human-readable experiment log (mirrors stdout).
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    run_log_path = gen_logs_dir / f"{cfg.alg_name}_CEC2017_Python_Result_{timestamp}.txt"

    # Smoke-test override (does not change the budget rule).
    if smoke:
        dims = (10,)
        funcs = (1, 3, 4, 5)
        runs = min(3, int(cfg.runs))
        exclude = cfg.exclude_funcs
    else:
        dims = tuple(int(d) for d in cfg.dims)
        funcs = tuple(int(f) for f in cfg.funcs)
        runs = int(cfg.runs)
        exclude = tuple(int(f) for f in cfg.exclude_funcs)

    funcs_to_run = tuple(sorted(int(f) for f in funcs if int(f) not in set(exclude)))

    generated: Dict[int, Path] = {}

    with run_log_path.open("w", encoding="utf-8", newline="\n") as log_f:

        def _log(msg: str = "") -> None:
            """Write a line to the run log and (optionally) to stdout."""
            if verbose:
                print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

        # ------------------------------------------------------------------
        # Header (environment + config)
        # ------------------------------------------------------------------
        _log("=" * 72)
        _log(f"{cfg.alg_name} | CEC source: {cec_source}")
        _log("=" * 72)
        _log("")
        _log("#" * 72)
        _log("# ENVIRONMENT / DETERMINISM")
        _log("#" * 72)
        for line in format_environment_metadata(env_meta).splitlines():
            _log(line)
        _log("")
        _log("#" * 72)
        _log("# CONFIG / METADATA")
        _log("#" * 72)
        _log(f"project_root : {project_root}")
        _log(f"results_root : {algo_dir}")
        _log(f"summary_dir  : {summary_dir}")
        _log(f"gen_logs_dir : {gen_logs_dir}")
        _log(f"cec_source   : {cec_source}")
        _log(f"cfg.alg_name      = {cfg.alg_name!r}")
        _log(f"cfg.runs          = {runs}")
        _log(f"cfg.dims          = {dims}")
        _log(f"cfg.funcs         = {funcs}")
        _log(f"cfg.exclude_funcs = {exclude}")
        _log(f"funcs_to_run      = {funcs_to_run}")
        _log(f"cfg.pop_size      = {cfg.pop_size}")
        _log(f"cfg.bounds        = {cfg.bounds}")
        _log(f"cfg.KF, KR, Kexp  = ({cfg.KF}, {cfg.KR}, {cfg.Kexp})")
        _log(f"rng_schedule      : seed = base_seed + func_id * stride_run + (run_id - 1)")
        _log(f"rng_params        : base_seed={cfg.base_seed} stride_run={cfg.stride_run}")
        _log("")

        # ------------------------------------------------------------------
        # Main experiment loops
        # ------------------------------------------------------------------
        for D in dims:
            D = int(D)
            max_nfes = 10000 * D

            _log("\n" + "=" * 72)
            _log(f"Dimension D={D} | max_nfes={max_nfes} | pop_size={cfg.pop_size} | runs={runs}")
            _log("=" * 72)

            # Per-run outcomes for this dimension (final best error per run).
            per_run_path = gen_logs_dir / f"Per_Run_Errors_D{D}.csv"
            with per_run_path.open("w", encoding="utf-8", newline="") as pr_f:
                pr_w = csv.writer(pr_f)
                pr_w.writerow(
                    [
                        "Function",
                        "Run",
                        "Seed",
                        "BestF",
                        "BestError",
                        "NFEsUsed",
                        "MaxNFEs",
                        "StopReason",
                        "Seconds",
                    ]
                )

                dim_rows: List[Dict[str, float]] = []

                for func_id in funcs_to_run:
                    func_id = int(func_id)

                    # Known optimum shift used in many CEC suites.
                    # (This matches the reference summary files shipped with the project.)
                    optimum = float(func_id) * 100.0

                    objective = cec2017_function(func_id)

                    outcomes: List[float] = []

                    _log(f"\n{cfg.alg_name} | F{func_id:02d} (D={D}): running {runs} trials .")

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
                            max_nfes=max_nfes,
                            seed=seed,
                            KF=float(cfg.KF),
                            KR=float(cfg.KR),
                            Kexp=float(cfg.Kexp),
                        )

                        t0 = time.perf_counter()
                        res = gsk_optimize(objective=objective, config=run_cfg)
                        t1 = time.perf_counter()

                        err = float(res.best_f - optimum)
                        outcomes.append(err)

                        seconds = float(t1 - t0)

                        pr_w.writerow(
                            [
                                func_id,
                                run_id,
                                seed,
                                f"{res.best_f:.16e}",
                                f"{err:.16e}",
                                int(res.nfes_used),
                                int(res.max_nfes),
                                str(res.stop_reason),
                                f"{seconds:.6f}",
                            ]
                        )

                        _log(
                            f"     Run {run_id:02d} | seed={seed} | err={err:.8e} | best_f={res.best_f:.8e} "
                            f"| evals={res.nfes_used} | t={seconds:.6f}s | stop={res.stop_reason}"
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

                    _log(
                        f"  Summary F{func_id:02d} | "
                        f"Best={row['Best']:.3e} Median={row['Median']:.3e} Mean={row['Mean']:.3e} "
                        f"Worst={row['Worst']:.3e} SD={row['SD']:.3e}"
                    )

            # Write dimension-level summary.
            summary_path = summary_dir / f"Summary_All_Results_D{D}.csv"
            _write_summary_csv(summary_path, dim_rows)
            generated[D] = summary_path

            # Print the summary to console/log as requested (mirrors the CSV).
            _log("\nSaved summary: " + str(summary_path))
            _log("Function,Best,Median,Mean,Worst,SD")
            for r in dim_rows:
                _log(
                    f"{int(r['Function'])},"
                    f"{_format_scientific(r['Best'])},"
                    f"{_format_scientific(r['Median'])},"
                    f"{_format_scientific(r['Mean'])},"
                    f"{_format_scientific(r['Worst'])},"
                    f"{_format_scientific(r['SD'])}"
                )

    return generated
