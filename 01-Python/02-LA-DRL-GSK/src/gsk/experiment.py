"""
CEC2017 Experiment Harness for GSK Algorithm
=============================================

This module provides the main experiment runner for evaluating the GSK
algorithm on the CEC2017 benchmark suite.

Experiment Workflow
-------------------
The experiment harness follows this workflow:

    1. SETUP
       ├── Resolve paths (project root, CEC2017 source)
       ├── Apply thread control for determinism
       ├── Import CEC2017 benchmark functions
       └── Save environment metadata for reproducibility

    2. FOR EACH DIMENSION (D ∈ {10, 30, 50, 100})
       ├── Initialize logging (console + file)
       │
       ├── FOR EACH FUNCTION (F1-F30, excluding F2)
       │   │
       │   ├── FOR EACH RUN (1-51)
       │   │   ├── Compute deterministic seed
       │   │   ├── Run GSK optimization
       │   │   ├── Record final error
       │   │   └── (Optional) Save generation logs
       │   │
       │   ├── Compute statistics (Best, Median, Mean, Worst, SD)
       │   ├── Save per-function results
       │   └── (Optional) Export convergence curves for median run
       │
       ├── Write dimension summary CSV
       └── Compare against baseline (if available)

    3. OUTPUTS
       └── results/<alg_name>/
           ├── summary/
           │   ├── Summary_All_Results_D10.csv
           │   ├── Summary_All_Results_D30.csv
           │   ├── ...
           │   ├── environment.json
           │   └── run_config.json
           ├── gen_logs/
           │   ├── GenLog_gsk_F1_D10_Run1.csv
           │   └── ...
           └── curves/
               ├── Figure_F1_D10_Run#26.csv
               └── ...

Output File Formats
-------------------

**Summary CSV** (Summary_All_Results_D{D}.csv):
    Function,Best,Median,Mean,Worst,SD
    1,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00
    3,0.00E+00,0.00E+00,0.00E+00,0.00E+00,0.00E+00
    ...

**Generation Log** (GenLog_gsk_F{F}_D{D}_Run{R}.csv):
    gen,evals_used,best_fitness,diversity,stagnation,...
    1,200,1.23E+05,42.5,0,...
    2,300,9.87E+04,38.2,0,...
    ...

**Convergence Curve** (Figure_F{F}_D{D}_Run#{R}.csv):
    Eval,BestError,Log10Error
    1,1.23E+05,5.09
    2,1.15E+05,5.06
    ...

Error Thresholding
------------------
Two-level thresholding matches the reference MATLAB implementation:

1. **Per-run threshold** (val_to_reach = 1e-8):
   Applied at the end of each run. Errors < 1e-8 are set to exactly 0.0.
   
2. **Report threshold** (report_zero_tol = 1e-7):
   Applied when displaying/writing statistics. Mean values ≤ 1e-7 display
   as "0.00E+00" to avoid confusing tiny values.

Reproducibility
---------------
Every run uses a deterministic seed computed from:
    seed = base_seed + func_id × stride_run + (run_id - 1)

Combined with single-threaded BLAS (optional), this ensures identical
results across runs and platforms.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cec2017_adapter import cec2017_function, ensure_cec2017_importable
from .config import Config
from .constants import CURVE_LOG10_CLAMP
from .gsk import GSKConfig, GSKGenerationLog, gsk_optimize
from .utils import (
    EnvironmentMetadata,
    apply_thread_env,
    collect_environment_metadata,
    ensure_dir,
    env_metadata_as_dict,
    format_environment_metadata,
    format_sci,
    seed_for_run,
    timestamp_now,
    write_json,
    zero_small,
)
from .validation import generate_mean_error_comparison_text


# ============================================================================
# Logging Utility
# ============================================================================

class DualLogger:
    """
    Logger that writes to both console and file simultaneously.
    
    - INFO level: Written to both console and file
    - DEBUG level: Written to file only
    
    This allows clean console output while preserving detailed logs.
    
    Parameters
    ----------
    path : Path
        Log file path.
    verbose_console : bool, default=True
        If True, print INFO messages to console.
    """

    def __init__(self, path: Path, *, verbose_console: bool = True) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self._fh = self.path.open("w", encoding="utf-8")
        self._verbose_console = bool(verbose_console)

    def close(self) -> None:
        """Close the log file."""
        try:
            self._fh.close()
        except Exception:
            pass

    def _write(self, msg: str) -> None:
        """Write message to file."""
        self._fh.write(msg)
        if not msg.endswith("\n"):
            self._fh.write("\n")
        self._fh.flush()

    def info(self, msg: str = "") -> None:
        """Log INFO message (console + file)."""
        self._write(msg)
        if self._verbose_console:
            print(msg)

    def debug(self, msg: str = "") -> None:
        """Log DEBUG message (file only)."""
        self._write(msg)


# ============================================================================
# CSV Writers
# ============================================================================

def _write_summary_csv(
    path: Path,
    rows: List[Dict[str, float]],
    *,
    report_zero_tol: float,
) -> None:
    """
    Write CEC-style summary CSV with formatted scientific notation.
    
    Output format matches the reference MATLAB implementation exactly.
    """
    ensure_dir(path.parent)
    fieldnames = ["Function", "Best", "Median", "Mean", "Worst", "SD"]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "Function": int(r["Function"]),
                    "Best": format_sci(r["Best"], tol=report_zero_tol),
                    "Median": format_sci(r["Median"], tol=report_zero_tol),
                    "Mean": format_sci(r["Mean"], tol=report_zero_tol),
                    "Worst": format_sci(r["Worst"], tol=report_zero_tol),
                    "SD": format_sci(r["SD"], tol=report_zero_tol),
                }
            )


def _write_run_errors_csv(path: Path, errors: List[float]) -> None:
    """Write per-run errors to CSV (full precision for reproducibility)."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Run", "Error"])
        for i, e in enumerate(errors, start=1):
            w.writerow([i, f"{float(e):.16e}"])


def _write_gen_log_csv(path: Path, logs: List[GSKGenerationLog]) -> None:
    """Write generation-by-generation diagnostics to CSV."""
    ensure_dir(path.parent)
    fieldnames = [
        "gen",
        "evals_used",
        "best_fitness",
        "diversity",
        "stagnation",
        "stage",
        "jr_mask_rate",
        "sr_mask_rate",
        "KF",
        "KR",
        "Kexp",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for log in logs:
            w.writerow(asdict(log))


# ============================================================================
# Utility Functions
# ============================================================================

def _median_run_id(errors: List[float]) -> int:
    """
    Find the 1-based run ID that achieved the median final error.
    
    Used to select which run's convergence curve to export.
    Median is preferred over best to show "typical" algorithm behavior.
    """
    arr = np.asarray(list(errors), dtype=np.float64)
    if arr.size == 0:
        return 1
    order = np.argsort(arr, kind="mergesort")
    median_idx = int((arr.size - 1) // 2)
    run0 = int(order[median_idx])
    return run0 + 1


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_cec2017_experiments(
    *,
    cfg: Config,
    cec_root: Optional[Path] = None,
    smoke: bool = False,
    verbose: bool = True,
) -> None:
    """
    Run GSK on CEC2017 benchmark and save results.
    
    This is the main entry point for running experiments. It orchestrates
    the full benchmark evaluation across all specified dimensions and
    functions.
    
    Parameters
    ----------
    cfg : Config
        Experiment configuration specifying dimensions, functions,
        algorithm parameters, and output options.
        
    cec_root : Path, optional
        Override path to external CEC2017 implementation.
        If None, uses cfg.cec_root or searches standard locations.
        
    smoke : bool, default=False
        If True, run minimal subset for quick sanity check:
        - D=10 only
        - F1, F3 only
        - All configured runs
        
    verbose : bool, default=True
        Print progress to console.
        
    Outputs
    -------
    Results are written to: <project_root>/results/<alg_name>/
    
    - summary/Summary_All_Results_D{D}.csv: Statistics per function
    - summary/environment.json: Environment metadata for reproducibility
    - summary/run_config.json: Configuration used for this run
    - gen_logs/GenLog_*.csv: Per-generation diagnostics (if enabled)
    - curves/Figure_*.csv: Convergence curves for median runs (if enabled)
    
    Example
    -------
    >>> cfg = Config(
    ...     runs=51,
    ...     dims=(10, 30),
    ...     funcs=tuple(range(1, 31)),
    ... )
    >>> run_cec2017_experiments(cfg=cfg)
    """
    
    # ========================================================================
    # Setup: Resolve Paths
    # ========================================================================
    
    project_root = cfg.project_root
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]

    # Create output directories
    results_root = ensure_dir(project_root / "results" / cfg.alg_name)
    summary_dir = ensure_dir(results_root / "summary")
    gen_logs_dir = ensure_dir(results_root / "gen_logs")
    curves_dir = ensure_dir(results_root / "curves")
    ensure_dir(curves_dir / "graphs")

    # ========================================================================
    # Setup: Determinism Controls
    # ========================================================================
    
    apply_thread_env(
        force_single_thread=cfg.force_single_thread,
        override=cfg.override_thread_env,
    )

    # ========================================================================
    # Setup: Import CEC2017 Functions
    # ========================================================================
    
    cec_source = ensure_cec2017_importable(
        project_root=project_root,
        override=cec_root or cfg.cec_root,
    )

    # ========================================================================
    # Setup: Save Metadata
    # ========================================================================
    
    run_stamp = timestamp_now()
    env_meta: EnvironmentMetadata = collect_environment_metadata(
        include_numpy_config_full=False
    )

    write_json(summary_dir / "environment.json", env_metadata_as_dict(env_meta))
    write_json(
        summary_dir / "run_config.json",
        cfg.to_dict(extra={"run_stamp": run_stamp, "cec_source": str(cec_source)}),
    )

    # ========================================================================
    # Setup: Determine Experiment Scope
    # ========================================================================
    
    dims: Tuple[int, ...] = tuple(int(d) for d in cfg.dims)
    funcs: Tuple[int, ...] = tuple(int(f) for f in cfg.funcs)
    exclude: Tuple[int, ...] = tuple(int(f) for f in cfg.exclude_funcs)

    # Smoke test: minimal configuration
    if smoke:
        dims = (10,)
        funcs = (1, 3)

    funcs_to_run = tuple(f for f in funcs if f not in exclude)
    enable_curves_dims = set(cfg.enabled_curve_dims(dims))

    if verbose:
        print("")

    # ========================================================================
    # Main Loop: Each Dimension
    # ========================================================================
    
    for D in dims:
        log_path = summary_dir / f"{cfg.alg_name}_D{D:02d}_log_{run_stamp}.txt"
        log = DualLogger(log_path, verbose_console=verbose)

        # Compute evaluation budget
        max_nfes = (
            int(cfg.max_nfes_override)
            if cfg.max_nfes_override is not None
            else int(10000 * D)
        )
        
        # ====================================================================
        # Print Experiment Header (eye-friendly box format)
        # ====================================================================
        
        log.info("")
        log.info("╔" + "═" * 58 + "╗")
        log.info(f"║{'GSK Baseline · CEC2017 · D=' + str(D):^58}║")
        log.info("╠" + "═" * 58 + "╣")
        log.info(f"║  {len(funcs_to_run)} functions × {cfg.runs} runs │ Budget: {max_nfes:,} NFEs".ljust(58) + "║")
        log.info(f"║  KF={cfg.KF}  KR={cfg.KR}  Kexp={cfg.Kexp}  Pop={cfg.pop_size}".ljust(58) + "║")
        log.info("╚" + "═" * 58 + "╝")
        log.info("")

        # Detailed config goes to file only
        log.debug("=" * 72)
        log.debug("DETAILED CONFIGURATION")
        log.debug("=" * 72)
        log.debug(f"project_root    : {project_root}")
        log.debug(f"cec_source      : {cec_source}")
        log.debug(f"funcs_to_run    : {funcs_to_run}")
        log.debug(f"excluded        : {exclude}")
        log.debug(f"bounds          : {cfg.bounds}")
        log.debug(f"zero_threshold  : {cfg.val_to_reach:g}")
        log.debug(f"base_seed       : {cfg.base_seed}")
        log.debug(f"stride_run      : {cfg.stride_run}")
        log.debug(f"curves_dims     : {tuple(sorted(enable_curves_dims))}")
        log.debug("")
        log.debug("ENVIRONMENT")
        log.debug("-" * 72)
        log.debug(format_environment_metadata(env_meta, header=False))
        log.debug("")

        summary_rows: List[Dict[str, float]] = []
        total_funcs = len(funcs_to_run)

        # ====================================================================
        # Print Results Table Header
        # ====================================================================
        
        log.info("┌" + "─" * 8 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┐")
        log.info(f"│{'Func':^8}│{'Best':^12}│{'Median':^12}│{'Mean':^12}│{'Worst':^12}│{'SD':^12}│")
        log.info("├" + "─" * 8 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤")

        # ====================================================================
        # Inner Loop: Each Function
        # ====================================================================
        
        for func_idx, func_id in enumerate(funcs_to_run, 1):
            log.debug(
                f"[{func_idx}/{total_funcs}] F{func_id:02d} (D={D}): running {cfg.runs} trials ..."
            )

            # Get objective function and optimal value
            objective, f_opt = cec2017_function(
                func_id=func_id, dim=D, bounds=cfg.bounds
            )

            run_errors: List[float] = []

            # ================================================================
            # Innermost Loop: Each Run
            # ================================================================
            
            for run_id in range(1, int(cfg.runs) + 1):
                # Compute deterministic seed for this (func, run) pair
                seed = seed_for_run(
                    base_seed=cfg.base_seed,
                    stride_run=cfg.stride_run,
                    func_id=func_id,
                    run_id=run_id,
                )

                # Optional: collect generation logs
                gen_logs: List[GSKGenerationLog] = []

                def gen_callback(gl: GSKGenerationLog) -> None:
                    gen_logs.append(gl)

                # Configure GSK for this run
                gsk_cfg = GSKConfig(
                    dim=D,
                    pop_size=cfg.pop_size,
                    bounds=cfg.bounds,
                    max_nfes=max_nfes,
                    KF=cfg.KF,
                    KR=cfg.KR,
                    Kexp=cfg.Kexp,
                    seed=seed,
                )

                # Run optimization
                res = gsk_optimize(
                    objective=objective,
                    config=gsk_cfg,
                    generation_callback=gen_callback if cfg.write_gen_logs else None,
                )

                # Compute error (best found - optimal)
                best_f = float(res.best_f)
                err = max(0.0, best_f - float(f_opt))
                
                # Apply per-run threshold (matches MATLAB reference)
                # Errors below val_to_reach are considered "solved"
                if err < cfg.val_to_reach:
                    err = 0.0
                    
                run_errors.append(err)

                log.debug(
                    f"  Run {run_id:02d}/{cfg.runs}: seed={seed} nfes={res.nfes_used} "
                    f"best_f={best_f:.6e} err={err:.6e}"
                )

                # Save generation log if enabled
                if cfg.write_gen_logs and gen_logs:
                    gen_log_path = (
                        gen_logs_dir
                        / f"GenLog_{cfg.alg_name}_F{func_id}_D{D}_Run{run_id}.csv"
                    )
                    _write_gen_log_csv(gen_log_path, gen_logs)

            # ================================================================
            # Compute Per-Function Statistics
            # ================================================================
            
            arr = np.asarray(run_errors, dtype=np.float64)
            best = float(np.min(arr))
            median = float(np.median(arr))
            mean = float(np.mean(arr))
            worst = float(np.max(arr))
            sd = float(np.std(arr, ddof=1))  # Sample std (matches reference)

            # Apply report tolerance for display
            best_r = zero_small(best, tol=cfg.report_zero_tol)
            median_r = zero_small(median, tol=cfg.report_zero_tol)
            mean_r = zero_small(mean, tol=cfg.report_zero_tol)
            worst_r = zero_small(worst, tol=cfg.report_zero_tol)
            sd_r = zero_small(sd, tol=cfg.report_zero_tol)

            summary_rows.append(
                {
                    "Function": float(func_id),
                    "Best": best_r,
                    "Median": median_r,
                    "Mean": mean_r,
                    "Worst": worst_r,
                    "SD": sd_r,
                }
            )

            # Print result row
            log.info(
                f"│{'F' + str(func_id).zfill(2):^8}│"
                f"{format_sci(best_r, tol=cfg.report_zero_tol):^12}│"
                f"{format_sci(median_r, tol=cfg.report_zero_tol):^12}│"
                f"{format_sci(mean_r, tol=cfg.report_zero_tol):^12}│"
                f"{format_sci(worst_r, tol=cfg.report_zero_tol):^12}│"
                f"{format_sci(sd_r, tol=cfg.report_zero_tol):^12}│"
            )

            # Save per-run errors
            _write_run_errors_csv(
                gen_logs_dir / f"RunErrors_{cfg.alg_name}_F{func_id}_D{D}.csv",
                run_errors,
            )

            # ================================================================
            # Export Convergence Curve for Median Run
            # ================================================================
            
            if D in enable_curves_dims:
                med_run = _median_run_id(run_errors)
                seed_med = seed_for_run(
                    base_seed=cfg.base_seed,
                    stride_run=cfg.stride_run,
                    func_id=func_id,
                    run_id=med_run,
                )

                gsk_cfg_med = GSKConfig(
                    dim=D,
                    pop_size=cfg.pop_size,
                    bounds=cfg.bounds,
                    max_nfes=max_nfes,
                    KF=cfg.KF,
                    KR=cfg.KR,
                    Kexp=cfg.Kexp,
                    seed=seed_med,
                )

                result_tuple = gsk_optimize(
                    objective=objective, config=gsk_cfg_med, return_history=True
                )
                res_med, best_history = result_tuple

                # Compute error curve
                best_f_hist = np.asarray(best_history, dtype=np.float64)
                err_hist = np.maximum(0.0, best_f_hist - float(f_opt))

                # Log10 with floor for visualization
                err_for_log = np.maximum(err_hist, float(cfg.val_to_reach))
                log10_err = np.log10(err_for_log)

                # Save curve
                curve_path = curves_dir / f"Figure_F{func_id}_D{D}_Run#{med_run}.csv"
                ensure_dir(curve_path.parent)
                with curve_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["Eval", "BestError", "Log10Error"])
                    for i in range(err_hist.size):
                        w.writerow([i + 1, f"{err_hist[i]:.16e}", f"{log10_err[i]:.16e}"])

        # ====================================================================
        # Print Table Footer and Summary
        # ====================================================================
        
        log.info("└" + "─" * 8 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┘")
        log.info("")
        log.info(f"✓ Completed {total_funcs} functions × {cfg.runs} runs = {total_funcs * cfg.runs} total runs")
        log.info("")

        # ====================================================================
        # Write Summary CSV
        # ====================================================================
        
        summary_csv = summary_dir / f"Summary_All_Results_D{D}.csv"
        _write_summary_csv(summary_csv, summary_rows, report_zero_tol=cfg.report_zero_tol)

        # ====================================================================
        # Compare Against Baseline
        # ====================================================================
        
        baseline_dir = project_root / "previous_results" / cfg.baseline_compare_alg
        baseline_csv = baseline_dir / f"Summary_All_Results_D{D}.csv"

        log.info("")
        if baseline_csv.exists():
            comp = generate_mean_error_comparison_text(
                old_summary_csv=baseline_csv,
                new_summary_csv=summary_csv,
                old_label=cfg.baseline_compare_label,
                new_label=cfg.baseline_compare_label_new,
                dims=D,
                excluded_funcs=exclude,
                zero_tol=cfg.report_zero_tol,
            )
            log.info(comp)
        else:
            log.info(
                f"[WARN] Baseline summary not found: {baseline_csv} (skipping mean comparison)."
            )

        log.close()
