"""
Experiment Configuration for GSK Algorithm
==========================================

This module provides a centralized configuration dataclass for running
GSK experiments on CEC2017 benchmarks.

Configuration Hierarchy
-----------------------
The Config class serves as the single source of truth for all experiment
parameters. Default values come from constants.py and can be overridden:

    # Use all defaults (standard CEC2017 experiment)
    cfg = Config()
    
    # Override specific parameters
    cfg = Config(
        runs=10,           # Quick test with fewer runs
        dims=(10,),        # Single dimension
        funcs=(1, 5, 9),   # Subset of functions
    )

Configuration Categories
------------------------

1. **Experiment Design**
   - runs: Number of independent runs (51 for CEC2017)
   - dims: Problem dimensions to test
   - funcs: Function IDs to evaluate
   - exclude_funcs: Functions to skip (F2 is unstable)

2. **Algorithm Parameters**
   - pop_size: Population size (100)
   - bounds: Search space limits (-100, 100)
   - KF, KR, Kexp: GSK hyperparameters

3. **Reproducibility**
   - base_seed, stride_run: Seeding scheme
   - force_single_thread: BLAS thread control

4. **Output Control**
   - verbose: Progress printing
   - write_gen_logs: Per-generation diagnostics
   - enable_curves_*: Convergence curve export

5. **Numerical Conventions**
   - val_to_reach: Per-run success threshold (1e-8)
   - report_zero_tol: Display tolerance (1e-7)

Path Configuration
------------------
Two paths are needed for experiments:

- project_root: Where to write results (./results/ subdirectory)
- cec_root: Location of CEC2017 benchmark files

If not specified, reasonable defaults are used based on working directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from .constants import (
    ALG_NAME,
    DEFAULT_BASE_SEED,
    DEFAULT_BOUNDS,
    DEFAULT_DIMS,
    DEFAULT_ENABLE_CURVES_10,
    DEFAULT_ENABLE_CURVES_30,
    DEFAULT_ENABLE_CURVES_50,
    DEFAULT_ENABLE_CURVES_100,
    DEFAULT_EXCLUDE_FUNCS,
    DEFAULT_FUNCS,
    DEFAULT_KEXP,
    DEFAULT_KF,
    DEFAULT_KR,
    DEFAULT_POP_SIZE,
    DEFAULT_RUNS,
    DEFAULT_STRIDE_RUN,
    REPORT_ZERO_TOL,
    VAL_TO_REACH,
)


@dataclass
class Config:
    """
    Experiment configuration for running baseline GSK on CEC2017.
    
    This dataclass holds all parameters needed to run a complete
    CEC2017 benchmark experiment with the GSK algorithm.
    
    Attributes
    ----------
    
    **Identity & Paths**
    
    alg_name : str, default="gsk"
        Algorithm identifier for output files.
        
    project_root : Path or None
        Project root directory. Results written to project_root/results/.
        
    cec_root : Path or None
        CEC2017 benchmark root directory containing function definitions.
        
    **Baseline Comparison Labels**
    
    baseline_compare_alg : str, default="gsk"
        Algorithm name for baseline comparison CSV lookup.
        
    baseline_compare_label : str, default="GSK"
        Label for reference baseline in comparison tables.
        
    baseline_compare_label_new : str, default="gsk"
        Label for new results in comparison tables.
        
    **Experiment Design**
    
    runs : int, default=51
        Number of independent runs per function.
        CEC2017 requires 51 for statistical significance.
        
    dims : tuple of int, default=(10, 30, 50, 100)
        Problem dimensions to test.
        
    funcs : tuple of int, default=(1, 2, ..., 30)
        CEC2017 function IDs to evaluate.
        
    exclude_funcs : tuple of int, default=(2,)
        Function IDs to skip. F2 is excluded due to instability.
        
    **GSK Algorithm Parameters**
    
    pop_size : int, default=100
        Population size (number of candidate solutions).
        
    bounds : tuple of float, default=(-100.0, 100.0)
        Search space bounds for all dimensions.
        
    max_nfes_override : int or None
        Override default budget (10000 × D) if specified.
        
    KF : float, default=0.5
        Knowledge Factor - mutation step size.
        
    KR : float, default=0.9
        Knowledge Ratio - crossover probability.
        
    Kexp : float, default=10.0
        Knowledge Rate exponent - exploration/exploitation balance.
        
    **Numerical Conventions**
    
    val_to_reach : float, default=1e-8
        Per-run success threshold. Errors below this are set to 0.0.
        
    report_zero_tol : float, default=1e-7
        Display tolerance. Values ≤ this shown as "0.00E+00".
        
    **Reproducibility**
    
    base_seed : int, default=123456
        Base random seed for seeding scheme.
        
    stride_run : int, default=9973
        Seed stride between functions (prime number).
        
    force_single_thread : bool, default=True
        Force single-threaded BLAS for determinism.
        
    override_thread_env : bool, default=False
        Override existing thread environment variables.
        
    **Output Control**
    
    verbose : bool, default=True
        Print progress to console.
        
    write_gen_logs : bool, default=True
        Write per-generation diagnostic logs.
        
    **Convergence Curve Export**
    
    enable_curves_all : bool, default=False
        Enable curve export for all dimensions.
        
    enable_curves_10 : bool, default=True
        Enable curve export for D=10.
        
    enable_curves_30 : bool, default=True
        Enable curve export for D=30.
        
    enable_curves_50 : bool, default=True
        Enable curve export for D=50.
        
    enable_curves_100 : bool, default=False
        Enable curve export for D=100 (large files).
        
    Examples
    --------
    Standard CEC2017 experiment:
    
    >>> cfg = Config()
    >>> # 51 runs × 29 functions × 4 dimensions = 5,916 total runs
    
    Quick smoke test:
    
    >>> cfg = Config(
    ...     runs=3,
    ...     dims=(10,),
    ...     funcs=(1, 5, 9),
    ... )
    
    Custom parameters:
    
    >>> cfg = Config(
    ...     KF=0.6,
    ...     KR=0.8,
    ...     pop_size=150,
    ... )
    """

    # ========================================================================
    # Identity & Paths
    # ========================================================================
    
    alg_name: str = ALG_NAME
    project_root: Optional[Path] = None
    cec_root: Optional[Path] = None

    # ========================================================================
    # Baseline Comparison Labels
    # ========================================================================
    
    baseline_compare_alg: str = "gsk"
    baseline_compare_label: str = "GSK"
    baseline_compare_label_new: str = "gsk"

    # ========================================================================
    # Experiment Design
    # ========================================================================
    
    runs: int = DEFAULT_RUNS
    dims: Tuple[int, ...] = DEFAULT_DIMS
    funcs: Tuple[int, ...] = DEFAULT_FUNCS
    exclude_funcs: Tuple[int, ...] = DEFAULT_EXCLUDE_FUNCS

    # ========================================================================
    # GSK Algorithm Parameters
    # ========================================================================
    
    pop_size: int = DEFAULT_POP_SIZE
    bounds: Tuple[float, float] = DEFAULT_BOUNDS
    max_nfes_override: Optional[int] = None
    
    KF: float = DEFAULT_KF
    KR: float = DEFAULT_KR
    Kexp: float = DEFAULT_KEXP

    # ========================================================================
    # Numerical Conventions
    # ========================================================================
    
    report_zero_tol: float = REPORT_ZERO_TOL
    val_to_reach: float = VAL_TO_REACH

    # ========================================================================
    # Reproducibility
    # ========================================================================
    
    base_seed: int = DEFAULT_BASE_SEED
    stride_run: int = DEFAULT_STRIDE_RUN
    force_single_thread: bool = True
    override_thread_env: bool = False

    # ========================================================================
    # Output Control
    # ========================================================================
    
    verbose: bool = True
    write_gen_logs: bool = True

    # ========================================================================
    # Convergence Curve Export
    # ========================================================================
    
    enable_curves_all: bool = False
    enable_curves_10: bool = DEFAULT_ENABLE_CURVES_10
    enable_curves_30: bool = DEFAULT_ENABLE_CURVES_30
    enable_curves_50: bool = DEFAULT_ENABLE_CURVES_50
    enable_curves_100: bool = DEFAULT_ENABLE_CURVES_100

    # ========================================================================
    # Methods
    # ========================================================================

    def enabled_curve_dims(
        self,
        requested_dims: Sequence[int],
    ) -> Tuple[int, ...]:
        """
        Return dimensions that have curve export enabled.
        
        Parameters
        ----------
        requested_dims : sequence of int
            Dimensions being tested in the experiment.
            
        Returns
        -------
        tuple of int
            Subset of requested_dims with curve export enabled.
            
        Example
        -------
        >>> cfg = Config(enable_curves_10=True, enable_curves_30=False)
        >>> cfg.enabled_curve_dims([10, 30, 50])
        (10, 50)
        """
        req = {int(d) for d in requested_dims}
        
        # If enable_curves_all, return all requested
        if self.enable_curves_all:
            return tuple(sorted(req))

        # Build set of enabled dimensions
        enabled: set[int] = set()
        if self.enable_curves_10:
            enabled.add(10)
        if self.enable_curves_30:
            enabled.add(30)
        if self.enable_curves_50:
            enabled.add(50)
        if self.enable_curves_100:
            enabled.add(100)

        return tuple(sorted(enabled.intersection(req)))

    def to_dict(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for JSON serialization.
        
        Parameters
        ----------
        extra : dict, optional
            Additional key-value pairs to include.
            
        Returns
        -------
        dict
            Configuration as dictionary with Path objects converted to strings.
        """
        d = asdict(self)
        
        # Convert Path objects to strings
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
                
        # Add extra fields if provided
        if extra:
            d.update(extra)
            
        return d


__all__ = ["Config"]
