"""
GSK Algorithm Constants and Configuration Defaults
===================================================

This module centralizes all constants used in the GSK implementation,
providing a single reference for default values, tolerances, and conventions.

Constant Categories
-------------------

1. **Algorithm Parameters**: Default GSK hyperparameters (KF, KR, K)
2. **Experiment Design**: CEC2017 standard settings (runs, dimensions, functions)
3. **Budget Convention**: Evaluation limits per CEC2017 rules
4. **Tolerance Thresholds**: Numerical precision for success/reporting
5. **Reproducibility**: Seeding scheme for deterministic experiments

CEC2017 Competition Standards
-----------------------------
This implementation follows CEC2017 benchmark conventions:

- **Dimensions**: D ∈ {10, 30, 50, 100}
- **Functions**: F1-F30 (F2 excluded due to instability)
- **Runs**: 51 independent runs per function
- **Budget**: 10000 × D function evaluations
- **Bounds**: [-100, +100] for all dimensions
- **Success threshold**: Error < 10⁻⁸ is considered "solved"

Two-Threshold System
--------------------
We distinguish between two tolerances:

1. **VAL_TO_REACH (10⁻⁸)**: Per-run success threshold
   - If error < 10⁻⁸ at end of run → error = 0.0
   - Used for individual run classification
   - Matches MATLAB reference implementation

2. **REPORT_ZERO_TOL (10⁻⁷)**: Display/reporting tolerance
   - Mean errors ≤ 10⁻⁷ are shown as "0.00E+00"
   - Prevents displaying misleading tiny values
   - Example: mean of [0, 0, 0, ..., 2.5e-8] ≈ 5e-10 → shows as 0.00E+00

Why this matters:
    Run results:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 2.15e-8]  (51 runs)
                   ↓ Mean ≈ 4.2e-10 (tiny but nonzero)
    
    Without two thresholds: Mean = 4.20E-10 (confusing - looks non-zero)
    With REPORT_ZERO_TOL:   Mean = 0.00E+00 (clear - effectively solved)
"""

from __future__ import annotations

from typing import Tuple


# ============================================================================
# Algorithm Identification
# ============================================================================

ALG_NAME: str = "gsk"
"""Algorithm name used in output files and reports."""


# ============================================================================
# Default GSK Hyperparameters
# ============================================================================
# These are the standard values from the original GSK paper.
# Modification is NOT recommended for baseline experiments.

DEFAULT_POP_SIZE: int = 100
"""
Population size (number of candidate solutions).

The original paper recommends 100 for all dimensions.
This provides good balance between exploration (diversity)
and exploitation (focused search).
"""

DEFAULT_KF: float = 0.5
"""
Knowledge Factor - controls mutation step size.

Range: [0, 1]
- KF = 0: No mutation (offspring = parent)
- KF = 1: Maximum mutation (full step toward target)
- KF = 0.5: Balanced step (recommended default)

Affects both junior and senior phases equally.
"""

DEFAULT_KR: float = 0.9
"""
Knowledge Ratio - crossover probability.

Range: [0, 1]
- KR = 0: No crossover (offspring = parent for all dimensions)
- KR = 1: Full crossover (all dimensions potentially modified)
- KR = 0.9: Most dimensions modified (recommended default)

Higher KR promotes faster exploration but may lose good partial solutions.
"""

DEFAULT_KEXP: float = 10.0
"""
Knowledge Rate exponent (K) - controls exploration/exploitation balance.

Controls the transition from junior (exploration) to senior (exploitation) phase.

D_junior = ceil(D × (1 - g/G_max)^K)

- K = 1: Linear transition
- K = 10: Rapid transition (stays in exploration longer, then quick shift)
- K > 10: Even more exploration-focused early on

With K = 10:
- At 50% budget: D_junior ≈ D × 0.001 (almost all senior)
- Most exploration happens in first 10-20% of budget
"""

DEFAULT_BOUNDS: Tuple[float, float] = (-100.0, 100.0)
"""
Search space bounds for CEC2017 benchmark.

All dimensions share the same bounds [-100, +100].
This is standard across all CEC2017 functions.
"""


# ============================================================================
# CEC2017 Experiment Design
# ============================================================================

DEFAULT_RUNS: int = 51
"""
Number of independent runs per function.

CEC2017 requires 51 runs for statistical significance.
Odd number allows meaningful median calculation.
"""

DEFAULT_DIMS: Tuple[int, ...] = (10, 30, 50, 100)
"""
Problem dimensions to test.

CEC2017 specifies these four dimension levels:
- D=10:  Low-dimensional (easier)
- D=30:  Medium (standard benchmark)
- D=50:  High-dimensional (challenging)
- D=100: Very high-dimensional (very challenging)
"""

DEFAULT_FUNCS: Tuple[int, ...] = tuple(range(1, 31))
"""
CEC2017 function IDs (F1 through F30).

Functions are grouped by type:
- F1-F3:   Unimodal
- F4-F10:  Simple multimodal
- F11-F20: Hybrid
- F21-F30: Composition
"""

DEFAULT_EXCLUDE_FUNCS: Tuple[int, ...] = (2,)
"""
Functions to exclude from experiments.

F2 is excluded because it is numerically unstable and produces
inconsistent results across platforms. This is standard practice
in CEC2017 experiments.
"""


# ============================================================================
# Budget Convention
# ============================================================================

MAX_NFES_MULT: int = 10000
"""
Budget multiplier per dimension.

max_nfes = 10000 × D

For D=10:  100,000 evaluations
For D=30:  300,000 evaluations
For D=50:  500,000 evaluations
For D=100: 1,000,000 evaluations
"""

F_OPT_SHIFT: float = 100.0
"""
Optimal value shift pattern for CEC2017 functions.

Optimal values: F1→100, F2→200, ..., F30→3000
Formula: f_opt(i) = 100 × i

Error is computed as: error = f_best - f_opt
"""


# ============================================================================
# Tolerance Thresholds
# ============================================================================

VAL_TO_REACH: float = 1e-8
"""
Per-run success threshold (Value to Reach).

If final error < 1e-8 → error is set to 0.0 for that run.
This matches CEC2017 convention for "solved" problems.

Applied at the end of each individual run:
    if error < VAL_TO_REACH:
        error = 0.0  # Considered solved
"""

REPORT_ZERO_TOL: float = 1e-7
"""
Display tolerance for tables and reports.

Mean/median values with |x| ≤ 1e-7 are displayed as "0.00E+00".
This prevents confusing displays like "4.20E-10" for effectively
zero values that arise from averaging mostly-zero runs.

Used only for formatting; does not affect actual stored values.
"""

CURVE_LOG10_CLAMP: float = 1e-8
"""
Minimum value for log-scale convergence curves.

Values below this are clamped to 1e-8 to avoid log(0) issues
and provide clean visualization. Matches VAL_TO_REACH.
"""


# ============================================================================
# Reproducibility: Seeding Scheme
# ============================================================================

DEFAULT_BASE_SEED: int = 123456
"""
Base random seed for reproducibility.

All runs derive their seeds from this base value:
    seed(func, run) = BASE_SEED + (func - 1) × STRIDE_RUN + run
"""

DEFAULT_STRIDE_RUN: int = 9973
"""
Seed stride between functions.

Using a prime number ensures seeds for different function/run
combinations don't overlap. 9973 is chosen because:
- It's prime (no common factors)
- It's larger than typical run counts (51)
- Provides good separation in seed space
"""


# ============================================================================
# Curve Export Defaults
# ============================================================================

DEFAULT_ENABLE_CURVES_10: bool = True
"""Enable convergence curve export for D=10."""

DEFAULT_ENABLE_CURVES_30: bool = True
"""Enable convergence curve export for D=30."""

DEFAULT_ENABLE_CURVES_50: bool = True
"""Enable convergence curve export for D=50."""

DEFAULT_ENABLE_CURVES_100: bool = False
"""
Disable convergence curve export for D=100 by default.

D=100 curves are large (1M evaluations each) and rarely needed.
Enable manually if required for specific analysis.
"""
