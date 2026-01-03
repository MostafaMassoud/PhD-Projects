"""
Result Validation and Comparison Tools
======================================

This module provides tools for validating GSK experiment results against
reference baselines and generating comparison reports.

Purpose
-------
When developing or modifying the GSK implementation, it's critical to verify
that results match the reference MATLAB implementation. This module provides:

1. **Mean Error Comparison**: Side-by-side comparison showing improvement/regression
2. **Validation**: Exact match verification for reproducibility testing

Comparison Table Format
-----------------------
The mean error comparison produces a formatted table:

    ┌────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────┐
    │  Func  │   GSK Mean   │   gsk Mean   │ Δ (New-Old)  │  % Improve   │  Status  │
    ├────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────┤
    │  F01   │   0.00E+00   │   0.00E+00   │   0.00E+00   │      —       │    ●     │
    │  F05   │   2.05E+01   │   1.97E+01   │  -8.00E-01   │    +3.9%     │    ▲     │
    │  F11   │   0.00E+00   │   2.15E-08   │   2.15E-08   │     N/A      │    ▼     │
    └────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────┘
    
      Summary: ▲ Improved: 1  ● Same: 27  ▼ Worse: 1

Status Symbols:
- ▲ (Improved): New mean is better (lower) than reference
- ● (Same): Means are equal within tolerance
- ▼ (Worse): New mean is worse (higher) than reference

Special Cases
-------------
When comparing against zero baselines:

- Both zero → "—" (dash), counted as Same
- Old zero, New non-zero → "N/A", counted as Worse (regression from solved)
- Old non-zero, New zero → Improvement (100%)

Tolerance Handling
------------------
The comparison uses report tolerance (default 1e-7) for zero-equivalence:
- Values ≤ 1e-7 are treated as zero
- This prevents false regressions from numerical noise
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .utils import ZERO_TOL_DEFAULT, format_sci, quantize_sci


# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class SummaryRow:
    """
    Single row from a CEC2017 summary CSV.
    
    Represents the statistics for one function from one dimension.
    
    Attributes
    ----------
    func_id : int
        Function ID (1-30).
    best : float
        Best (minimum) error across all runs.
    median : float
        Median error across all runs.
    mean : float
        Mean error across all runs.
    worst : float
        Worst (maximum) error across all runs.
    sd : float
        Standard deviation of errors (sample std, ddof=1).
    """
    func_id: int
    best: float
    median: float
    mean: float
    worst: float
    sd: float


# ============================================================================
# CSV Loading
# ============================================================================

def load_summary_csv(path: Path) -> Dict[int, SummaryRow]:
    """
    Load a CEC2017 summary CSV into a dictionary.
    
    Parameters
    ----------
    path : Path
        Path to Summary_All_Results_D{D}.csv file.
        
    Returns
    -------
    dict
        Mapping from function ID to SummaryRow.
        
    Example
    -------
    >>> rows = load_summary_csv(Path("Summary_All_Results_D10.csv"))
    >>> rows[1].mean  # Mean error for F1
    0.0
    """
    rows: Dict[int, SummaryRow] = {}
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for line in r:
            fid = int(line["Function"])
            rows[fid] = SummaryRow(
                func_id=fid,
                best=float(line["Best"]),
                median=float(line["Median"]),
                mean=float(line["Mean"]),
                worst=float(line["Worst"]),
                sd=float(line["SD"]),
            )
    return rows


# ============================================================================
# Internal Utilities
# ============================================================================

def _iter_func_ids(
    old: Dict[int, SummaryRow],
    new: Dict[int, SummaryRow],
    excluded_funcs: Sequence[int],
) -> List[int]:
    """
    Get function IDs present in BOTH datasets, excluding specified ones.
    
    Only returns functions that exist in both old and new summaries,
    enabling comparison of partial results.
    """
    excluded = set(int(x) for x in excluded_funcs)
    common = set(old.keys()) & set(new.keys())  # Intersection
    return sorted(common - excluded)


# ============================================================================
# Mean Error Comparison
# ============================================================================

def generate_mean_error_comparison_text(
    old_summary_csv: Path,
    new_summary_csv: Path,
    *,
    old_label: str = "GSK",
    new_label: str = "gsk",
    dims: Optional[int] = None,
    excluded_funcs: Sequence[int] = (2,),
    zero_tol: float = ZERO_TOL_DEFAULT,
) -> str:
    """
    Generate a formatted mean error comparison table.
    
    Compares mean errors between reference (old) and new results,
    showing improvement percentage and status for each function.
    
    Parameters
    ----------
    old_summary_csv : Path
        Path to reference summary CSV (e.g., from MATLAB).
        
    new_summary_csv : Path
        Path to new summary CSV (from this implementation).
        
    old_label : str, default="GSK"
        Column header label for reference results.
        
    new_label : str, default="gsk"
        Column header label for new results.
        
    dims : int, optional
        Problem dimension (shown in title if provided).
        
    excluded_funcs : sequence of int, default=(2,)
        Function IDs to exclude from comparison.
        
    zero_tol : float, default=1e-7
        Tolerance for treating values as zero.
        
    Returns
    -------
    str
        Formatted comparison table as multi-line string.
        
    Notes
    -----
    Improvement calculation:
        % Improve = ((old - new) / old) × 100
        
    Status assignment:
    - Both zero: Same (●)
    - Old zero, new non-zero: Worse (▼)
    - Improvement > 0.01%: Improved (▲)
    - Regression > 0.01%: Worse (▼)
    - Otherwise: Same (●)
    
    Example
    -------
    >>> text = generate_mean_error_comparison_text(
    ...     old_summary_csv=Path("reference/Summary_D10.csv"),
    ...     new_summary_csv=Path("results/Summary_D10.csv"),
    ...     dims=10,
    ... )
    >>> print(text)
    """
    old_rows = load_summary_csv(old_summary_csv)
    new_rows = load_summary_csv(new_summary_csv)

    lines: List[str] = []
    
    # ========================================================================
    # Title
    # ========================================================================
    
    title = f"Mean Error Comparison: {old_label} vs {new_label}"
    if dims is not None:
        title += f" (D={dims})"
    
    lines.append("")
    lines.append("╔" + "═" * 78 + "╗")
    lines.append(f"║{title:^78}║")
    lines.append("╚" + "═" * 78 + "╝")
    lines.append(f"  Zero threshold: {zero_tol:g} (errors ≤ this are considered solved)")
    lines.append("")

    # ========================================================================
    # Table Header
    # ========================================================================
    
    lines.append("┌" + "─" * 8 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 10 + "┐")
    lines.append(f"│{'Func':^8}│{old_label + ' Mean':^14}│{new_label + ' Mean':^14}│{'Δ (New-Old)':^14}│{'% Improve':^14}│{'Status':^10}│")
    lines.append("├" + "─" * 8 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 10 + "┤")

    # ========================================================================
    # Table Rows
    # ========================================================================
    
    improved_count = 0
    same_count = 0
    worse_count = 0

    for fid in _iter_func_ids(old_rows, new_rows, excluded_funcs):
        old_mean = old_rows.get(fid).mean if fid in old_rows else float("nan")
        new_mean = new_rows.get(fid).mean if fid in new_rows else float("nan")

        # Quantize to CSV precision for fair comparison
        old_q = quantize_sci(old_mean, tol=zero_tol)
        new_q = quantize_sci(new_mean, tol=zero_tol)
        diff_q = quantize_sci(new_q - old_q, tol=zero_tol)

        # Determine status and improvement string
        if old_q == 0.0:
            # Special case: reference is zero (solved)
            if new_q == 0.0:
                # Both solved → Same
                imp = "—"
                status = "●"
                same_count += 1
            else:
                # Was solved, now not → Worse (regression)
                imp = "N/A"
                status = "▼"
                worse_count += 1
        else:
            # Normal case: reference is non-zero
            imp_val = ((old_q - new_q) / old_q) * 100.0
            
            if imp_val > 0.01:
                # Significant improvement
                imp = f"+{imp_val:.1f}%"
                status = "▲"
                improved_count += 1
            elif imp_val < -0.01:
                # Significant regression
                imp = f"{imp_val:.1f}%"
                status = "▼"
                worse_count += 1
            else:
                # Within tolerance → Same
                imp = "0.0%"
                status = "●"
                same_count += 1

        # Add row to table
        lines.append(
            f"│{'F' + str(fid).zfill(2):^8}│"
            f"{format_sci(old_q, tol=zero_tol):^14}│"
            f"{format_sci(new_q, tol=zero_tol):^14}│"
            f"{format_sci(diff_q, tol=zero_tol):^14}│"
            f"{imp:^14}│"
            f"{status:^10}│"
        )

    # ========================================================================
    # Table Footer and Summary
    # ========================================================================
    
    lines.append("└" + "─" * 8 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 10 + "┘")
    
    total = improved_count + same_count + worse_count
    if total > 0:
        lines.append("")
        lines.append(f"  Summary: ▲ Improved: {improved_count}  ● Same: {same_count}  ▼ Worse: {worse_count}")
    
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# Exact Validation
# ============================================================================

def validate_summary_pair(
    ref_csv: Path,
    new_csv: Path,
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
    excluded_funcs: Sequence[int] = (2,),
    zero_tol: float = ZERO_TOL_DEFAULT,
    columns: Sequence[str] = ("Best", "Median", "Mean", "Worst", "SD"),
) -> Tuple[bool, str]:
    """
    Validate that new results match reference within tolerances.
    
    Used for regression testing to ensure implementation changes
    don't affect numerical results.
    
    Parameters
    ----------
    ref_csv : Path
        Reference summary CSV (ground truth).
        
    new_csv : Path
        New summary CSV to validate.
        
    abs_tol : float, default=0.0
        Absolute tolerance for comparison.
        Values within abs_tol are considered equal.
        
    rel_tol : float, default=0.0
        Relative tolerance for comparison.
        Values within rel_tol × |reference| are considered equal.
        
    excluded_funcs : sequence of int, default=(2,)
        Function IDs to exclude from validation.
        
    zero_tol : float, default=1e-7
        Tolerance for treating values as zero.
        
    columns : sequence of str
        Statistics columns to compare.
        Default: ("Best", "Median", "Mean", "Worst", "SD")
        
    Returns
    -------
    ok : bool
        True if all values match within tolerance.
        
    report : str
        "OK" if ok=True, otherwise detailed mismatch report.
        
    Notes
    -----
    Comparison uses effective tolerance:
        tol = max(abs_tol, rel_tol × |reference_value|)
        
    For exact match validation, use abs_tol=0, rel_tol=0.
    
    Example
    -------
    >>> ok, report = validate_summary_pair(
    ...     ref_csv=Path("reference/Summary_D10.csv"),
    ...     new_csv=Path("results/Summary_D10.csv"),
    ... )
    >>> if not ok:
    ...     print(report)
    """
    ref = load_summary_csv(ref_csv)
    new = load_summary_csv(new_csv)

    mismatches: List[str] = []

    for fid in _iter_func_ids(ref, new, excluded_funcs):
        # Check for missing functions
        if fid not in ref:
            mismatches.append(f"Missing in reference: F{fid:02d}")
            continue
        if fid not in new:
            mismatches.append(f"Missing in new results: F{fid:02d}")
            continue

        r = ref[fid]
        n = new[fid]

        # Map column names to values
        ref_map = {
            "Best": r.best,
            "Median": r.median,
            "Mean": r.mean,
            "Worst": r.worst,
            "SD": r.sd,
        }
        new_map = {
            "Best": n.best,
            "Median": n.median,
            "Mean": n.mean,
            "Worst": n.worst,
            "SD": n.sd,
        }

        # Compare each column
        for col in columns:
            rv = quantize_sci(ref_map[col], tol=zero_tol)
            nv = quantize_sci(new_map[col], tol=zero_tol)

            diff = abs(nv - rv)
            tol = max(abs_tol, rel_tol * abs(rv))
            
            if diff > tol:
                mismatches.append(
                    f"F{fid:02d} {col}: ref={format_sci(rv, tol=zero_tol)} "
                    f"new={format_sci(nv, tol=zero_tol)} "
                    f"diff={diff:.3E} tol={tol:.3E}"
                )

    # Generate report
    ok = len(mismatches) == 0
    if ok:
        return True, "OK"

    report = [
        f"Validation FAILED for {new_csv.name} vs {ref_csv.name}",
        f"abs_tol={abs_tol:g} rel_tol={rel_tol:g} zero_tol={zero_tol:g}",
        "Mismatches:",
        *mismatches,
    ]
    return False, "\n".join(report)
