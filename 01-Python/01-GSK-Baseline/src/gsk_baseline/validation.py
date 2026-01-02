from __future__ import annotations

"""gsk_baseline.validation

Validation against reference baseline results
============================================

Correctness for a baseline metaheuristic implementation is best demonstrated by
**reproducing known reference results** under an identical experimental
protocol.

This module implements a comparison utility for the CSV summaries produced by
:func:`gsk_baseline.experiment.run_cec2017_experiments`.

Reference results
-----------------
The expected reference folder (by requirement) is:

```
<Project Root>/previous_results/gsk
```

with files:

- ``Summary_All_Results_D10.csv``
- ``Summary_All_Results_D30.csv``
- ``Summary_All_Results_D50.csv``
- ``Summary_All_Results_D100.csv``

CSV schema (must match exactly):

```
Function, Best, Median, Mean, Worst, SD
```

The validator:
- loads reference summaries,
- loads newly generated summaries,
- computes absolute and relative differences per metric,
- writes detailed comparison artifacts, and
- returns a pass/fail outcome (or can be used by a CLI to exit non-zero).
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


METRICS: Tuple[str, ...] = ("Best", "Median", "Mean", "Worst", "SD")


@dataclass(frozen=True)
class ValidationResult:
    """Summary of a validation run."""

    ok: bool
    n_mismatched: int
    artifacts_dir: Path


def _load_summary_csv(path: Path) -> Dict[int, Dict[str, float]]:
    """Load a summary CSV.

    Returns
    -------
    dict
        Mapping: function_id -> metric mapping.
    """

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]

        # Basic schema check (order is not critical for parsing, but we validate columns exist)
        required = {"Function", *METRICS}
        missing = required - set(header)
        if missing:
            raise ValueError(f"CSV {path} is missing columns: {sorted(missing)}")

        out: Dict[int, Dict[str, float]] = {}
        for row in reader:
            if not row:
                continue
            fid = int(float(row["Function"]))
            out[fid] = {m: float(row[m]) for m in METRICS}
        return out


def _safe_rel_diff(ref: float, new: float, eps: float = 1e-12) -> float:
    denom = max(abs(ref), eps)
    return abs(new - ref) / denom


def _write_comparison_csv(
    path: Path,
    merged_rows: List[Dict[str, object]],
) -> None:
    """Write a detailed comparison CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)

    # Stable column order
    fields: List[str] = ["Function"]
    for m in METRICS:
        fields.extend([
            f"Ref_{m}",
            f"New_{m}",
            f"AbsDiff_{m}",
            f"RelDiff_{m}",
        ])
    fields.append("Mismatch")

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in merged_rows:
            w.writerow(r)


def compare_dimension(
    *,
    ref_csv: Path,
    new_csv: Path,
    abs_tol: float,
    rel_tol: float,
) -> tuple[bool, int, List[Dict[str, object]]]:
    """Compare a single (dimension) summary file.

    A metric is flagged as a mismatch if its absolute difference exceeds
    ``abs_tol`` **and** its relative difference exceeds ``rel_tol``.

    This dual criterion is robust to near-zero reference values (where relative
    errors explode) and to large-scale values (where absolute tolerance alone is
    too strict).

    Returns
    -------
    ok, n_mismatch, merged_rows
    """

    ref = _load_summary_csv(ref_csv)
    new = _load_summary_csv(new_csv)

    fids = sorted(set(ref.keys()) | set(new.keys()))

    merged_rows: List[Dict[str, object]] = []
    mismatched = 0

    for fid in fids:
        row: Dict[str, object] = {"Function": fid}
        any_mismatch = False

        if fid not in ref or fid not in new:
            any_mismatch = True
            for m in METRICS:
                row[f"Ref_{m}"] = ref.get(fid, {}).get(m, float("nan"))
                row[f"New_{m}"] = new.get(fid, {}).get(m, float("nan"))
                row[f"AbsDiff_{m}"] = float("nan")
                row[f"RelDiff_{m}"] = float("nan")
        else:
            for m in METRICS:
                r = float(ref[fid][m])
                n = float(new[fid][m])
                ad = abs(n - r)
                rd = _safe_rel_diff(r, n)

                row[f"Ref_{m}"] = r
                row[f"New_{m}"] = n
                row[f"AbsDiff_{m}"] = ad
                row[f"RelDiff_{m}"] = rd

                if (ad > abs_tol) and (rd > rel_tol):
                    any_mismatch = True

        row["Mismatch"] = bool(any_mismatch)
        if any_mismatch:
            mismatched += 1

        merged_rows.append(row)

    ok = mismatched == 0
    return ok, mismatched, merged_rows


def validate_against_reference(
    *,
    project_root: Path,
    results_dir: Path,
    reference_dir: Optional[Path] = None,
    dims: Sequence[int] = (10, 30, 50, 100),
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
    artifacts_dir: Optional[Path] = None,
    verbose: bool = True,
) -> ValidationResult:
    """Validate generated results against reference CSV summaries.

    Parameters
    ----------
    project_root:
        Root of this project.
    results_dir:
        Directory that contains newly generated ``Summary_All_Results_D*.csv``.
    reference_dir:
        Directory containing reference summaries. If None, defaults to
        ``<project_root>/previous_results/gsk``.
    dims:
        Dimensions to validate.
    abs_tol, rel_tol:
        Tolerances for mismatches.
    artifacts_dir:
        Directory to write comparison outputs. If None, defaults to
        ``<project_root>/validation``.
    verbose:
        If True, print a short validation summary.

    Returns
    -------
    ValidationResult
        Contains pass/fail and artifact location.
    """

    project_root = project_root.resolve()
    results_dir = results_dir.resolve()

    if reference_dir is None:
        reference_dir = project_root / "previous_results" / "gsk"
    reference_dir = reference_dir.resolve()

    if artifacts_dir is None:
        artifacts_dir = project_root / "validation"
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    total_mismatch = 0

    for D in dims:
        D = int(D)
        ref_csv = reference_dir / f"Summary_All_Results_D{D}.csv"
        new_csv = results_dir / f"Summary_All_Results_D{D}.csv"

        if not ref_csv.exists():
            raise FileNotFoundError(f"Reference CSV not found: {ref_csv}")
        if not new_csv.exists():
            raise FileNotFoundError(f"New results CSV not found: {new_csv}")

        ok, mismatched, rows = compare_dimension(
            ref_csv=ref_csv,
            new_csv=new_csv,
            abs_tol=float(abs_tol),
            rel_tol=float(rel_tol),
        )

        total_mismatch += int(mismatched)
        all_ok = all_ok and ok

        out_csv = artifacts_dir / f"comparison_D{D}.csv"
        _write_comparison_csv(out_csv, rows)

        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"D={D}: {status} (mismatched functions: {mismatched}) -> {out_csv}")

    return ValidationResult(ok=all_ok, n_mismatched=total_mismatch, artifacts_dir=artifacts_dir)
