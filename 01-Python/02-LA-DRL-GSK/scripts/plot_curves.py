#!/usr/bin/env python3
"""Generate convergence graphs from exported median-run curves.

Usage:
    python scripts/plot_curves.py
    python scripts/plot_curves.py --dims 10 30 50 100

Generates one graph per function per dimension.
Expected: 29 functions × 4 dimensions = 116 graphs (excluding F02).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_curve(path: Path) -> Tuple[List[int], List[float], List[float]]:
    """Load a curve CSV file."""
    evals: List[int] = []
    errors: List[float] = []
    log10_errors: List[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evals.append(int(row["Eval"]))
            errors.append(float(row["BestError"]))
            log10_errors.append(float(row["Log10Error"]))
    return evals, errors, log10_errors


def parse_curve_filename(path: Path) -> Tuple[int, int, int]:
    """Extract func_id, dimension, and run from filename."""
    # Pattern: Figure_F{func_id}_D{dim}_Run#{run}.csv
    match = re.match(r"Figure_F(\d+)_D(\d+)_Run#(\d+)", path.stem)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return 0, 0, 0


def main() -> None:
    if not HAS_MATPLOTLIB:
        print("matplotlib is required for plotting. Install with:")
        print("  pip install matplotlib")
        return

    ap = argparse.ArgumentParser(
        description="Plot convergence curves (one graph per function per dimension)."
    )
    ap.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[10, 30, 50, 100],
        help="Dimensions to plot.",
    )
    ap.add_argument(
        "--combined",
        action="store_true",
        help="Also generate combined plots (all functions per dimension).",
    )
    args = ap.parse_args()

    curves_dir = PROJECT_ROOT / "results" / "gsk" / "curves"
    graphs_dir = curves_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    total_graphs = 0

    print("=" * 60)
    print("GSK Convergence Curve Plotter")
    print("=" * 60)

    for D in args.dims:
        pattern = f"Figure_F*_D{D}_Run#*.csv"
        curve_files = sorted(curves_dir.glob(pattern))

        if not curve_files:
            print(f"\n[SKIP] D={D}: No curve files found.")
            continue

        print(f"\n[D={D}] Found {len(curve_files)} curve files")
        print("-" * 40)

        # Generate individual graphs for each function
        for curve_path in curve_files:
            func_id, dim, run = parse_curve_filename(curve_path)
            if func_id == 0:
                continue

            evals, errors, log10_err = load_curve(curve_path)

            # Create single-panel figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot log10(Error)
            ax.plot(evals, log10_err, color="#1f77b4", linewidth=1.2)
            
            ax.set_xlabel("Function Evaluations", fontsize=11)
            ax.set_ylabel("log10(Error)", fontsize=11)
            ax.set_title(
                f"GSK Convergence — F{func_id} (D{D}), Run {run}",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.ticklabel_format(style="plain", axis="x")

            plt.tight_layout()

            # Save
            out_path = graphs_dir / f"F{func_id:02d}_D{D}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"  ✓ F{func_id:02d}_D{D}.png")
            total_graphs += 1

        # Optionally generate combined plot
        if args.combined:
            fig, ax = plt.subplots(figsize=(12, 7))

            colors = plt.cm.tab20(range(20))
            color_idx = 0

            for curve_path in curve_files:
                func_id, dim, run = parse_curve_filename(curve_path)
                if func_id == 0:
                    continue

                evals, _, log10_err = load_curve(curve_path)
                ax.plot(
                    evals,
                    log10_err,
                    label=f"F{func_id:02d}",
                    linewidth=0.8,
                    color=colors[color_idx % 20],
                )
                color_idx += 1

            ax.set_xlabel("Function Evaluations", fontsize=11)
            ax.set_ylabel("log10(Error)", fontsize=11)
            ax.set_title(f"GSK Convergence — All Functions (D{D})", fontsize=12, fontweight="bold")
            ax.legend(ncol=6, fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

            out_path = graphs_dir / f"combined_D{D}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            print(f"  ✓ combined_D{D}.png")
            total_graphs += 1

    print("\n" + "=" * 60)
    print(f"✓ Generated {total_graphs} graphs in: {graphs_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
