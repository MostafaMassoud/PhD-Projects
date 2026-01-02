# GSK-Baseline

**Gaining-Sharing Knowledge (GSK) Algorithm** — Baseline Implementation for CEC2017 Benchmark

## Algorithm Overview

GSK is a population-based metaheuristic for continuous optimization, inspired by the human process of gaining and sharing knowledge. The algorithm was introduced in:

> Mohamed, A. W., Hadi, A. A., & Mohamed, A. K. (2020). "Gaining-Sharing Knowledge Based Algorithm for Solving Optimization Problems: A Novel Nature-Inspired Algorithm." *International Journal of Machine Learning and Cybernetics*, 11, 1501-1529.

### Key Concepts

GSK maintains a population of candidate solutions that evolve through two complementary phases:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GSK ALGORITHM FLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INITIALIZATION                                                         │
│  └── Generate random population in search space                         │
│                                                                         │
│  FOR each generation until budget exhausted:                            │
│  │                                                                      │
│  ├── Compute D_junior = ceil(D × (1 - g/G_max)^K)                       │
│  │   └── Controls exploration/exploitation balance                      │
│  │                                                                      │
│  ├── JUNIOR PHASE (Exploration)                                         │
│  │   └── Learn from rank-based neighbors                                │
│  │                                                                      │
│  ├── SENIOR PHASE (Exploitation)                                        │
│  │   └── Learn from elite/middle/worst groups                           │
│  │                                                                      │
│  ├── CROSSOVER (Dimension masking with KR probability)                  │
│  │                                                                      │
│  └── SELECTION (Greedy: keep better solution)                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase Details

**1. Junior Gaining-Sharing (Exploration)**

Models how junior individuals learn from immediate peers in fitness ranking:

```
For each individual i:
  R1 = better rank neighbor
  R2 = worse rank neighbor  
  R3 = random individual

  If fitness[i] > fitness[R3]:  (i is worse, learns from R3)
      x_junior = x_i + KF × (x_R1 - x_R2 + x_R3 - x_i)
  Else:  (i is better, shares with R3)
      x_junior = x_i + KF × (x_R1 - x_R2 + x_i - x_R3)
```

**2. Senior Gaining-Sharing (Exploitation)**

Models how experienced individuals learn from stratified groups:

```
Population divided into:
  - Top 10% (Elite)
  - Middle 80%
  - Bottom 10% (Worst)

For each individual i:
  R1 = random from Elite
  R2 = random from Middle
  R3 = random from Worst

  If fitness[i] > fitness[R2]:  (i worse than middle)
      x_senior = x_i + KF × (x_R1 - x_i + x_R2 - x_R3)
  Else:
      x_senior = x_i + KF × (x_R1 - x_R2 + x_i - x_R3)
```

**3. Dynamic Balance**

The ratio of junior to senior contributions shifts over generations:

```
D_junior = ceil(D × (1 - g/G_max)^K)

Early generations: D_junior ≈ D (all exploration)
Late generations:  D_junior ≈ 0 (all exploitation)
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Smoke Test
```bash
python scripts/run_smoke.py
```

### Full CEC2017 Experiment
```bash
python scripts/run_gsk.py --runs 51 --dims 10 30 50 100
```

### Validate Against Reference
```bash
python scripts/validate_gsk.py --abs-tol 0 --rel-tol 0
```

## Project Structure

```
GSK-Baseline/
├── src/gsk/
│   ├── gsk.py                    # Core GSK algorithm
│   ├── gained_shared_junior.py   # Junior phase index selection
│   ├── gained_shared_senior.py   # Senior phase index selection
│   ├── bound_constraint.py       # L-SHADE boundary repair
│   ├── rand_matlab.py            # MATLAB-compatible RNG
│   ├── budget.py                 # Evaluation budget controller
│   ├── config.py                 # Experiment configuration
│   ├── constants.py              # Default parameters
│   ├── experiment.py             # Main experiment harness
│   ├── cec2017_adapter.py        # CEC2017 benchmark adapter
│   ├── utils.py                  # Utilities
│   └── validation.py             # Result validation
├── scripts/
│   ├── run_gsk.py                # Main CLI entrypoint
│   ├── run_smoke.py              # Quick smoke test
│   ├── validate_gsk.py           # Validate results
│   └── plot_curves.py            # Convergence plots
├── previous_results/gsk/         # Reference baseline results
└── results/                      # Output directory
```

## MATLAB Correspondence

This implementation mirrors the reference MATLAB code:

| MATLAB File                      | Python Module              | Function                          |
|----------------------------------|----------------------------|-----------------------------------|
| `GSK.m`                          | `gsk.py`                   | `gsk_optimize()`                  |
| `Gained_Shared_Junior_R1R2R3.m`  | `gained_shared_junior.py`  | `gained_shared_junior_r1r2r3()`   |
| `Gained_Shared_Senior_R1R2R3.m`  | `gained_shared_senior.py`  | `gained_shared_senior_r1r2r3()`   |
| `boundConstraint.m`              | `bound_constraint.py`      | `bound_constraint()`              |

## Algorithm Parameters

| Parameter   | Default | Description                                        |
|-------------|---------|---------------------------------------------------|
| `KF`        | 0.5     | Knowledge Factor — mutation step size              |
| `KR`        | 0.9     | Knowledge Ratio — crossover probability            |
| `Kexp`      | 10.0    | Knowledge Rate — exploration/exploitation balance  |
| `pop_size`  | 100     | Population size                                    |
| `max_nfes`  | 10000×D | Evaluation budget (CEC2017 standard)               |

## CEC2017 Benchmark

Requires external CEC2017 Python implementation. Place at:
- `../00-CEC-Root/cec2017/functions.py`
- `../00-CEC2017/cec2017/functions.py`

Or specify: `python scripts/run_gsk.py --cec-root /path/to/cec2017`

### Benchmark Properties

| Category          | Functions | Characteristics          |
|-------------------|-----------|--------------------------|
| Unimodal          | F1-F3     | Single global optimum    |
| Simple Multimodal | F4-F10    | Multiple local optima    |
| Hybrid            | F11-F20   | Mixed problem types      |
| Composition       | F21-F30   | Complex landscapes       |

**Note**: F2 is excluded due to numerical instability (standard practice).

## Numerical Conventions

Two-threshold system for handling small values:

1. **VAL_TO_REACH (10⁻⁸)**: Per-run success threshold
   - Errors < 10⁻⁸ are set to 0.0
   - Determines if a problem is "solved"

2. **REPORT_ZERO_TOL (10⁻⁷)**: Display tolerance
   - Values ≤ 10⁻⁷ shown as "0.00E+00"
   - Prevents misleading tiny values in reports

## Reproducibility

- **Deterministic seeding**: `seed = base_seed + func_id × stride + (run - 1)`
- **Single-threaded BLAS**: Optional (default enabled)
- **Stable sorting**: Uses mergesort for consistent rankings
- **MATLAB-compatible RNG**: Column-major random number generation

## Output Files

```
results/gsk/
├── summary/
│   ├── Summary_All_Results_D10.csv   # Statistics per function
│   ├── environment.json               # Environment metadata
│   └── run_config.json                # Configuration used
├── gen_logs/
│   └── GenLog_gsk_F1_D10_Run1.csv    # Per-generation diagnostics
└── curves/
    └── Figure_F1_D10_Run#26.csv      # Convergence curves
```

## License

MIT
