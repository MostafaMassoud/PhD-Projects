# Baseline GSK (Gaining–Sharing Knowledge) — Publication-Grade Baseline (CEC2017)

## Quick start (step-by-step)

### Minimal commands (copy/paste)

From the project root (the folder containing ``scripts/`` and ``src/``):

```bash
pip install -r requirements.txt

python scripts/run_gsk.py --runs 51 --dims 10 30 50 100
python scripts/validate_gsk.py --abs-tol 0 --rel-tol 0
```

### What the commands do

1) **Install dependencies** (NumPy only)

```bash
pip install -r requirements.txt
```

2) **Run the full CEC2017 protocol** (29 functions: F1..F30 excluding F2, dimensions 10/30/50/100, 51 runs each)

```bash
python scripts/run_gsk.py --runs 51 --dims 10 30 50 100
```

3) **Validate correctness** against the authoritative reference CSVs in ``previous_results/gsk``

```bash
python scripts/validate_gsk.py --abs-tol 0 --rel-tol 0
```

Validation artifacts are written to the ``validation/`` folder.

---

This package contains a **clean, baseline implementation** of the
**Gaining–Sharing Knowledge (GSK)** metaheuristic optimizer in Python.

**What this package is**
- ✅ Pure baseline GSK (junior + senior knowledge sharing).
- ✅ Strictly **NFE-budget safe** via a centralized `BudgetController`.
- ✅ Deterministic/reproducible: explicit seed schedule and metadata logging.
- ✅ Integrated with **CEC2017** benchmarks.
- ✅ Includes a **validation utility** that compares produced summary tables against
  provided **reference baseline results**.

**What this package is NOT**
- ❌ No Reinforcement Learning (RL).
- ❌ No Model-Based (MB) layer.
- ❌ No hybrid extensions.

---

## Folder layout

```
.
├─ src/
│  └─ gsk_baseline/
│     ├─ gsk.py                 # Baseline GSK implementation (fully documented)
│     ├─ budget.py              # Centralized NFE budget enforcement
│     ├─ cec2017_adapter.py     # Robust import + evaluation wrapper for CEC2017
│     ├─ experiment.py          # Experiment loop (CEC2017 protocol)
│     ├─ validation.py          # Reference-vs-new comparison utilities
│     └─ utils.py               # Repro helpers, parsing, metadata
├─ scripts/
│  ├─ run_gsk.py                # CLI: run experiments, write Summary_All_Results_D*.csv
│  └─ validate_gsk.py           # CLI: run + validate vs reference CSVs
├─ previous_results/
│  └─ gsk/
│     ├─ Summary_All_Results_D10.csv
│     ├─ Summary_All_Results_D30.csv
│     ├─ Summary_All_Results_D50.csv
│     └─ Summary_All_Results_D100.csv
├─ results/                     # created/overwritten by experiments
├─ logs/                        # runtime logs + config dumps
└─ validation/                  # comparison artifacts
```

---

## Requirements

- Python >= 3.9
- NumPy

Install:

```bash
pip install -r requirements.txt
```

---

## CEC2017 dependency (external, **not included**)

This project **does not ship** the CEC2017 library. It must exist at:

```
../00-CEC2017
```

relative to this project directory.

Expected layouts are either:

- `../00-CEC2017/cec2017/functions.py`  (package directly), or
- `../00-CEC2017/cec2017/cec2017/functions.py` (wrapper folder + package folder).

The runner also supports a common **flat layout** where the contents of the upstream
``cec2017`` package are copied directly into ``../00-CEC2017``:

- `../00-CEC2017/functions.py` (with sibling modules like `simple.py`, `hybrid.py`, `composition.py`)

The runner will automatically add the correct folder to `sys.path`.

---

## Run experiments (CEC2017 protocol)

Run the full protocol (29 functions: F1..F30 excluding F2, dims 10/30/50/100, 51 runs):

```bash
python scripts/run_gsk.py --runs 51 --dims 10 30 50 100
```

Outputs:

- `results/Summary_All_Results_D10.csv`
- `results/Summary_All_Results_D30.csv`
- `results/Summary_All_Results_D50.csv`
- `results/Summary_All_Results_D100.csv`

CSV schema (exact):

`Function, Best, Median, Mean, Worst, SD`

All values are **errors**: `f_best - f_opt`, where `f_opt = 100 * func_id`.

---

## Smoke test mode (quick sanity check)

A quick run with fewer runs/functions/dimensions:

```bash
python scripts/run_gsk.py --smoke
```

---

## Validate against reference baseline

This will:
1) run baseline GSK with matching settings,
2) write new `results/Summary_All_Results_D*.csv`,
3) compare them to `previous_results/gsk/Summary_All_Results_D*.csv`,
4) write detailed artifacts into `validation/`, and
5) exit with non-zero code if mismatches exceed tolerances.

```bash
python scripts/validate_gsk.py --abs-tol 0 --rel-tol 0
```

For a fast validation check:

```bash
python scripts/validate_gsk.py --smoke --abs-tol 1e-8 --rel-tol 1e-8
```

---

## Reproducibility notes

- Deterministic seed schedule:

`seed = base_seed + func_id * stride_run + (run_id - 1)`

- The implementation uses `numpy.random.RandomState` and a MATLAB-compatible
  `rand()` helper (`rand_matlab`) using **Fortran-order reshape**.

- Each run prints environment metadata and writes a JSON config snapshot under `logs/`.

---

## License

This code is intended as a research-grade baseline implementation.

