from __future__ import annotations

"""gsk_baseline.utils

This module contains **experiment-level** utilities used by the command line
scripts.

The baseline GSK algorithm is fully contained in :mod:`gsk_baseline.gsk`.
Nothing in this module performs objective evaluations.

Reproducibility policy
----------------------
For deterministic experimentation we use an explicit, documented seed schedule.
For a given (function_id, run_id) pair we derive the RNG seed as:

    seed = base_seed + function_id * stride_run + (run_id - 1)

This schedule is consistent with many CEC-style experimental protocols: runs are
independent, yet reproducible and easily enumerable.

The package prints and logs environment metadata (Python/Numpy versions and the
platform) because small differences in numerical libraries can affect floating
point results.
"""

from dataclasses import asdict, dataclass
import json
import os
import platform
from pathlib import Path
import re
import sys
from typing import Iterable, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EnvironmentMetadata:
    """Minimal environment metadata for reproducibility logs."""

    python_version: str
    numpy_version: str
    platform: str
    platform_release: str
    platform_version: str
    machine: str
    processor: str


def collect_environment_metadata() -> EnvironmentMetadata:
    """Collect metadata about the current runtime environment.

    Returns
    -------
    EnvironmentMetadata
        A compact structure that can be printed or serialized.
    """

    return EnvironmentMetadata(
        python_version=sys.version.replace("\n", " "),
        numpy_version=np.__version__,
        platform=platform.system(),
        platform_release=platform.release(),
        platform_version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
    )


def format_environment_metadata(meta: EnvironmentMetadata) -> str:
    """Format environment metadata as a human-readable block."""

    lines = [
        "Environment metadata:",
        f"  Python   : {meta.python_version}",
        f"  NumPy    : {meta.numpy_version}",
        f"  OS       : {meta.platform} {meta.platform_release}",
        f"  OS ver.  : {meta.platform_version}",
        f"  Machine  : {meta.machine}",
        f"  Processor: {meta.processor}",
    ]
    return "\n".join(lines)


def seed_for_run(*, base_seed: int, stride_run: int, func_id: int, run_id: int) -> int:
    """Deterministic seed schedule.

    Parameters
    ----------
    base_seed:
        Global base seed.
    stride_run:
        Multiplier used to separate functions in the seed space.
    func_id:
        CEC function id (1..30).
    run_id:
        Run index (1..N).

    Returns
    -------
    int
        Seed to initialize :class:`numpy.random.RandomState`.
    """

    if run_id < 1:
        raise ValueError("run_id must be 1-based (>= 1)")
    return int(base_seed + int(func_id) * int(stride_run) + (int(run_id) - 1))


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Mapping) -> None:
    """Write a JSON file with stable formatting."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


_RANGE_RE = re.compile(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$")


def parse_int_set(spec: str) -> List[int]:
    """Parse a compact integer list specification.

    Supported forms
    ---------------
    - "1,3,4"              -> [1,3,4]
    - "1-5" or "1:5"       -> [1,2,3,4,5]
    - "10"                 -> [10]

    Parameters
    ----------
    spec:
        Input string.

    Returns
    -------
    list[int]
        Sorted unique integers.
    """

    spec = spec.strip()
    if not spec:
        return []

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    out: List[int] = []
    for p in parts:
        m = _RANGE_RE.match(p)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            if b < a:
                a, b = b, a
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(p))
    # unique + sorted
    return sorted(set(out))


def set_single_thread_env_if_requested(force_single_thread: bool = True) -> None:
    """Optionally force NumPy/BLAS to a single thread for determinism.

    Notes
    -----
    This does not change algorithmic behavior, but can reduce run-to-run noise
    in wall-clock timings and avoids accidental non-determinism in some BLAS
    configurations.

    We set these variables only if they are not already present, to avoid
    surprising users who intentionally configured their environment.
    """

    if not force_single_thread:
        return

    env_defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    for k, v in env_defaults.items():
        os.environ.setdefault(k, v)


def as_pretty_dict(obj: object) -> Mapping:
    """Best-effort conversion of dataclass-like configs to a JSON-serializable dict."""

    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)  # type: ignore[arg-type]
    elif isinstance(obj, dict):
        d = obj
    else:
        # fallback
        d = {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}

    # Convert Path objects
    clean = {}
    for k, v in d.items():
        if isinstance(v, Path):
            clean[k] = str(v)
        else:
            clean[k] = v
    return clean
