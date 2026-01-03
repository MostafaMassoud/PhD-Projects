"""
Utility Functions for GSK Implementation
========================================

This module provides common utilities used throughout the GSK project:

1. **Numerical Formatting**: Scientific notation with zero-handling
2. **File Operations**: Directory creation, JSON writing
3. **Input Parsing**: Integer set parsing for CLI
4. **Reproducibility**: Seeding scheme, environment metadata
5. **Thread Control**: Single-threaded BLAS configuration

Numerical Precision Handling
----------------------------
The CEC benchmark comparison requires careful handling of small values:

Problem: When is a result "zero"?
    - Algorithm returns 4.2e-10
    - Is this zero? Almost zero? Non-zero?
    
Solution: Use reporting tolerance (REPORT_ZERO_TOL = 1e-7)
    - Values |x| ≤ 1e-7 are treated as zero for display
    - Stored values remain unchanged for reproducibility
    - Only affects human-readable output

Functions:
    - zero_small(x): Returns 0.0 if |x| ≤ tol
    - format_sci(x): "0.00E+00" for small x, else "X.XXE±YY"
    - quantize_sci(x): Rounds to 2 decimal scientific notation precision

Reproducibility Infrastructure
------------------------------
For scientific reproducibility, we need:

1. **Deterministic seeding**: Each (function, run) pair gets unique seed
   seed = BASE_SEED + func_id × STRIDE_RUN + (run_id - 1)
   
2. **Environment logging**: Capture Python version, NumPy version,
   BLAS backend, thread settings for debugging cross-platform differences

3. **Single-threaded mode**: Optional BLAS thread limiting for
   deterministic floating-point operations
"""

from __future__ import annotations

import io
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .constants import REPORT_ZERO_TOL


# ============================================================================
# Constants
# ============================================================================

ZERO_TOL_DEFAULT: float = float(REPORT_ZERO_TOL)
"""Default tolerance for treating values as zero (1e-7)."""


# ============================================================================
# Time Utilities
# ============================================================================

def timestamp_now() -> str:
    """
    Generate filesystem-friendly timestamp.
    
    Returns
    -------
    str
        Timestamp in format "YYYY-MM-DD_HH.MM.SS"
        
    Example
    -------
    >>> timestamp_now()
    '2024-01-15_14.30.45'
    """
    return datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


# ============================================================================
# File System Utilities
# ============================================================================

def ensure_dir(path: Path) -> Path:
    """
    Create directory if it doesn't exist.
    
    Creates parent directories as needed (like mkdir -p).
    
    Parameters
    ----------
    path : Path
        Directory path to create.
        
    Returns
    -------
    Path
        The same path (for chaining).
        
    Example
    -------
    >>> ensure_dir(Path("results/2024/run1"))
    PosixPath('results/2024/run1')
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """
    Write data to JSON file with stable ordering.
    
    Features:
    - Creates parent directories automatically
    - Sorts keys for reproducible output
    - Handles Path objects in data
    
    Parameters
    ----------
    path : Path
        Output file path.
    data : Any
        Data to serialize (dict, list, etc.).
    indent : int, default=2
        JSON indentation level.
    """
    ensure_dir(path.parent)

    def _default(o: Any) -> Any:
        """Handle non-serializable types."""
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(
        json.dumps(data, indent=indent, sort_keys=True, default=_default),
        encoding="utf-8",
    )


# ============================================================================
# Input Parsing
# ============================================================================

def parse_int_set(spec: str) -> Set[int]:
    """
    Parse integer set specification from string.
    
    Supports:
    - Single values: "5"
    - Comma-separated: "1,3,7"
    - Ranges: "1-5" (inclusive)
    - Mixed: "1-5,7,10-12"
    
    Parameters
    ----------
    spec : str
        Integer set specification string.
        
    Returns
    -------
    set of int
        Parsed integers.
        
    Raises
    ------
    ValueError
        If spec is empty.
        
    Examples
    --------
    >>> parse_int_set("1,3,5")
    {1, 3, 5}
    >>> parse_int_set("1-5")
    {1, 2, 3, 4, 5}
    >>> parse_int_set("1-3,7,10-12")
    {1, 2, 3, 7, 10, 11, 12}
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("Empty integer-set specification")

    out: Set[int] = set()
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


# ============================================================================
# Numerical Formatting with Zero Handling
# ============================================================================

def zero_small(x: float, *, tol: float = ZERO_TOL_DEFAULT) -> float:
    """
    Return 0.0 if |x| ≤ tolerance, else return x.
    
    This implements the "report zero" convention where very small
    values are treated as exactly zero for display purposes.
    
    Parameters
    ----------
    x : float
        Value to check.
    tol : float, default=1e-7
        Tolerance threshold.
        
    Returns
    -------
    float
        0.0 if |x| ≤ tol, else float(x).
        
    Examples
    --------
    >>> zero_small(1e-8)
    0.0
    >>> zero_small(1e-6)
    1e-06
    >>> zero_small(0.5)
    0.5
    """
    return 0.0 if abs(float(x)) <= float(tol) else float(x)


def format_sci(x: float, *, tol: float = ZERO_TOL_DEFAULT) -> str:
    """
    Format number in 2-decimal scientific notation with zero handling.
    
    Small values (|x| ≤ tol) are displayed as "0.00E+00".
    Other values use standard scientific notation.
    
    Parameters
    ----------
    x : float
        Value to format.
    tol : float, default=1e-7
        Tolerance for zero treatment.
        
    Returns
    -------
    str
        Formatted string like "3.06E+01" or "0.00E+00".
        
    Examples
    --------
    >>> format_sci(30.6)
    '3.06E+01'
    >>> format_sci(1e-9)
    '0.00E+00'
    >>> format_sci(0.00123)
    '1.23E-03'
    """
    xz = zero_small(x, tol=tol)
    return f"{xz:.2E}"


def quantize_sci(x: float, *, tol: float = ZERO_TOL_DEFAULT) -> float:
    """
    Quantize float to 2-decimal scientific notation precision.
    
    This matches the precision of reference CSV files, enabling
    exact comparison between implementations.
    
    Process:
    1. Apply zero tolerance
    2. Format as scientific notation
    3. Parse back to float
    
    Parameters
    ----------
    x : float
        Value to quantize.
    tol : float, default=1e-7
        Tolerance for zero treatment.
        
    Returns
    -------
    float
        Quantized value.
        
    Examples
    --------
    >>> quantize_sci(30.567)
    30.6
    >>> quantize_sci(1.234567e-5)
    1.23e-05
    """
    xz = zero_small(x, tol=tol)
    return float(f"{xz:.2E}")


# ============================================================================
# Reproducibility: Seeding
# ============================================================================

def seed_for_run(
    *,
    base_seed: int,
    stride_run: int,
    func_id: int,
    run_id: int,
) -> int:
    """
    Generate deterministic seed for a specific (function, run) pair.
    
    Seeding scheme ensures:
    - Same seed for same (func, run) across executions
    - Different seeds for different (func, run) pairs
    - No seed collision within reasonable parameter ranges
    
    Formula:
        seed = base_seed + func_id × stride_run + (run_id - 1)
    
    Parameters
    ----------
    base_seed : int
        Starting seed (e.g., 123456).
    stride_run : int
        Offset between functions (e.g., 9973).
        Should be prime and > max runs.
    func_id : int
        Function ID (1-30 for CEC2017).
    run_id : int
        Run number (1-51 typically).
        
    Returns
    -------
    int
        Computed seed for this (func, run) pair.
        
    Examples
    --------
    >>> seed_for_run(base_seed=123456, stride_run=9973, func_id=1, run_id=1)
    133429
    >>> seed_for_run(base_seed=123456, stride_run=9973, func_id=1, run_id=2)
    133430
    >>> seed_for_run(base_seed=123456, stride_run=9973, func_id=2, run_id=1)
    143402
    """
    return int(base_seed) + int(func_id) * int(stride_run) + (int(run_id) - 1)


# ============================================================================
# Environment Metadata for Reproducibility
# ============================================================================

@dataclass(frozen=True)
class EnvironmentMetadata:
    """
    Captured environment information for reproducibility logging.
    
    Records system configuration that may affect numerical results:
    - Python and NumPy versions
    - Platform and architecture
    - BLAS backend (OpenBLAS, MKL, etc.)
    - Thread environment variables
    
    Attributes
    ----------
    python_version : str
        Full Python version string.
    python_executable : str
        Path to Python interpreter.
    platform : str
        OS and kernel version.
    machine : str
        Hardware architecture (x86_64, arm64, etc.).
    processor : str
        Processor description.
    cpu_count : int
        Number of CPUs available.
    numpy_version : str
        NumPy version string.
    thread_env : dict
        Thread-related environment variables.
    numpy_blas_backend : str
        Detected BLAS library (OpenBLAS, MKL, etc.).
    numpy_config_text : str or None
        Full numpy.show_config() output if requested.
    """
    python_version: str
    python_executable: str
    platform: str
    machine: str
    processor: str
    cpu_count: int
    numpy_version: str
    thread_env: Dict[str, str]
    numpy_blas_backend: str
    numpy_config_text: Optional[str]


# Thread environment variables that affect BLAS parallelism
_THREAD_ENV_KEYS: List[str] = [
    "OMP_NUM_THREADS",       # OpenMP threads
    "MKL_NUM_THREADS",       # Intel MKL threads
    "OPENBLAS_NUM_THREADS",  # OpenBLAS threads
    "VECLIB_MAXIMUM_THREADS", # macOS Accelerate threads
    "NUMEXPR_NUM_THREADS",   # NumExpr threads
]


def _capture_numpy_show_config() -> str:
    """Capture numpy.show_config() output as string."""
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        try:
            np.show_config()
        except Exception:
            try:
                np.__config__.show()
            except Exception:
                pass
    finally:
        sys.stdout = old
    return buf.getvalue().strip()


def _infer_numpy_blas_backend() -> str:
    """Best-effort detection of BLAS backend from NumPy config."""
    try:
        info = np.__config__.get_info("blas_opt_info")
    except Exception:
        info = {}

    libs = info.get("libraries") if isinstance(info, dict) else None
    if libs:
        return " ".join(str(x) for x in libs)

    txt = _capture_numpy_show_config()
    first = txt.splitlines()[0].strip() if txt else "(unknown)"
    return first or "(unknown)"


def collect_environment_metadata(
    *,
    include_numpy_config_full: bool = False,
) -> EnvironmentMetadata:
    """
    Collect current environment metadata.
    
    Parameters
    ----------
    include_numpy_config_full : bool, default=False
        Include full numpy.show_config() output.
        
    Returns
    -------
    EnvironmentMetadata
        Populated metadata object.
    """
    thread_env: Dict[str, str] = {}
    for k in _THREAD_ENV_KEYS:
        v = os.environ.get(k)
        if v is not None:
            thread_env[k] = v

    numpy_cfg = _capture_numpy_show_config() if include_numpy_config_full else None

    return EnvironmentMetadata(
        python_version=sys.version.replace("\n", " "),
        python_executable=sys.executable,
        platform=f"{platform.system()}-{platform.release()}-{platform.version()}",
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count() or 0,
        numpy_version=np.__version__,
        thread_env=thread_env,
        numpy_blas_backend=_infer_numpy_blas_backend(),
        numpy_config_text=numpy_cfg,
    )


def format_environment_metadata(
    meta: EnvironmentMetadata,
    *,
    header: bool = True,
) -> str:
    """
    Format environment metadata as human-readable text.
    
    Parameters
    ----------
    meta : EnvironmentMetadata
        Metadata to format.
    header : bool, default=True
        Include section header.
        
    Returns
    -------
    str
        Formatted text block.
    """
    lines: List[str] = []
    if header:
        lines.append("#" * 72)
        lines.append("# ENVIRONMENT / DETERMINISM")
        lines.append("#" * 72)

    lines.append(f"python_version   : {meta.python_version}")
    lines.append(f"python_executable: {meta.python_executable}")
    lines.append(f"platform         : {meta.platform}")
    lines.append(f"machine          : {meta.machine}")
    lines.append(f"processor        : {meta.processor}")
    lines.append(f"cpu_count        : {meta.cpu_count}")
    lines.append(f"numpy_version    : {meta.numpy_version}")

    lines.append("thread_env:")
    if meta.thread_env:
        for k in _THREAD_ENV_KEYS:
            if k in meta.thread_env:
                lines.append(f"  {k}={meta.thread_env[k]}")
    else:
        lines.append("  (not set)")

    lines.append(f"numpy_blas_backend: {meta.numpy_blas_backend}")

    if meta.numpy_config_text is not None:
        lines.append("numpy_config:")
        lines.append(meta.numpy_config_text)
    else:
        lines.append(
            "numpy_config: (omitted; enable with --log-numpy-config-full=1)"
        )

    return "\n".join(lines) + "\n"


def apply_thread_env(*, force_single_thread: bool, override: bool) -> None:
    """
    Set environment variables for single-threaded BLAS.
    
    Single-threaded mode can improve determinism by avoiding
    thread-dependent floating-point summation order.
    
    Parameters
    ----------
    force_single_thread : bool
        If True, set all thread variables to "1".
    override : bool
        If True, override existing values.
        If False, only set if not already defined.
    """
    if not force_single_thread:
        return

    for k in _THREAD_ENV_KEYS:
        if override or (k not in os.environ):
            os.environ[k] = "1"


def env_metadata_as_dict(meta: EnvironmentMetadata) -> Dict[str, Any]:
    """
    Convert EnvironmentMetadata to JSON-safe dictionary.
    
    Parameters
    ----------
    meta : EnvironmentMetadata
        Metadata to convert.
        
    Returns
    -------
    dict
        Dictionary representation.
    """
    return asdict(meta)
