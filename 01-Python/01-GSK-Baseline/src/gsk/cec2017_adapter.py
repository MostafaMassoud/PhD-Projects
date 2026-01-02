"""
CEC2017 Benchmark Function Adapter
==================================

This module provides an adapter for external CEC2017 benchmark implementations,
enabling the GSK algorithm to work with standard competition test functions.

CEC2017 Benchmark Overview
--------------------------
The CEC2017 benchmark suite consists of 30 scalable test functions designed
for evaluating real-parameter single-objective optimization algorithms:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Category          │ Functions │ Characteristics               │
    ├──────────────────────────────────────────────────────────────────┤
    │  Unimodal          │  F1-F3    │ Single global optimum         │
    │  Simple Multimodal │  F4-F10   │ Multiple local optima         │
    │  Hybrid            │  F11-F20  │ Mixed problem types           │
    │  Composition       │  F21-F30  │ Complex landscapes            │
    └──────────────────────────────────────────────────────────────────┘

Key Properties:
- All functions are minimization problems
- Search space: [-100, +100]^D for all dimensions
- Optimal value: f_opt = 100 × function_id (F1→100, F2→200, ..., F30→3000)
- F2 is excluded from experiments due to numerical instability

External Implementation
-----------------------
This adapter connects to an external Python CEC2017 implementation (not bundled).
The external code must provide:

    cec2017/functions.py:
        all_functions: list  # 30 callable functions
        
    Each function: f(X) -> y
        X: np.ndarray of shape (n, D) - n candidates, D dimensions
        y: np.ndarray of shape (n,) - fitness values

Directory Layout Support
------------------------
Three common layouts are automatically detected:

    Layout A: <cec_root>/cec2017/functions.py
    Layout B: <cec_root>/cec2017/cec2017/functions.py  
    Layout C: <cec_root>/functions.py

The adapter tries each layout and uses the first that works.

Usage
-----
    # Ensure CEC2017 is importable
    ensure_cec2017_importable(project_root, cec_root)
    
    # Get function and optimal value
    objective, f_opt = cec2017_function(func_id=1, dim=10)
    
    # Evaluate candidates
    X = np.random.uniform(-100, 100, (100, 10))
    fitness = objective(X)  # Shape: (100,)
    
    # Compute error
    error = fitness - f_opt
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np


def ensure_cec2017_importable(
    project_root: Path,
    override: Optional[Path] = None,
) -> str:
    """
    Ensure the external CEC2017 implementation can be imported.
    
    This function locates the CEC2017 functions.py file, adds its parent
    directory to sys.path, and verifies the module can be imported.
    
    Parameters
    ----------
    project_root : Path
        This project's root folder (used for relative path resolution).
        
    override : Path, optional
        Explicit cec_root path. If provided, only this path is searched.
        If None, several common sibling directories are tried.
        
    Returns
    -------
    str
        Resolved path to the successfully imported functions.py module.
        
    Raises
    ------
    ModuleNotFoundError
        If no valid CEC2017 implementation is found in any searched location.
        Error message includes all tried paths and layout requirements.
        
    Notes
    -----
    Search order when override is None:
    
    1. <project_root>/../00-CEC2017/
    2. <project_root>/../00-CEC-Root/
    3. <project_root>/../00-CEC-ROOT/
    
    Each root is checked for all three layout patterns (A, B, C).
    
    Example
    -------
    >>> project_root = Path("/home/user/gsk_fixed")
    >>> cec_root = Path("/home/user/00-CEC-Root")
    >>> path = ensure_cec2017_importable(project_root, override=cec_root)
    >>> print(path)
    '/home/user/00-CEC-Root/cec2017/functions.py'
    """
    # Determine which root directories to search
    if override is not None:
        roots = [override]
    else:
        # Common sibling directory names
        roots = [
            project_root / ".." / "00-CEC2017",
            project_root / ".." / "00-CEC-Root",
            project_root / ".." / "00-CEC-ROOT",
        ]

    last_exc: Optional[Exception] = None
    tried_roots: list[Path] = []

    # Search each root directory
    for root in roots:
        cec_root = Path(root).resolve()
        tried_roots.append(cec_root)

        # Define candidate layouts: (file_path, sys_path_addition, import_name)
        candidates = [
            # Layout A: cec2017/functions.py
            (cec_root / "cec2017" / "functions.py", cec_root, "cec2017.functions"),
            
            # Layout B: cec2017/cec2017/functions.py
            (
                cec_root / "cec2017" / "cec2017" / "functions.py",
                cec_root / "cec2017",
                "cec2017.functions",
            ),
            
            # Layout C: functions.py (flat)
            (cec_root / "functions.py", cec_root, "functions"),
        ]

        # Try each candidate layout
        for sentinel, sys_path_to_prepend, import_target in candidates:
            if not sentinel.exists():
                continue

            # Add to Python path and attempt import
            sys.path.insert(0, str(sys_path_to_prepend))
            try:
                mod = importlib.import_module(import_target)
                mod_file = getattr(mod, "__file__", None)
                return str(
                    Path(mod_file).resolve() if mod_file else sentinel.resolve()
                )
            except Exception as exc:
                last_exc = exc
                continue

    # No valid implementation found - construct helpful error message
    tried_txt = "\n".join(f"  - {p}" for p in tried_roots)
    hint = (
        "Cannot import the external CEC2017 implementation.\n"
        "Tried these cec_root candidates:\n"
        f"{tried_txt}\n\n"
        "Supported layouts:\n"
        "  A) <cec_root>/cec2017/functions.py\n"
        "  B) <cec_root>/cec2017/cec2017/functions.py\n"
        "  C) <cec_root>/functions.py\n\n"
        "Please ensure the external CEC2017 Python files exist in one of these layouts,\n"
        "or pass an explicit path via:  python scripts/run_gsk.py --cec-root <path>\n"
    )
    raise ModuleNotFoundError(hint) from last_exc


def cec2017_function(
    func_id: int,
    dim: int,
    bounds: Tuple[float, float] = (-100.0, 100.0),
) -> Tuple[Callable[[np.ndarray], np.ndarray], float]:
    """
    Get a vectorized CEC2017 objective function.
    
    Parameters
    ----------
    func_id : int
        Function ID in range [1, 30].
        
    dim : int
        Problem dimension.
        
    bounds : tuple of float, default=(-100.0, 100.0)
        Search space bounds. Passed for reference but not enforced by
        the returned function (boundary handling is done by the optimizer).
        
    Returns
    -------
    objective : callable
        Vectorized objective function.
        Signature: objective(X) -> y where:
        - X: np.ndarray of shape (n, D) or (D,)
        - y: np.ndarray of shape (n,)
        
    f_opt : float
        Known global optimum value.
        CEC2017 convention: f_opt = 100 × func_id
        
    Raises
    ------
    ValueError
        If func_id is outside [1, 30].
        
    ModuleNotFoundError
        If CEC2017 module is not importable.
        Call ensure_cec2017_importable() first.
        
    AttributeError
        If imported module doesn't have 'all_functions' attribute.
        
    Notes
    -----
    The returned objective function handles both single vectors (D,) and
    batches (n, D). Single vectors are automatically reshaped.
    
    If the underlying CEC2017 implementation doesn't support batch
    evaluation, the wrapper automatically falls back to row-by-row
    evaluation.
    
    Example
    -------
    >>> # Get F1 (Shifted and Rotated Bent Cigar) for D=10
    >>> objective, f_opt = cec2017_function(func_id=1, dim=10)
    >>> print(f"Optimal value: {f_opt}")
    Optimal value: 100.0
    >>> 
    >>> # Evaluate 100 random candidates
    >>> X = np.random.uniform(-100, 100, (100, 10))
    >>> fitness = objective(X)
    >>> print(f"Best fitness: {fitness.min():.2f}")
    >>> print(f"Best error: {fitness.min() - f_opt:.2f}")
    """
    # Validate function ID
    if not (1 <= int(func_id) <= 30):
        raise ValueError(f"CEC2017 func_id must be in [1, 30], got {func_id!r}")

    # Import the CEC2017 functions module
    try:
        functions = importlib.import_module("cec2017.functions")
    except ModuleNotFoundError:
        functions = importlib.import_module("functions")

    # Get the list of all functions
    all_funcs = getattr(functions, "all_functions", None)
    if all_funcs is None:
        raise AttributeError(
            "The imported CEC2017 'functions' module does not define "
            "'all_functions'. Please use a compatible CEC2017 implementation."
        )

    # Validate function availability
    if len(all_funcs) < int(func_id):
        raise ValueError(
            f"CEC2017 implementation provides only {len(all_funcs)} functions, "
            f"but func_id={func_id} was requested."
        )

    # Get the specific function (0-indexed)
    f = all_funcs[int(func_id) - 1]

    # CEC2017 optimal value convention: f_opt = 100 × func_id
    f_opt = 100.0 * float(func_id)

    def objective(X: np.ndarray) -> np.ndarray:
        """
        Vectorized objective function wrapper.
        
        Handles both single vectors and batches, with automatic fallback
        to row-by-row evaluation if batch evaluation fails.
        
        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n, D) for batch or (D,) for single.
            
        Returns
        -------
        np.ndarray
            Fitness values of shape (n,).
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Handle 1D input (single candidate)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if X.ndim != 2:
            raise ValueError(
                f"CEC2017 objective expects 1D or 2D array, got shape={X.shape}"
            )

        n = X.shape[0]
        
        # Try batch evaluation first (faster)
        try:
            y = f(X)
            y = np.asarray(y, dtype=np.float64).reshape(-1)
            if y.shape[0] == n:
                return y
        except Exception:
            pass

        # Fallback: evaluate row by row
        y2 = np.empty((n,), dtype=np.float64)
        for i in range(n):
            yi = f(X[i : i + 1, :])
            yi = np.asarray(yi, dtype=np.float64).reshape(-1)
            if yi.size == 0:
                raise ValueError("CEC2017 function returned an empty result")
            y2[i] = float(yi[0])
        return y2

    return objective, f_opt
