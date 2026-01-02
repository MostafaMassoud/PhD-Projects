from __future__ import annotations

"""gsk_baseline.cec2017_adapter

Adapter for an external CEC2017 Python implementation
=====================================================

This project intentionally does **not** vendor the CEC2017 benchmark code.
Instead, you provide a local copy (typically placed at ``../00-CEC2017``).

Unfortunately, third-party CEC2017 Python ports appear in multiple folder layouts.
To be robust, this adapter supports three common layouts:

Layout A (standard package under cec_root)
    ``<cec_root>/cec2017/functions.py``

    Add ``<cec_root>`` to ``sys.path`` then import ``cec2017.functions``.

Layout B (wrapper folder + package folder)
    ``<cec_root>/cec2017/cec2017/functions.py``

    Add ``<cec_root>/cec2017`` to ``sys.path`` then import ``cec2017.functions``.

Layout C (flat module layout; modules directly in cec_root)
    ``<cec_root>/functions.py``

    Add ``<cec_root>`` to ``sys.path`` then import ``functions``.

The :func:`ensure_cec2017_importable` helper tries these layouts in order and returns
the resolved path to the loaded ``functions.py`` for logging/debugging.
"""

from pathlib import Path
from typing import Callable, Optional
import importlib
import sys

import numpy as np


def ensure_cec2017_importable(project_root: Path, override: Optional[Path] = None) -> str:
    """Ensure the external CEC2017 implementation can be imported.

    Parameters
    ----------
    project_root:
        This project's root folder.
    override:
        If provided, this path is used as ``cec_root`` instead of ``../00-CEC-Root``.

    Returns
    -------
    str
        Resolved path to the imported ``functions.py`` module.

    Raises
    ------
    ModuleNotFoundError
        If no supported layout is detected or the module cannot be imported.
    """

    cec_root = (override if override is not None else (project_root / ".." / "00-CEC-Root")).resolve()

    # (sentinel_path, sys_path_to_prepend, import_target, description)
    candidates = [
        (cec_root / "cec2017" / "functions.py", cec_root, "cec2017.functions", "Layout A"),
        (cec_root / "cec2017" / "cec2017" / "functions.py", cec_root / "cec2017", "cec2017.functions", "Layout B"),
        (cec_root / "functions.py", cec_root, "functions", "Layout C"),
    ]

    last_exc: Exception | None = None

    for sentinel, sys_path_to_prepend, import_target, _desc in candidates:
        if not sentinel.exists():
            continue

        # Prepend candidate import root and attempt import.
        sys.path.insert(0, str(sys_path_to_prepend))
        try:
            mod = importlib.import_module(import_target)
            mod_file = getattr(mod, "__file__", None)
            return str(Path(mod_file).resolve() if mod_file else sentinel.resolve())
        except Exception as exc:  # pragma: no cover (depends on user environment)
            last_exc = exc
            continue

    hint = (
        "Cannot import the external CEC2017 implementation.\n"
        f"Tried cec_root: {cec_root}\n\n"
        "Supported layouts:\n"
        "  A) <cec_root>/cec2017/functions.py\n"
        "  B) <cec_root>/cec2017/cec2017/functions.py\n"
        "  C) <cec_root>/functions.py\n\n"
        "Please ensure that '../00-CEC2017' exists relative to this project (or pass --cec-root)\n"
        "and that it contains the required CEC2017 Python files (including 'functions.py').\n"
    )
    raise ModuleNotFoundError(hint) from last_exc


def cec2017_function(func_id: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a vectorized objective function for CEC2017 function ``func_id``.

    The returned callable accepts an array of shape ``(n, D)`` and returns a vector
    of shape ``(n,)`` with the function value for each row.

    Notes
    -----
    The underlying CEC2017 implementations typically evaluate one point at a time,
    so we vectorize via a Python loop. This is fine for baseline benchmarking and
    keeps the adapter simple and dependency-free.
    """

    if not (1 <= int(func_id) <= 30):
        raise ValueError(f"CEC2017 func_id must be in [1, 30], got {func_id!r}")

    # Support both package-style and flat-style layouts.
    try:
        functions = importlib.import_module("cec2017.functions")  # type: ignore
    except ModuleNotFoundError:
        functions = importlib.import_module("functions")  # type: ignore

    all_funcs = getattr(functions, "all_functions", None)
    if all_funcs is None:
        raise AttributeError(
            "The imported CEC2017 'functions' module does not define 'all_functions'. "
            "Please use a compatible CEC2017 Python implementation."
        )

    if len(all_funcs) < int(func_id):
        raise ValueError(
            f"CEC2017 implementation provides only {len(all_funcs)} functions, "
            f"but func_id={func_id} was requested."
        )

    f = all_funcs[int(func_id) - 1]

    def objective(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        y = np.empty((X.shape[0],), dtype=float)
        for i in range(X.shape[0]):
            y[i] = float(f(X[i]))
        return y

    return objective
