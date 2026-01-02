from __future__ import annotations

"""gsk_baseline.cec2017_adapter

CEC2017 benchmark integration (external dependency)
==================================================

This project deliberately **does not vendor** the CEC2017 benchmark code.
Instead, it expects an existing CEC2017 Python implementation to be available
*outside* this repository, typically at:

```
../00-CEC2017
```

relative to the project root (the directory containing ``scripts/`` and
``src/``).

Why an adapter?
---------------
In many research codebases, the benchmark suite is kept in a separate folder to
avoid licensing issues and to keep the optimizer code self-contained. However,
CEC2017 repositories come in multiple directory layouts depending on how the
benchmark was downloaded/extracted.

This adapter provides a robust and deterministic way to import and use the
external CEC2017 implementation without requiring the user to rename folders or
edit the benchmark source.

Supported external-library layouts
----------------------------------
In practice, the external ``00-CEC2017`` folder is often arranged in one of the
following ways.

Layout A (recommended)
    ``<cec_root>/cec2017/functions.py``

Layout B (wrapper directory)
    ``<cec_root>/cec2017/cec2017/functions.py``

Layout C ("flat" folder)
    ``<cec_root>/functions.py`` plus the sibling modules
    ``simple.py``, ``hybrid.py``, ``composition.py``, etc.

The flat layout typically happens when the contents of the upstream ``cec2017``
package directory are copied directly into the folder named ``00-CEC2017``.
Remember that the folder name ``00-CEC2017`` is *not* a valid Python identifier
(it starts with digits and contains a dash), so it cannot be imported as a
normal package.

To support Layout C robustly (and without requiring the user to rename
directories), this adapter can **install an import alias** so that the flat
folder is importable as a package named ``cec2017``. This is done in a
controlled way via ``sys.modules``.

Import safety
-------------
This module performs **no objective evaluations**. It only exposes a thin
wrapper that provides a batch-evaluation callable for a given CEC2017 function
ID.
"""

from pathlib import Path
import importlib
import importlib.util
import sys
import types
from typing import Callable, Optional

import numpy as np


def _prepend_sys_path(path: Path) -> None:
    """Prepend a path to ``sys.path`` if it is not already present."""

    p = str(path)
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _purge_cec2017_modules() -> None:
    """Remove previously imported ``cec2017`` modules.

    This is important for reproducibility in long-lived Python sessions
    (e.g. notebooks) where ``cec2017`` may already be imported from a different
    location.
    """

    for k in list(sys.modules.keys()):
        if k == "cec2017" or k.startswith("cec2017."):
            del sys.modules[k]


def _install_cec2017_alias_package(flat_dir: Path) -> None:
    """Install a *directory* as an importable package named ``cec2017``.

    This supports the "flat" external-library layout where the benchmark files
    live directly inside ``<cec_root>`` (e.g. ``<cec_root>/functions.py``)
    rather than inside ``<cec_root>/cec2017/functions.py``.

    Parameters
    ----------
    flat_dir:
        Directory that contains the CEC2017 modules directly (``functions.py``,
        ``simple.py``, ``hybrid.py``, ...).
    """

    flat_dir = flat_dir.resolve()

    # If 'cec2017' is already imported, keep it only if it points to our target.
    if "cec2017" in sys.modules:
        mod = sys.modules["cec2017"]
        try:
            paths = [str(p) for p in getattr(mod, "__path__", [])]
        except Exception:
            paths = []
        if str(flat_dir) in paths:
            return

        # Otherwise remove and replace.
        _purge_cec2017_modules()

    init_py = flat_dir / "__init__.py"
    if init_py.exists():
        spec = importlib.util.spec_from_file_location(
            "cec2017",
            init_py,
            submodule_search_locations=[str(flat_dir)],
        )
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(
                f"Could not create an import spec for the CEC2017 alias package at {init_py}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules["cec2017"] = module
        spec.loader.exec_module(module)
    else:
        # Minimal namespace-style package.
        module = types.ModuleType("cec2017")
        module.__path__ = [str(flat_dir)]  # type: ignore[attr-defined]
        module.__file__ = str(init_py)
        sys.modules["cec2017"] = module


def resolve_cec2017_root(project_root: Path, override: Optional[Path] = None) -> Path:
    """Resolve the external CEC2017 library root."""

    if override is not None:
        return override.resolve()
    return (project_root.parent / "00-CEC2017").resolve()


def ensure_cec2017_importable(project_root: Path, override: Optional[Path] = None) -> str:
    """Ensure the external CEC2017 implementation can be imported.

    Parameters
    ----------
    project_root:
        Root of this GSK project.
    override:
        Optional explicit path to the external CEC2017 folder.

    Returns
    -------
    str
        File path of the imported ``cec2017.functions`` module. This is used for
        metadata logging.

    Raises
    ------
    ModuleNotFoundError
        If the external CEC2017 library cannot be imported.

    Notes
    -----
    The function supports Layout A/B/C described in the module docstring. It is
    intentionally defensive because repository layouts vary and many users run
    the CLI from different working directories.
    """

    cec_root = resolve_cec2017_root(project_root, override)

    def _try_import() -> str:
        functions = importlib.import_module("cec2017.functions")
        return str(getattr(functions, "__file__", "<cec2017.functions>"))

    attempted: list[str] = []
    last_exc: Optional[BaseException] = None

    # Start from a clean slate so path changes take effect.
    _purge_cec2017_modules()

    # --- Direct (non-scanning) checks in priority order ---
    # Layout A: <cec_root>/cec2017/functions.py
    if (cec_root / "cec2017" / "functions.py").exists():
        attempted.append(f"Layout A via sys.path: {cec_root}")
        _prepend_sys_path(cec_root)
        try:
            return _try_import()
        except Exception as exc:  # pragma: no cover
            last_exc = exc

    # Layout B: <cec_root>/cec2017/cec2017/functions.py
    if (cec_root / "cec2017" / "cec2017" / "functions.py").exists():
        attempted.append(f"Layout B via sys.path: {cec_root / 'cec2017'}")
        _prepend_sys_path(cec_root / "cec2017")
        _purge_cec2017_modules()
        try:
            return _try_import()
        except Exception as exc:  # pragma: no cover
            last_exc = exc

    # Layout C: <cec_root>/functions.py (flat folder)
    if (cec_root / "functions.py").exists():
        attempted.append(f"Layout C via alias package: {cec_root}")
        _purge_cec2017_modules()
        try:
            _install_cec2017_alias_package(cec_root)
            return _try_import()
        except Exception as exc:  # pragma: no cover
            last_exc = exc

    # --- Fallback: scan for common nested layouts under cec_root ---
    max_depth = 6

    nested_pkg: list[Path] = []
    nested_flat: list[Path] = []
    try:
        for p in cec_root.rglob("functions.py"):
            # Limit depth for predictability.
            try:
                rel = p.relative_to(cec_root)
                if len(rel.parts) > max_depth:
                    continue
            except Exception:
                continue

            # */cec2017/functions.py
            if p.name == "functions.py" and p.parent.name == "cec2017":
                nested_pkg.append(p)
                continue

            # Possible flat layout in a subfolder.
            base = p.parent
            if (
                (base / "simple.py").exists()
                and (base / "hybrid.py").exists()
                and (base / "composition.py").exists()
            ):
                nested_flat.append(p)
    except Exception:
        # Ignore scanning failures and proceed to error.
        nested_pkg = []
        nested_flat = []

    # Try nested package candidates first (shortest path wins).
    nested_pkg.sort(key=lambda x: len(x.relative_to(cec_root).parts))
    for func_path in nested_pkg:
        parent = func_path.parent.parent
        attempted.append(f"Scanned nested package via sys.path: {parent}")
        _prepend_sys_path(parent)
        _purge_cec2017_modules()
        try:
            return _try_import()
        except Exception as exc:  # pragma: no cover
            last_exc = exc

    # Try nested flat candidates.
    nested_flat.sort(key=lambda x: len(x.relative_to(cec_root).parts))
    for func_path in nested_flat:
        flat_dir = func_path.parent
        attempted.append(f"Scanned flat layout via alias package: {flat_dir}")
        _purge_cec2017_modules()
        try:
            _install_cec2017_alias_package(flat_dir)
            return _try_import()
        except Exception as exc:  # pragma: no cover
            last_exc = exc

    # If nothing worked, raise a helpful error.
    expected_a = cec_root / "cec2017" / "functions.py"
    expected_b = cec_root / "cec2017" / "cec2017" / "functions.py"
    expected_c = cec_root / "functions.py"

    found_functions: list[str] = []
    try:
        for p in cec_root.rglob("functions.py"):
            try:
                rel = p.relative_to(cec_root)
                if len(rel.parts) <= max_depth:
                    found_functions.append(str(p))
            except Exception:
                continue
    except Exception:
        found_functions = []

    found_block = "\n".join(f"  - {p}" for p in found_functions[:20]) if found_functions else "  (none)"
    attempts_block = "\n".join(f"  - {a}" for a in attempted) if attempted else "  (none)"

    msg = (
        "Cannot import the external CEC2017 implementation as package 'cec2017'.\n\n"
        f"External root: {cec_root}\n\n"
        "Supported layouts:\n"
        f"  A) {expected_a}\n"
        f"  B) {expected_b}\n"
        f"  C) {expected_c}\n\n"
        "Discovered 'functions.py' candidates (limited depth):\n"
        f"{found_block}\n\n"
        "Import strategies attempted:\n"
        f"{attempts_block}\n\n"
        "Tip: If your CEC2017 folder is somewhere else, pass --cec-root <path> to scripts/run_gsk.py.\n"
    )
    if last_exc is not None:
        msg += f"\nLast import error: {type(last_exc).__name__}: {last_exc}\n"

    raise ModuleNotFoundError(msg)


def cec2017_function(func_id: int) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that evaluates a CEC2017 function for a batch population.

    Parameters
    ----------
    func_id:
        CEC2017 function id in ``{1, ..., 30}``.

    Returns
    -------
    Callable
        A function ``f(X)`` where ``X`` is a 2D array with shape ``(n, D)`` and
        the return value is a 1D array with shape ``(n,)``.

    Notes
    -----
    This wraps ``cec2017.functions.all_functions[func_id - 1]``.
    """

    import cec2017.functions as functions  # type: ignore

    if not (1 <= int(func_id) <= len(functions.all_functions)):
        raise ValueError(
            f"Invalid CEC2017 function id {func_id}; expected 1..{len(functions.all_functions)}."
        )

    f = functions.all_functions[int(func_id) - 1]

    def _evaluate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"CEC2017 objective expects 2D array (n,D), got shape={X.shape}")
        y = np.asarray(f(X), dtype=np.float64)
        if y.shape != (X.shape[0],):
            raise ValueError(f"CEC2017 returned shape {y.shape}, expected ({X.shape[0]},)")
        return y

    return _evaluate
