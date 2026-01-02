# cec2017.functions
# Author: Duncan Tilley
# Combines simple, hybrid and composition functions (f1 - f30) into a single
# module

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .simple import *  # noqa: F401,F403
from .hybrid import *  # noqa: F401,F403
from .composition import *  # noqa: F401,F403

# ---------------------------------------------------------------------------
# Canonical function list (1..30)
# ---------------------------------------------------------------------------

all_functions = [
    f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
    f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
]

# ---------------------------------------------------------------------------
# Compatibility wrappers (non-invasive)
# ---------------------------------------------------------------------------
# Some refactored runners expect helper names like `cec2017_func`,
# `cec2017_bounds`, and `cec2017_f_name`. The benchmark functions above remain
# unchanged; these helpers only provide a thin convenience layer.
#
# IMPORTANT: These wrappers must not change benchmark behaviour.
# ---------------------------------------------------------------------------

_CEC2017_DIMS = {2, 10, 20, 30, 50, 100}


def _guess_dim_from_args(args: Sequence[object], dim: Optional[int]) -> int:
    if dim is not None:
        return int(dim)
    if len(args) == 1:
        return int(args[0])  # type: ignore[arg-type]
    if len(args) >= 2:
        a = int(args[0])  # type: ignore[arg-type]
        b = int(args[1])  # type: ignore[arg-type]

        # Prefer known benchmark dimensions.
        a_is_dim = a in _CEC2017_DIMS
        b_is_dim = b in _CEC2017_DIMS
        if a_is_dim and not b_is_dim:
            return a
        if b_is_dim and not a_is_dim:
            return b
        if a_is_dim and b_is_dim:
            return a  # ambiguous but identical

        # If one looks like a function id (1..30), the other is probably the dimension.
        a_is_fid = 1 <= a <= 30
        b_is_fid = 1 <= b <= 30
        if a_is_fid and not b_is_fid:
            return b
        if b_is_fid and not a_is_fid:
            return a

        # Fallback: larger number is usually the dimension.
        return max(a, b)

    raise TypeError(
        "cec2017_bounds requires at least one positional argument "
        "(dimension or (fid, dimension))."
    )


def cec2017_bounds(
    *args: int,
    dim: Optional[int] = None,
    min_region: float = -100.0,
    max_region: float = 100.0,
) -> np.ndarray:
    """Return benchmark bounds as a (2, D) matrix [[lb..],[ub..]].

    Accepts:
      - cec2017_bounds(D)
      - cec2017_bounds(func_id, D)
      - cec2017_bounds(D, func_id)
      - cec2017_bounds(dim=D, ...)

    All CEC2017 functions use the same default bounds [-100, 100].
    """
    D = _guess_dim_from_args(args, dim)
    lb = np.full(D, float(min_region), dtype=np.float64)
    ub = np.full(D, float(max_region), dtype=np.float64)
    return np.vstack((lb, ub))


def cec2017_f_name(func_id: int) -> str:
    """Human-friendly function label."""
    return f"F{int(func_id):02d}"


def cec2017_fopt(func_id: int) -> float:
    """Known optimum/bias value for CEC2017 functions in this implementation.

    The official suite uses biases 100, 200, ..., 3000 for F1..F30.
    """
    return 100.0 * float(int(func_id))


def cec2017_optimum(func_id: int) -> float:
    """Return the known optimum (bias) value for a given function.

    Some refactored runners expect `cec2017_optimum`, while others use
    `cec2017_fopt`. In this benchmark implementation, the official optimum
    values are simply the bias terms 100, 200, ..., 3000 for F1..F30.
    """
    return cec2017_fopt(func_id)


def cec2017_test_func(
    x: Union[np.ndarray, Sequence[float], int],
    func_id: Optional[int] = None,
    *args: object,
    **kwargs: object,
) -> Union[np.ndarray, float]:
    """Evaluate a CEC2017 function.

    Supports both calling conventions:
      - cec2017_test_func(pop, func_id)
      - cec2017_test_func(func_id, pop)

    Where pop can be:
      - 2D array: (pop_size, D)  -> returns (pop_size,) array
      - 1D array: (D,)           -> returns scalar float
    """
    # Allow keyword alias
    if func_id is None:
        fid_kw = kwargs.get("func_num", kwargs.get("func", kwargs.get("f", None)))
        if fid_kw is not None:
            try:
                func_id = int(fid_kw)  # type: ignore[arg-type]
            except Exception:
                func_id = None

    # Handle swapped arguments: (func_id, pop)
    if func_id is None and isinstance(x, (int, np.integer)):
        func_id = int(x)
        if not args:
            raise TypeError("cec2017_test_func(func_id, pop) missing pop argument.")
        x = args[0]  # type: ignore[assignment]

    if func_id is None:
        raise TypeError("cec2017_test_func requires a func_id (1..30).")

    fid = int(func_id)
    if fid < 1 or fid > len(all_functions):
        raise ValueError(f"Invalid func_id={fid}; expected 1..{len(all_functions)}.")

    pop = np.asarray(x, dtype=np.float64)
    if pop.ndim == 1:
        pop2 = pop.reshape(1, -1)
        vals = np.asarray(all_functions[fid - 1](pop2), dtype=np.float64).ravel()
        return float(vals[0])
    if pop.ndim != 2:
        raise ValueError(f"x must be 1D or 2D, got shape {pop.shape}")

    vals = np.asarray(all_functions[fid - 1](pop), dtype=np.float64)
    return vals


# Backward-compatible aliases (different refactors expect different names)
cec2017_func = cec2017_test_func  # <-- required by your experiment.py
cec17_func = cec2017_test_func  # type: ignore

cec17_bounds = cec2017_bounds
cec17_f_name = cec2017_f_name
cec17_fopt = cec2017_fopt
