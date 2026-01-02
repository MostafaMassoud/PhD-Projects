from __future__ import annotations

"""gsk_baseline.budget

Centralized objective-evaluation budget control (NFE)
====================================================

CEC-style optimization studies typically measure algorithmic cost in terms of
**NFEs** (Number of Function Evaluations). In this package:

- Every objective-function call counts as **exactly 1 NFE**.
- The total budget is fixed per run as: ``max_nfes = 10000 * D``.
- **No objective function may be called outside this controller.**

Why a controller?
-----------------
A centralized controller makes it easy to *prove* that the implementation is
budget safe. All evaluation sites in the optimizer route through this class.

Budget safety guarantee
-----------------------
The controller enforces the following hard guarantees:

1. The internal counter is incremented by exactly the number of evaluated points.
2. If ``nfes == max_nfes``, *no further evaluations are allowed*.
3. The controller will never allow ``nfes > max_nfes``.

If a caller requests more evaluations than remaining budget, the controller can
be used in two safe ways:

- **Preferred**: the caller checks :meth:`remaining` and stops before requesting.
- **Fallback**: the caller can request a batch evaluation and accept a truncated
  batch of size ``remaining`` (this still never exceeds budget).

This design supports the requirement: "Stop immediately once the budget is exhausted.".
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


class BudgetExhausted(RuntimeError):
    """Raised when an evaluation is attempted after the budget is exhausted."""


@dataclass
class BudgetReport:
    """A small summary returned by :meth:`BudgetController.report`."""

    max_nfes: int
    nfes_used: int

    @property
    def remaining(self) -> int:
        return max(0, int(self.max_nfes) - int(self.nfes_used))


class BudgetController:
    """Centralized NFE (Number of Function Evaluations) controller.

    Parameters
    ----------
    max_nfes:
        Maximum number of evaluations allowed.
    objective:
        Callable objective function ``f(X)`` where ``X`` is a 2D array of shape
        ``(n, D)`` and the return value is a 1D array of shape ``(n,)``.

    Notes
    -----
    - The controller is **stateful**: it tracks evaluations used.
    - The controller is **strictly budget-safe**: it never evaluates beyond
      ``max_nfes``.
    """

    def __init__(self, *, max_nfes: int, objective: Callable[[np.ndarray], np.ndarray]):
        self._max_nfes = int(max_nfes)
        if self._max_nfes < 0:
            raise ValueError("max_nfes must be non-negative")
        self._objective = objective
        self._nfes_used = 0

    @property
    def max_nfes(self) -> int:
        return self._max_nfes

    @property
    def nfes_used(self) -> int:
        return self._nfes_used

    def remaining(self) -> int:
        """Return remaining budget (never negative)."""

        return max(0, self._max_nfes - self._nfes_used)

    def exhausted(self) -> bool:
        """Return True if no evaluations remain."""

        return self._nfes_used >= self._max_nfes

    def report(self) -> BudgetReport:
        """Return a simple budget report for logging."""

        return BudgetReport(max_nfes=self._max_nfes, nfes_used=self._nfes_used)

    def _consume(self, n: int) -> None:
        """Consume ``n`` evaluations and assert budget safety."""

        self._nfes_used += int(n)
        # Hard guard: exceeding budget is a bug.
        if self._nfes_used > self._max_nfes:
            raise AssertionError(
                f"BudgetController exceeded budget: nfes_used={self._nfes_used} > max_nfes={self._max_nfes}"
            )

    def eval_batch(self, X: np.ndarray, *, allow_truncate: bool = True) -> np.ndarray:
        """Evaluate a batch of candidate solutions.

        Parameters
        ----------
        X:
            2D array of candidates with shape ``(n, D)``.
        allow_truncate:
            If True (default), when ``n > remaining`` the controller evaluates
            only the first ``remaining`` candidates and returns the shorter
            vector. If False, a :class:`BudgetExhausted` error is raised.

        Returns
        -------
        np.ndarray
            1D vector of objective values with shape ``(k,)`` where
            ``k = min(n, remaining)`` if truncation is allowed, else ``k = n``.

        Raises
        ------
        BudgetExhausted
            If the budget is exhausted, or if truncation is disabled and the
            requested batch does not fit.
        """

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n,D), got shape={X.shape}")

        n = int(X.shape[0])
        if n <= 0:
            return np.empty((0,), dtype=np.float64)

        rem = self.remaining()
        if rem <= 0:
            raise BudgetExhausted("Evaluation budget exhausted")

        if n > rem:
            if not allow_truncate:
                raise BudgetExhausted(
                    f"Requested {n} evaluations but only {rem} remain (truncation disabled)."
                )
            n_eval = rem
            X_eval = X[:n_eval, :]
        else:
            n_eval = n
            X_eval = X

        y = np.asarray(self._objective(X_eval), dtype=np.float64)
        if y.shape != (n_eval,):
            raise ValueError(f"Objective returned shape {y.shape}, expected ({n_eval},)")

        self._consume(n_eval)
        return y

    def eval_one(self, x: np.ndarray) -> float:
        """Evaluate a single candidate.

        This is a convenience wrapper around :meth:`eval_batch`.
        """

        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D (D,), got shape={x.shape}")
        y = self.eval_batch(x.reshape(1, -1), allow_truncate=False)
        return float(y[0])
