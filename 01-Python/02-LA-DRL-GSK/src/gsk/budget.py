"""
Budget Controller for Function Evaluation Tracking
===================================================

This module provides centralized budget control for optimization algorithms,
ensuring strict enforcement of the maximum number of function evaluations (NFEs).

Why Budget Control Matters
--------------------------
In benchmark optimization (e.g., CEC competitions), algorithms are compared
under a fixed computational budget measured in function evaluations:

    Budget = max_nfes = 10000 × D

where D is the problem dimension. This creates a fair comparison:
- All algorithms get the same number of "chances" to find the optimum
- No algorithm can "cheat" by doing more evaluations
- Results are comparable across different implementations

Budget Controller Guarantees
----------------------------
1. **Strict accounting**: Every evaluation is counted exactly once
2. **Never exceeds budget**: nfes_used ≤ max_nfes is always maintained
3. **Clear exhaustion signals**: Raises BudgetExhausted when limit reached
4. **Truncation support**: Can partially evaluate batches at budget boundary

Usage Pattern
-------------
The controller wraps the objective function and tracks all calls:

    # Create controller with budget and objective
    budget = BudgetController(max_nfes=100000, objective=cec_func)
    
    # Evaluate candidates (budget automatically tracked)
    fitness = budget.eval_batch(population)
    
    # Check remaining budget
    if budget.remaining() < pop_size:
        break  # Not enough budget for another generation
        
    # Check if exhausted
    if budget.exhausted():
        return best_solution

Integration with GSK
--------------------
The GSK algorithm uses the budget controller to:
1. Evaluate initial population (100 NFEs for pop_size=100)
2. Evaluate offspring each generation (100 NFEs per generation)
3. Stop when budget exhausted or insufficient for next generation

With max_nfes = 100,000 and pop_size = 100:
- Initial population: 100 NFEs
- Remaining for evolution: 99,900 NFEs
- Maximum generations: 99,900 / 100 = 999 generations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class BudgetExhausted(RuntimeError):
    """
    Exception raised when attempting evaluation after budget is exhausted.
    
    This exception signals that the optimization should terminate gracefully.
    The algorithm should catch this and return the best solution found so far.
    
    Example
    -------
    >>> try:
    ...     fitness = budget.eval_batch(offspring)
    ... except BudgetExhausted:
    ...     return best_solution
    """
    pass


@dataclass
class BudgetReport:
    """
    Summary of budget usage for logging and monitoring.
    
    Attributes
    ----------
    max_nfes : int
        Total budget allocated.
    nfes_used : int
        Evaluations consumed so far.
    """
    max_nfes: int
    nfes_used: int

    @property
    def remaining(self) -> int:
        """Remaining evaluations available."""
        return max(0, int(self.max_nfes) - int(self.nfes_used))
    
    @property
    def fraction_used(self) -> float:
        """Fraction of budget consumed (0.0 to 1.0)."""
        if self.max_nfes <= 0:
            return 1.0
        return min(1.0, self.nfes_used / self.max_nfes)


class BudgetController:
    """
    Centralized controller for tracking function evaluation budget.
    
    This class wraps an objective function and enforces strict budget limits.
    All evaluations must go through this controller to ensure accurate counting.
    
    Parameters
    ----------
    max_nfes : int
        Maximum number of function evaluations allowed.
        For CEC2017: max_nfes = 10000 × dimension.
        
    objective : callable
        Objective function to minimize.
        Signature: f(X) -> y where:
        - X is np.ndarray of shape (n, D) with n candidates
        - y is np.ndarray of shape (n,) with fitness values
        
    Attributes
    ----------
    max_nfes : int
        Budget limit (read-only).
    nfes_used : int
        Current evaluation count (read-only).
        
    Methods
    -------
    eval_batch(X, allow_truncate=True)
        Evaluate multiple candidates.
    eval_one(x)
        Evaluate single candidate.
    remaining()
        Return remaining budget.
    exhausted()
        Check if budget is exhausted.
    report()
        Get budget summary.
        
    Example
    -------
    >>> def sphere(X):
    ...     return np.sum(X**2, axis=1)
    >>> 
    >>> budget = BudgetController(max_nfes=1000, objective=sphere)
    >>> 
    >>> # Evaluate 100 candidates
    >>> pop = np.random.randn(100, 10)
    >>> fitness = budget.eval_batch(pop)
    >>> 
    >>> print(f"Used: {budget.nfes_used}, Remaining: {budget.remaining()}")
    Used: 100, Remaining: 900
    """

    def __init__(
        self,
        *,
        max_nfes: int,
        objective: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """
        Initialize budget controller.
        
        Parameters
        ----------
        max_nfes : int
            Maximum evaluations allowed. Must be non-negative.
        objective : callable
            Objective function f(X) -> y.
        """
        self._max_nfes = int(max_nfes)
        if self._max_nfes < 0:
            raise ValueError("max_nfes must be non-negative")
        self._objective = objective
        self._nfes_used = 0

    # ========================================================================
    # Properties (read-only access to internal state)
    # ========================================================================

    @property
    def max_nfes(self) -> int:
        """Maximum number of evaluations allowed."""
        return self._max_nfes

    @property
    def nfes_used(self) -> int:
        """Number of evaluations consumed so far."""
        return self._nfes_used

    # ========================================================================
    # Budget Status Methods
    # ========================================================================

    def remaining(self) -> int:
        """
        Return remaining evaluation budget.
        
        Returns
        -------
        int
            Number of evaluations remaining (always >= 0).
        """
        return max(0, self._max_nfes - self._nfes_used)

    def exhausted(self) -> bool:
        """
        Check if budget is exhausted.
        
        Returns
        -------
        bool
            True if no evaluations remain.
        """
        return self._nfes_used >= self._max_nfes

    def report(self) -> BudgetReport:
        """
        Get budget summary for logging.
        
        Returns
        -------
        BudgetReport
            Summary with max_nfes, nfes_used, and computed properties.
        """
        return BudgetReport(max_nfes=self._max_nfes, nfes_used=self._nfes_used)

    # ========================================================================
    # Internal Budget Accounting
    # ========================================================================

    def _consume(self, n: int) -> None:
        """
        Consume n evaluations from budget.
        
        This is an internal method that updates the counter.
        Raises AssertionError if budget would be exceeded (should never
        happen if eval_batch is used correctly).
        
        Parameters
        ----------
        n : int
            Number of evaluations to consume.
        """
        self._nfes_used += int(n)
        
        # Safety assertion - should never trigger if eval_batch is correct
        if self._nfes_used > self._max_nfes:
            raise AssertionError(
                f"BudgetController internal error: exceeded budget. "
                f"nfes_used={self._nfes_used} > max_nfes={self._max_nfes}"
            )

    # ========================================================================
    # Evaluation Methods
    # ========================================================================

    def eval_batch(
        self,
        X: np.ndarray,
        *,
        allow_truncate: bool = True,
    ) -> np.ndarray:
        """
        Evaluate a batch of candidate solutions.
        
        Parameters
        ----------
        X : np.ndarray
            Candidate solutions, shape (n, D) where:
            - n = number of candidates
            - D = problem dimension
            
        allow_truncate : bool, default=True
            Behavior when n > remaining budget:
            - True: Evaluate first `remaining` candidates only
            - False: Raise BudgetExhausted immediately
            
        Returns
        -------
        np.ndarray
            Fitness values, shape (n_eval,) where n_eval ≤ n.
            If truncated, n_eval = remaining budget.
            
        Raises
        ------
        BudgetExhausted
            If budget exhausted and no evaluations possible.
            Also raised if allow_truncate=False and n > remaining.
            
        ValueError
            If X is not 2D or objective returns wrong shape.
            
        Notes
        -----
        Truncation behavior:
        
        If budget has 50 evaluations remaining and you request 100:
        - allow_truncate=True: Evaluates first 50, returns 50 fitness values
        - allow_truncate=False: Raises BudgetExhausted immediately
        
        The truncation option is useful for:
        - Initial population: allow_truncate=True (partial init is ok)
        - Offspring evaluation: allow_truncate=False (need full generation)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n, D), got shape={X.shape}")

        n = int(X.shape[0])
        if n <= 0:
            return np.empty((0,), dtype=np.float64)

        rem = self.remaining()
        
        # Check if budget is already exhausted
        if rem <= 0:
            raise BudgetExhausted("Evaluation budget exhausted")

        # Handle case where request exceeds remaining budget
        if n > rem:
            if not allow_truncate:
                raise BudgetExhausted(
                    f"Requested {n} evaluations but only {rem} remain "
                    "(truncation disabled)."
                )
            # Truncate: evaluate only what we can afford
            n_eval = rem
            X_eval = X[:n_eval, :]
        else:
            # Full batch fits within budget
            n_eval = n
            X_eval = X

        # Call objective function
        y = np.asarray(self._objective(X_eval), dtype=np.float64)
        
        # Validate output shape
        if y.shape != (n_eval,):
            raise ValueError(
                f"Objective returned shape {y.shape}, expected ({n_eval},)"
            )

        # Update budget counter
        self._consume(n_eval)
        
        return y

    def eval_one(self, x: np.ndarray) -> float:
        """
        Evaluate a single candidate solution.
        
        Convenience method for evaluating one solution at a time.
        Equivalent to eval_batch(x.reshape(1, -1))[0].
        
        Parameters
        ----------
        x : np.ndarray
            Single candidate, shape (D,).
            
        Returns
        -------
        float
            Fitness value.
            
        Raises
        ------
        BudgetExhausted
            If no evaluations remain.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError(f"x must be 1D (D,), got shape={x.shape}")
        y = self.eval_batch(x.reshape(1, -1), allow_truncate=False)
        return float(y[0])
