"""
Gaining-Sharing Knowledge (GSK) Algorithm - Baseline Implementation
====================================================================

This module implements the GSK algorithm as described in:

    Mohamed, A. W., Hadi, A. A., & Mohamed, A. K. (2020).
    "Gaining-Sharing Knowledge Based Algorithm for Solving Optimization
    Problems: A Novel Nature-Inspired Algorithm."
    International Journal of Machine Learning and Cybernetics, 11, 1501-1529.
    https://doi.org/10.1007/s13042-019-01053-x

Algorithm Overview
------------------
GSK is a population-based metaheuristic inspired by the human process of
gaining and sharing knowledge. The algorithm maintains a population of
candidate solutions that evolve through two complementary phases:

1. **Junior Gaining-Sharing Phase (Exploration)**
   - Models how junior individuals learn from their immediate peers
   - Uses rank-based neighborhood: each individual learns from neighbors
     close to it in the fitness ranking
   - Promotes exploration and diversity

2. **Senior Gaining-Sharing Phase (Exploitation)**  
   - Models how senior individuals learn from stratified groups
   - Population is divided into three groups based on fitness:
     * Top 10% (best/elite)
     * Middle 80%
     * Bottom 10% (worst)
   - Promotes exploitation of promising regions

Key Parameters
--------------
- KF (Knowledge Factor): Controls step size, default 0.5
- KR (Knowledge Ratio): Crossover probability, default 0.9
- K (Knowledge Rate): Controls junior/senior phase balance, default 10

The balance between junior (exploration) and senior (exploitation) phases
shifts dynamically during optimization:
- Early generations: More junior phase (exploration)
- Late generations: More senior phase (exploitation)

This transition is controlled by:
    D_junior = ceil(D * (1 - g/G_max)^K)
    
where D is dimension, g is current generation, G_max is max generations,
and K is the knowledge rate parameter.

Implementation Notes
--------------------
- This is a baseline implementation without any enhancements
- Uses MATLAB-compatible random number generation for reproducibility
- Boundary violations are handled using L-SHADE midpoint repair
- Greedy selection: offspring replaces parent only if better
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np

from .budget import BudgetController, BudgetExhausted
from .rand_matlab import rand_matlab
from .bound_constraint import bound_constraint
from .gained_shared_junior import gained_shared_junior_r1r2r3
from .gained_shared_senior import gained_shared_senior_r1r2r3


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class GSKConfig:
    """
    Configuration for a single GSK optimization run.
    
    Attributes
    ----------
    dim : int
        Problem dimensionality (number of decision variables).
        
    pop_size : int, default=100
        Population size (number of candidate solutions).
        Recommended: 100 for most problems.
        
    bounds : tuple of float, default=(-100.0, 100.0)
        Search space bounds (lower, upper) applied to all dimensions.
        CEC2017 benchmark uses [-100, 100].
        
    max_nfes : int or None, default=None
        Maximum number of function evaluations.
        If None, uses CEC2017 convention: 10000 * dim.
        
    seed : int, default=123456
        Random seed for reproducibility.
        
    KF : float, default=0.5
        Knowledge Factor - controls mutation step size.
        Range: [0, 1]. Higher values = larger steps.
        
    KR : float, default=0.9
        Knowledge Ratio - crossover probability.
        Range: [0, 1]. Higher values = more dimensions modified.
        
    Kexp : float, default=10.0
        Knowledge Rate exponent - controls exploration/exploitation balance.
        Higher values = faster transition to exploitation.
    """
    dim: int
    pop_size: int = 100
    bounds: Tuple[float, float] = (-100.0, 100.0)
    max_nfes: Optional[int] = None
    seed: int = 123456
    KF: float = 0.5
    KR: float = 0.9
    Kexp: float = 10.0

    def resolved_max_nfes(self) -> int:
        """Return evaluation budget (default: 10000 * D per CEC2017)."""
        if self.max_nfes is None:
            return 10000 * self.dim
        return int(self.max_nfes)

    def bounds_matrix(self) -> np.ndarray:
        """
        Return bounds as (2, D) array.
        
        Returns
        -------
        np.ndarray
            Shape (2, D) where row 0 is lower bounds, row 1 is upper bounds.
        """
        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        return np.vstack([
            np.full(self.dim, lo, dtype=np.float64),
            np.full(self.dim, hi, dtype=np.float64),
        ])


# ============================================================================
# Output Structures
# ============================================================================

@dataclass
class GSKGenerationLog:
    """Per-generation diagnostics for analysis and debugging."""
    gen: int              # Generation number
    evals_used: int       # Cumulative function evaluations
    best_fitness: float   # Best-so-far fitness value
    diversity: float      # Population diversity (mean std across dimensions)
    stagnation: int       # Generations without improvement
    stage: str            # Optimization stage: EARLY/MID/LATE
    jr_mask_rate: float   # Fraction of dimensions using junior phase
    sr_mask_rate: float   # Fraction of dimensions using senior phase
    KF: float             # Current KF value
    KR: float             # Current KR value
    Kexp: float           # Current Kexp value


@dataclass
class GSKResult:
    """Final result of a GSK optimization run."""
    best_x: np.ndarray    # Best solution found (D-dimensional vector)
    best_f: float         # Best fitness value (objective function value)
    nfes_used: int        # Total function evaluations consumed
    max_nfes: int         # Budget limit
    stop_reason: str      # Why optimization stopped


# ============================================================================
# Main Algorithm
# ============================================================================

def gsk_optimize(
    *,
    objective: Callable[[np.ndarray], np.ndarray],
    config: GSKConfig,
    return_history: bool = False,
    generation_callback: Optional[Callable[[GSKGenerationLog], None]] = None,
) -> Union[GSKResult, Tuple[GSKResult, np.ndarray]]:
    """
    Run the GSK algorithm on an objective function.
    
    Parameters
    ----------
    objective : callable
        Objective function to minimize.
        Signature: f(X) -> y where X is (n, D) array, y is (n,) array.
        
    config : GSKConfig
        Algorithm configuration.
        
    return_history : bool, default=False
        If True, also return best-so-far fitness after each evaluation.
        
    generation_callback : callable, optional
        Called after each generation with GSKGenerationLog for monitoring.
        
    Returns
    -------
    GSKResult or (GSKResult, np.ndarray)
        Optimization result, optionally with convergence history.
        
    Algorithm Steps
    ---------------
    1. Initialize population uniformly in search space
    2. For each generation until budget exhausted:
       a. Compute junior/senior dimension ratio based on generation
       b. Sort population by fitness
       c. Generate junior vectors using neighborhood learning
       d. Generate senior vectors using group-based learning
       e. Apply boundary constraints
       f. Create offspring by mixing junior/senior contributions
       g. Evaluate offspring
       h. Greedy selection: keep better of parent/offspring
    3. Return best solution found
    """
    
    # ========================================================================
    # Setup
    # ========================================================================
    
    D = int(config.dim)           # Problem dimension
    NP = int(config.pop_size)     # Population size
    
    if D <= 0:
        raise ValueError("dim must be positive")
    if NP < 4:
        raise ValueError("pop_size must be at least 4 (for index selection)")

    max_nfes = config.resolved_max_nfes()
    rng = np.random.RandomState(config.seed)
    
    # Bounds matrix: lu[0,:] = lower, lu[1,:] = upper
    lu = config.bounds_matrix()
    lower = lu[0, :]
    upper = lu[1, :]
    
    # Budget controller tracks function evaluations
    budget = BudgetController(max_nfes=max_nfes, objective=objective)

    # ========================================================================
    # Population Initialization
    # ========================================================================
    # Initialize population uniformly in [lower, upper]
    # Formula: x = lower + rand * (upper - lower)
    
    popold = lower + rand_matlab(rng, NP, D) * (upper - lower)

    # Evaluate initial population
    try:
        fitness = budget.eval_batch(popold, allow_truncate=True)
    except BudgetExhausted:
        return _make_empty_result(D, budget, max_nfes, return_history)

    if fitness.shape[0] == 0:
        return _make_empty_result(D, budget, max_nfes, return_history)

    # Track best solution found
    best_idx = int(np.argmin(fitness))
    best_f = float(fitness[best_idx])
    best_x = popold[best_idx, :].copy()
    stagnation = 0

    # Optional: track convergence history
    history = np.minimum.accumulate(fitness.copy()) if return_history else None

    # Algorithm parameters
    KF = float(config.KF)      # Knowledge Factor (step size)
    KR = float(config.KR)      # Knowledge Ratio (crossover rate)
    Kexp = float(config.Kexp)  # Knowledge Rate exponent

    # Maximum generations (for computing D_junior ratio)
    # G_max = floor(max_nfes / pop_size)
    G_max = max(1, max_nfes // NP)

    # Pre-allocate arrays for junior and senior vectors
    Gained_Shared_Junior = np.empty((NP, D), dtype=np.float64)
    Gained_Shared_Senior = np.empty((NP, D), dtype=np.float64)

    g = 0  # Generation counter

    # ========================================================================
    # Main Evolution Loop
    # ========================================================================
    
    while not budget.exhausted():
        # Check if enough budget remains for one generation
        if budget.remaining() < NP:
            break

        g += 1
        prev_best_f = best_f

        # ====================================================================
        # Step 1: Compute Junior/Senior Dimension Ratio
        # ====================================================================
        # D_junior = ceil(D * (1 - g/G_max)^K)
        #
        # This formula creates a smooth transition:
        # - At g=0: D_junior ≈ D (all exploration)
        # - At g=G_max: D_junior ≈ 0 (all exploitation)
        # - K controls the transition speed (higher K = faster transition)
        
        progress = 1.0 - float(g) / float(G_max)
        progress = max(0.0, progress)  # Clamp to [0, 1]
        
        D_junior = int(np.ceil(D * (progress ** Kexp)))
        D_junior = max(0, min(D_junior, D))  # Clamp to [0, D]
        
        # Probability of using junior phase for each dimension
        p_junior = D_junior / float(D) if D > 0 else 0.0

        # ====================================================================
        # Step 2: Sort Population by Fitness
        # ====================================================================
        # ind_best[0] = index of best individual
        # ind_best[-1] = index of worst individual
        
        pop = popold.copy()
        ind_best = np.argsort(fitness, kind="mergesort")

        # ====================================================================
        # Step 3: Generate Index Sets for Junior and Senior Phases
        # ====================================================================
        # Junior: Rg1, Rg2 = rank neighbors; Rg3 = random
        # Senior: R1 = top 10%, R2 = middle 80%, R3 = bottom 10%
        
        Rg1, Rg2, Rg3 = gained_shared_junior_r1r2r3(ind_best, rng)
        R1, R2, R3 = gained_shared_senior_r1r2r3(ind_best, rng)

        # ====================================================================
        # Step 4: Junior Gaining-Sharing Phase (Exploration)
        # ====================================================================
        # For each individual i, compare with random individual Rg3:
        #
        # If fitness[i] > fitness[Rg3] (i is worse):
        #   x_junior = x_i + KF * (x_Rg1 - x_Rg2 + x_Rg3 - x_i)
        #   → Move toward Rg3 (learn from better random individual)
        #
        # If fitness[i] <= fitness[Rg3] (i is better or equal):
        #   x_junior = x_i + KF * (x_Rg1 - x_Rg2 + x_i - x_Rg3)
        #   → Move away from Rg3 (share knowledge with worse individual)
        
        Gained_Shared_Junior.fill(0.0)
        
        # Case 1: Individual is worse than Rg3 (learn from Rg3)
        worse_mask = fitness > fitness[Rg3]
        if np.any(worse_mask):
            m = worse_mask
            Gained_Shared_Junior[m, :] = (
                pop[m, :]
                + KF * (pop[Rg1[m], :] - pop[Rg2[m], :] + pop[Rg3[m], :] - pop[m, :])
            )

        # Case 2: Individual is better or equal (share with Rg3)
        better_mask = ~worse_mask
        if np.any(better_mask):
            m = better_mask
            Gained_Shared_Junior[m, :] = (
                pop[m, :]
                + KF * (pop[Rg1[m], :] - pop[Rg2[m], :] + pop[m, :] - pop[Rg3[m], :])
            )

        # ====================================================================
        # Step 5: Senior Gaining-Sharing Phase (Exploitation)
        # ====================================================================
        # For each individual i, compare with middle-group individual R2:
        #
        # If fitness[i] > fitness[R2] (i is worse):
        #   x_senior = x_i + KF * (x_R1 - x_i + x_R2 - x_R3)
        #   → Strong guidance from elite (R1), differential from middle/worst
        #
        # If fitness[i] <= fitness[R2] (i is better or equal):
        #   x_senior = x_i + KF * (x_R1 - x_R2 + x_i - x_R3)
        #   → Balanced guidance, less aggressive movement
        
        Gained_Shared_Senior.fill(0.0)
        
        # Case 1: Individual is worse than R2 (strong elite guidance)
        worse_mask2 = fitness > fitness[R2]
        if np.any(worse_mask2):
            m = worse_mask2
            Gained_Shared_Senior[m, :] = (
                pop[m, :]
                + KF * (pop[R1[m], :] - pop[m, :] + pop[R2[m], :] - pop[R3[m], :])
            )

        # Case 2: Individual is better or equal (balanced guidance)
        better_mask2 = ~worse_mask2
        if np.any(better_mask2):
            m = better_mask2
            Gained_Shared_Senior[m, :] = (
                pop[m, :]
                + KF * (pop[R1[m], :] - pop[R2[m], :] + pop[m, :] - pop[R3[m], :])
            )

        # ====================================================================
        # Step 6: Boundary Constraint Handling
        # ====================================================================
        # L-SHADE style midpoint repair:
        # If x < lower: x = (parent + lower) / 2
        # If x > upper: x = (parent + upper) / 2
        
        bound_constraint(Gained_Shared_Junior, pop, lu)
        bound_constraint(Gained_Shared_Senior, pop, lu)

        # ====================================================================
        # Step 7: Dimension Masking (Crossover)
        # ====================================================================
        # For each dimension j of each individual i:
        # 1. Decide junior vs senior: rand <= p_junior → junior, else senior
        # 2. Apply KR gate: only modify if rand <= KR
        #
        # This creates a mix of junior and senior contributions
        # with the ratio shifting from junior→senior over generations
        
        # Junior mask: which dimensions use junior vector
        D_J_rand = rand_matlab(rng, NP, D)
        D_J_mask = D_J_rand <= p_junior
        
        # Senior mask: remaining dimensions
        D_S_mask = ~D_J_mask

        # KR gate: only apply changes with probability KR
        J_gate = rand_matlab(rng, NP, D) <= KR
        S_gate = rand_matlab(rng, NP, D) <= KR
        D_J_mask &= J_gate
        D_S_mask &= S_gate

        # ====================================================================
        # Step 8: Create Offspring
        # ====================================================================
        # Start with parent, selectively replace dimensions
        
        ui = pop.copy()
        ui[D_J_mask] = Gained_Shared_Junior[D_J_mask]
        ui[D_S_mask] = Gained_Shared_Senior[D_S_mask]

        # ====================================================================
        # Step 9: Evaluate Offspring
        # ====================================================================
        
        try:
            child_fit = budget.eval_batch(ui, allow_truncate=False)
        except BudgetExhausted:
            break

        # Update best-so-far
        gen_best_idx = int(np.argmin(child_fit))
        gen_best_val = float(child_fit[gen_best_idx])
        if gen_best_val < best_f:
            best_f = gen_best_val
            best_x = ui[gen_best_idx, :].copy()

        # Track stagnation
        if best_f < prev_best_f:
            stagnation = 0
        else:
            stagnation += 1

        # Update convergence history
        if return_history and history is not None:
            prev = float(history[-1])
            hist_block = np.minimum.accumulate(np.concatenate(([prev], child_fit)))[1:]
            history = np.concatenate([history, hist_block])

        # ====================================================================
        # Step 10: Greedy Selection
        # ====================================================================
        # Replace parent with offspring only if offspring is better
        # This is standard (μ + λ) selection with μ = λ = NP
        
        improved = child_fit < fitness
        if np.any(improved):
            popold[improved, :] = ui[improved, :]
            fitness[improved] = child_fit[improved]

        # ====================================================================
        # Optional: Generation Callback for Monitoring
        # ====================================================================
        
        if generation_callback is not None:
            # Compute diversity as mean standard deviation across dimensions
            diversity = float(np.mean(np.std(popold, axis=0))) if NP > 1 else 0.0
            
            # Determine optimization stage
            frac = budget.nfes_used / max_nfes if max_nfes > 0 else 1.0
            if frac < 1/3:
                stage = "EARLY"
            elif frac < 2/3:
                stage = "MID"
            else:
                stage = "LATE"

            generation_callback(GSKGenerationLog(
                gen=g,
                evals_used=budget.nfes_used,
                best_fitness=best_f,
                diversity=diversity,
                stagnation=stagnation,
                stage=stage,
                jr_mask_rate=float(np.mean(D_J_mask)),
                sr_mask_rate=float(np.mean(D_S_mask)),
                KF=KF, KR=KR, Kexp=Kexp,
            ))

    # ========================================================================
    # Return Result
    # ========================================================================
    
    if budget.exhausted():
        stop_reason = "budget_exhausted"
    elif budget.remaining() < NP:
        stop_reason = "budget_insufficient_for_generation"
    else:
        stop_reason = "completed"

    result = GSKResult(
        best_x=best_x,
        best_f=best_f,
        nfes_used=budget.nfes_used,
        max_nfes=max_nfes,
        stop_reason=stop_reason,
    )

    if return_history:
        return result, history
    return result


def _make_empty_result(
    D: int, budget: BudgetController, max_nfes: int, return_history: bool
) -> Union[GSKResult, Tuple[GSKResult, np.ndarray]]:
    """Create result when optimization fails at initialization."""
    res = GSKResult(
        best_x=np.full(D, np.nan),
        best_f=float("inf"),
        nfes_used=budget.nfes_used,
        max_nfes=max_nfes,
        stop_reason="budget_exhausted_at_init",
    )
    if return_history:
        return res, np.empty((0,), dtype=np.float64)
    return res
