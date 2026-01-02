from __future__ import annotations

"""gsk_baseline.gsk

Gaining–Sharing Knowledge (GSK) — baseline implementation
========================================================

Overview
--------
The **Gaining–Sharing Knowledge (GSK)** algorithm is a population-based,
continuous metaheuristic for black-box optimization.

GSK models a simplified knowledge transfer process in a group of agents
("individuals"). Each generation, individuals update their positions by
**gaining** knowledge from selected peers and **sharing** it back into their own
decision variables.

A key feature of GSK is that knowledge transfer operates in **two phases**:

1. **Junior gaining–sharing (exploration-oriented):**
   Each individual uses information from its local neighborhood in the
   population's fitness ranking and an additional randomly selected peer.

2. **Senior gaining–sharing (exploitation-oriented):**
   Each individual uses information from three fitness-stratified groups:
   elite (top ~10%), middle (~80%), and worst (bottom ~10%).

For each individual and each dimension, GSK decides whether the update comes
from the **junior** or **senior** operator by sampling a binary mask.
The probability of selecting the junior operator decreases over time, driving a
natural exploration→exploitation transition.

This file intentionally implements **only the baseline algorithm**:

- NO reinforcement learning (RL)
- NO model-based (MB) layer
- NO hybrid components

The surrounding project adds experiment scripts and validation utilities.


Mathematical sketch (per generation)
-----------------------------------
Let:

- population size: ``NP``
- dimension: ``D``
- individuals: ``x_i ∈ ℝ^D`` for ``i = 0..NP-1``
- fitness values: ``f_i = f(x_i)`` (minimization)

The algorithm constructs two trial vectors for each individual:

- ``J_i`` (junior gained–shared vector)
- ``S_i`` (senior gained–shared vector)

and then mixes them using two boolean masks:

- ``M_J[i,j]``: dimension ``j`` updated by junior operator for individual ``i``
- ``M_S[i,j]``: dimension ``j`` updated by senior operator for individual ``i``

Finally, the trial individual is:

``u_i[j] = J_i[j]  if M_J[i,j] is True
          S_i[j]  if M_S[i,j] is True
          x_i[j]  otherwise``

The new population is created by **greedy selection**:

``x_i ← u_i  if f(u_i) < f(x_i)
else keep x_i``

Tie-breaking is deterministic: if equal, the parent survives.


Exploration→exploitation schedule (junior mask probability)
----------------------------------------------------------
GSK gradually reduces the probability of applying the junior operator as the
run progresses.

Let ``g`` be the generation counter (starting at 1) and let
``G_max = floor(max_nfes / NP)`` (consistent with many CEC implementations).

We define the number of *junior dimensions* as:

``D_j(g) = ceil( D * (1 - g / G_max) ^ Kexp )``

and the per-dimension probability of choosing the junior operator as:

``p_j(g) = D_j(g) / D``

As ``g`` increases, ``p_j`` decreases, shifting the search from exploratory
junior learning to exploitative senior learning.


Junior operator (baseline update)
---------------------------------
For each individual ``i`` we select indices ``Rg1, Rg2, Rg3``.
In the baseline implementation these are:

- ``Rg1`` and ``Rg2``: deterministic neighbor ranks around ``i`` in the sorted
  fitness order.
- ``Rg3``: a uniformly random index, re-sampled until it differs from
  ``i, Rg1, Rg2``.

The baseline junior proposal is then:

If ``f_i > f_{Rg3}`` (i is worse than Rg3):

``J_i = x_i + KF * (x_{Rg1} - x_{Rg2} + x_{Rg3} - x_i)``

Else:

``J_i = x_i + KF * (x_{Rg1} - x_{Rg2} + x_i - x_{Rg3})``


Senior operator (baseline update)
---------------------------------
For each individual ``i`` we select indices ``R1, R2, R3`` by sampling from
three rank-based groups:

- top 10% (elite) → R1
- middle 80%      → R2
- bottom 10%      → R3

The baseline senior proposal is:

If ``f_i > f_{R2}`` (i is worse than R2):

``S_i = x_i + KF * (x_{R1} - x_i + x_{R2} - x_{R3})``

Else:

``S_i = x_i + KF * (x_{R1} - x_{R2} + x_i - x_{R3})``


Parameters (baseline defaults)
------------------------------
This implementation exposes the baseline GSK parameters (with common defaults):

- ``KF``: knowledge factor (step scaling), default ``0.5``.
- ``KR``: knowledge ratio (dimension update probability), default ``0.9``.
- ``Kexp``: exponent controlling the decay of junior dimensions over time,
  default ``10``.

These values match the baseline used in the attached reference codebase.

Typical ranges seen in the literature / practice (not prescriptions):

- ``KF`` is usually in ``(0, 1]`` (controls step size).
- ``KR`` is usually in ``(0, 1]`` (controls how many dimensions are updated).
- ``Kexp`` is typically a positive number (often >= 1) controlling how quickly
  the algorithm transitions from junior to senior learning.


Boundary handling
-----------------
CEC-style benchmarks define a hyper-rectangle ``[L, U]^D``.

This implementation uses the standard GSK *midpoint repair* rule:

- if ``v < L`` then set ``v = 0.5 * (x + L)``
- if ``v > U`` then set ``v = 0.5 * (x + U)``

This keeps repaired points inside bounds while retaining some information from
both the parent and the violated bound.


Termination and evaluation budget (NFE)
---------------------------------------
In CEC2017 experiments, the evaluation budget is defined as:

``max_nfes = 10000 * D``

Every objective evaluation counts as **exactly 1 NFE per candidate point**.

**Hard requirement enforced by this package:** no objective evaluation occurs
outside :class:`gsk_baseline.budget.BudgetController`. The controller guarantees
that ``nfes_used`` never exceeds ``max_nfes``.


Implementation notes
--------------------
- RNG: We use :class:`numpy.random.RandomState` and a MATLAB-compatible
  ``rand()`` helper (Fortran-order reshape) because the provided reference
  baseline used this convention.
- Sorting: we use stable mergesort for deterministic tie behavior.
- All arrays are float64 for numerical stability.


References
----------
This module is written to be self-contained and readable without external
resources. For academic background on GSK, consult the original GSK literature
and subsequent benchmarking studies.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .budget import BudgetController, BudgetExhausted


# ---------------------------------------------------------------------------
# RNG helper (MATLAB-compatible rand)
# ---------------------------------------------------------------------------

def rand_matlab(rng: np.random.RandomState, *size: int | Tuple[int, ...]) -> np.ndarray:
    """MATLAB-compatible ``rand(...)`` using NumPy RandomState.

    MATLAB stores arrays in column-major order. A frequent source of
    irreproducibility in ports of legacy MATLAB metaheuristics is that MATLAB's
    ``rand(m,n)`` fills values column-wise, whereas NumPy's default reshape fills
    row-wise (C order).

    To match the reference baseline, this helper draws a flat vector of random
    values and reshapes with **Fortran order**.

    Parameters
    ----------
    rng:
        NumPy RandomState.
    *size:
        Output shape. ``rand_matlab(rng, m, n)`` or ``rand_matlab(rng, (m, n))``.

    Returns
    -------
    np.ndarray
        Random numbers in [0,1) with the requested shape.
    """

    if len(size) == 1 and isinstance(size[0], tuple):
        shape = tuple(int(s) for s in size[0])
    else:
        shape = tuple(int(s) for s in size)

    if len(shape) == 0:
        return float(rng.random_sample())  # type: ignore[return-value]

    if len(shape) == 1:
        return rng.random_sample(shape[0])

    n = int(np.prod(shape))
    flat = rng.random_sample(n)
    return flat.reshape(shape, order="F")


# ---------------------------------------------------------------------------
# Baseline GSK helpers (index selection + bounds)
# ---------------------------------------------------------------------------

def gained_shared_junior_indices(
    ind_best: np.ndarray, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select peer indices for the **junior** gaining-sharing operator.

    Given a ranking of individuals (best→worst), this method selects three
    indices (R1, R2, R3) for each individual i.

    - R1 and R2 are deterministic neighbors in rank-space.
    - R3 is a random index (uniform over population) that is forced to be
      different from i, R1, and R2.

    This matches the reference baseline implementation.

    Parameters
    ----------
    ind_best:
        Array of indices of length NP, sorted by ascending fitness.
    rng:
        RandomState.

    Returns
    -------
    (R1, R2, R3):
        Each is an array of length NP containing indices in ``[0, NP-1]``.
    """

    ind_best = np.asarray(ind_best, dtype=np.int64)
    pop_size = int(ind_best.shape[0])

    idx = np.arange(pop_size, dtype=np.int64)
    rank_of = np.empty(pop_size, dtype=np.int64)
    rank_of[ind_best] = idx

    R1 = np.empty(pop_size, dtype=np.int64)
    R2 = np.empty(pop_size, dtype=np.int64)

    best_mask = rank_of == 0
    worst_mask = rank_of == pop_size - 1
    middle_mask = ~(best_mask | worst_mask)

    # Best individual uses the 2nd and 3rd best as neighbors.
    if np.any(best_mask):
        R1[best_mask] = ind_best[1]
        R2[best_mask] = ind_best[2]

    # Worst individual uses the 3rd-worst and 2nd-worst.
    if np.any(worst_mask):
        R1[worst_mask] = ind_best[pop_size - 3]
        R2[worst_mask] = ind_best[pop_size - 2]

    # Middle individuals use immediate neighbors in rank space.
    if np.any(middle_mask):
        r_mid = rank_of[middle_mask]
        R1[middle_mask] = ind_best[r_mid - 1]
        R2[middle_mask] = ind_best[r_mid + 1]

    # R3: random index, reject conflicts.
    R3 = np.floor(rand_matlab(rng, pop_size) * pop_size).astype(np.int64)
    self_idx = idx

    mask_bad = (R3 == self_idx) | (R3 == R1) | (R3 == R2)
    it = 0
    while np.any(mask_bad):
        n_conflict = int(mask_bad.sum())
        R3[mask_bad] = np.floor(rand_matlab(rng, n_conflict) * pop_size).astype(np.int64)
        mask_bad = (R3 == self_idx) | (R3 == R1) | (R3 == R2)
        it += 1
        if it > 1000:
            raise RuntimeError("Cannot generate conflict-free R3 in 1000 iterations.")

    return R1, R2, R3


def gained_shared_senior_indices(
    ind_best: np.ndarray, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select peer indices for the **senior** gaining-sharing operator.

    The senior operator samples peers from three rank-based groups:

    - R1-group: elite top ~10%
    - R2-group: middle ~80%
    - R3-group: worst bottom ~10%

    For each individual i, we sample one index from each group uniformly.

    Parameters
    ----------
    ind_best:
        Array of indices of length NP, sorted by ascending fitness.
    rng:
        RandomState.

    Returns
    -------
    (R1, R2, R3):
        Each is an array of length NP.
    """

    ind_best = np.asarray(ind_best, dtype=np.int64)
    pop_size = int(ind_best.shape[0])

    i10 = int(round(pop_size * 0.1))
    i90 = int(round(pop_size * 0.9))

    # Guard against tiny populations.
    if i10 < 1:
        i10 = 1
    if i90 < i10 + 1:
        i90 = i10 + 1
    if i90 > pop_size:
        i90 = pop_size

    R1_group = ind_best[0:i10]
    R2_group = ind_best[i10:i90]
    R3_group = ind_best[i90:pop_size]

    # Degenerate guards (should not happen for NP=100 but included for completeness).
    if R2_group.size == 0:
        R2_group = R1_group
    if R3_group.size == 0:
        R3_group = R2_group

    r = rand_matlab(rng, (pop_size, 1))
    R1 = R1_group[np.floor(r * R1_group.shape[0]).astype(np.int64).ravel()]

    r = rand_matlab(rng, (pop_size, 1))
    R2 = R2_group[np.floor(r * R2_group.shape[0]).astype(np.int64).ravel()]

    r = rand_matlab(rng, (pop_size, 1))
    R3 = R3_group[np.floor(r * R3_group.shape[0]).astype(np.int64).ravel()]

    return R1, R2, R3


def bound_constraint(vi: np.ndarray, pop: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Repair candidates that violate box constraints.

    Parameters
    ----------
    vi:
        Candidate population of shape (NP, D).
    pop:
        Parent population of shape (NP, D). Used in midpoint repair.
    bounds:
        Array of shape (2, D): bounds[0] = lower, bounds[1] = upper.

    Returns
    -------
    np.ndarray
        Repaired candidate population (same array object as ``vi``).

    Notes
    -----
    This is the classic GSK midpoint repair:

    - if vi < lower: vi = 0.5*(pop + lower)
    - if vi > upper: vi = 0.5*(pop + upper)

    The operation is vectorized and deterministic.
    """

    vi = np.asarray(vi, dtype=np.float64)
    pop = np.asarray(pop, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)

    xl = bounds[0, :]
    xu = bounds[1, :]

    vi_lower = 0.5 * (pop + xl)
    mask_low = vi < xl
    vi[mask_low] = vi_lower[mask_low]

    vi_upper = 0.5 * (pop + xu)
    mask_high = vi > xu
    vi[mask_high] = vi_upper[mask_high]

    return vi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GSKConfig:
    """Configuration for a single GSK optimization run.

    Parameters
    ----------
    dim:
        Problem dimension D.
    pop_size:
        Population size NP. Baseline commonly uses NP=100 for CEC2017.
    bounds:
        Tuple (lower, upper) specifying a symmetric or asymmetric search box.
        Scalars are broadcast to all dimensions.
    max_nfes:
        Evaluation budget. For CEC2017 this must be ``10000 * D``.
    seed:
        Seed for the RandomState. Determinism is guaranteed for a fixed seed.
    KF:
        Knowledge factor (step scaling). Baseline default: 0.5.
    KR:
        Knowledge ratio (probability of updating a dimension). Baseline default: 0.9.
    Kexp:
        Exponent controlling the decay of junior dimensions. Baseline default: 10.

    Notes
    -----
    Although the baseline uses fixed KF/KR/Kexp, they are exposed here to
    support ablation or sensitivity studies. Changing them will change results.
    """

    dim: int
    pop_size: int = 100
    bounds: Tuple[float, float] = (-100.0, 100.0)
    max_nfes: Optional[int] = None
    seed: int = 123456

    # Baseline parameters
    KF: float = 0.5
    KR: float = 0.9
    Kexp: float = 10.0

    def resolved_max_nfes(self) -> int:
        """Return the resolved evaluation budget.

        If ``max_nfes`` is None, we apply the CEC2017 budget rule:
        ``max_nfes = 10000 * D``.
        """

        if self.max_nfes is None:
            return int(10000 * int(self.dim))
        return int(self.max_nfes)

    def bounds_matrix(self) -> np.ndarray:
        """Return bounds as a (2, D) float64 array."""

        lo, hi = float(self.bounds[0]), float(self.bounds[1])
        D = int(self.dim)
        return np.vstack(
            [
                np.full(D, lo, dtype=np.float64),
                np.full(D, hi, dtype=np.float64),
            ]
        )


@dataclass
class GSKResult:
    """Result of one GSK optimization run."""

    best_x: np.ndarray
    best_f: float
    nfes_used: int
    max_nfes: int
    # Human-readable termination reason (useful for logs/validation).
    # Typical values:
    #   - "budget_exhausted"                : nfes_used == max_nfes
    #   - "budget_exhausted_at_init"       : budget ran out during initialization
    #   - "budget_insufficient_for_step"   : remaining NFEs < population size
    stop_reason: str


def gsk_optimize(
    *,
    objective: Callable[[np.ndarray], np.ndarray],
    config: GSKConfig,
    return_history: bool = False,
) -> GSKResult | tuple[GSKResult, np.ndarray]:
    """Run baseline GSK on a single objective function.

    Parameters
    ----------
    objective:
        Callable objective function ``f(X)`` accepting a 2D array of shape
        ``(n, D)`` and returning a 1D array of shape ``(n,)``.

        **Important:** the objective is called only through the
        :class:`~gsk_baseline.budget.BudgetController` created inside this
        function.

    config:
        :class:`GSKConfig` for this run.
    return_history:
        If True, also return the best-so-far fitness after each evaluated
        candidate (length equals nfes_used).

    Returns
    -------
    GSKResult or (GSKResult, history)
        Best solution found and budget information.

    Budget safety
    -------------
    - The total budget is resolved as ``max_nfes = 10000 * D`` unless overridden.
    - The internal :class:`~gsk_baseline.budget.BudgetController` guarantees
      ``nfes_used <= max_nfes``.

    Determinism
    -----------
    The full run is deterministic given the same:

    - objective implementation,
    - configuration,
    - and random seed.
    """

    D = int(config.dim)
    NP = int(config.pop_size)
    if D <= 0:
        raise ValueError("dim must be positive")
    if NP < 4:
        raise ValueError("pop_size must be at least 4 (required by index logic)")

    max_nfes = int(config.resolved_max_nfes())

    rng = np.random.RandomState(int(config.seed))

    bounds = config.bounds_matrix()
    lower = bounds[0, :]
    upper = bounds[1, :]

    # Initialize budget controller (sole entry point to objective evaluations).
    budget = BudgetController(max_nfes=max_nfes, objective=objective)

    # -------------------------
    # 1) Population initialization
    # -------------------------
    popold = lower + rand_matlab(rng, NP, D) * (upper - lower)

    # Initial evaluation (counts NP NFEs, unless the budget is smaller).
    try:
        fitness = budget.eval_batch(popold, allow_truncate=True)
    except BudgetExhausted:
        # No evaluations possible.
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

    # If truncated (should not happen under CEC2017 budgets), keep only evaluated prefix.
    n_init = int(fitness.shape[0])
    if n_init <= 0:
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

    # Best-so-far (BSF)
    best_idx = int(np.argmin(fitness))
    best_f = float(fitness[best_idx])
    best_x = popold[best_idx, :].copy()

    if return_history:
        history = np.minimum.accumulate(fitness.copy())
    else:
        history = None

    # -------------------------
    # 2) Main generational loop
    # -------------------------

    KF = float(config.KF)
    KR = float(config.KR)
    Kexp = float(config.Kexp)

    # For the classic CEC budget and NP=100, G_max = 100*D.
    G_max = int(max_nfes // NP)
    if G_max <= 0:
        G_max = 1

    g = 0

    # Use buffers to avoid repeated allocations.
    Gained_Shared_Junior = np.empty((NP, D), dtype=np.float64)
    Gained_Shared_Senior = np.empty((NP, D), dtype=np.float64)

    # Important: the loop guard is based on the evaluation budget, not on g.
    while not budget.exhausted():
        # Preferred safety rule: do not enter a generation if we cannot evaluate
        # the full offspring population.
        if budget.remaining() < NP:
            break

        g += 1

        # Probability of junior dimensions.
        base = 1.0 - float(g) / float(G_max)
        if base < 0.0:
            base = 0.0
        D_j = int(np.ceil(D * (base ** Kexp)))
        D_j = max(0, min(D_j, D))
        p_j = (D_j / float(D)) if D > 0 else 0.0

        pop = popold.copy()
        ind_best = np.argsort(fitness, kind="mergesort")

        # 2.1) Indices for junior/senior operators
        Rg1, Rg2, Rg3 = gained_shared_junior_indices(ind_best, rng)
        R1, R2, R3 = gained_shared_senior_indices(ind_best, rng)

        # 2.2) Junior phase trial vectors
        Gained_Shared_Junior.fill(0.0)
        worse_than_Rg3 = fitness > fitness[Rg3]
        if np.any(worse_than_Rg3):
            m = worse_than_Rg3
            Gained_Shared_Junior[m, :] = (
                pop[m, :]
                + KF * (pop[Rg1[m], :] - pop[Rg2[m], :] + pop[Rg3[m], :] - pop[m, :])
            )
        better_or_equal = ~worse_than_Rg3
        if np.any(better_or_equal):
            m = better_or_equal
            Gained_Shared_Junior[m, :] = (
                pop[m, :]
                + KF * (pop[Rg1[m], :] - pop[Rg2[m], :] + pop[m, :] - pop[Rg3[m], :])
            )

        # 2.3) Senior phase trial vectors
        Gained_Shared_Senior.fill(0.0)
        worse_than_R2 = fitness > fitness[R2]
        if np.any(worse_than_R2):
            m = worse_than_R2
            Gained_Shared_Senior[m, :] = (
                pop[m, :]
                + KF * (pop[R1[m], :] - pop[m, :] + pop[R2[m], :] - pop[R3[m], :])
            )
        better_or_equal2 = ~worse_than_R2
        if np.any(better_or_equal2):
            m = better_or_equal2
            Gained_Shared_Senior[m, :] = (
                pop[m, :]
                + KF * (pop[R1[m], :] - pop[R2[m], :] + pop[m, :] - pop[R3[m], :])
            )

        # 2.4) Boundary repair
        bound_constraint(Gained_Shared_Junior, pop, bounds)
        bound_constraint(Gained_Shared_Senior, pop, bounds)

        # 2.5) Dimension masks (junior vs senior) + KR gates
        D_J_rand = rand_matlab(rng, NP, D)
        D_J_mask = D_J_rand <= p_j
        D_S_mask = ~D_J_mask

        # KR gates are sampled independently for junior and senior masks
        J_gate = rand_matlab(rng, NP, D) <= KR
        D_J_mask &= J_gate
        S_gate = rand_matlab(rng, NP, D) <= KR
        D_S_mask &= S_gate

        # 2.6) Build offspring population
        ui = pop.copy()
        ui[D_J_mask] = Gained_Shared_Junior[D_J_mask]
        ui[D_S_mask] = Gained_Shared_Senior[D_S_mask]

        # 2.7) Evaluate offspring (exactly NP evaluations; guarded by remaining())
        try:
            child_fit = budget.eval_batch(ui, allow_truncate=False)
        except BudgetExhausted:
            break

        # 2.8) Best-so-far update
        gen_best_idx = int(np.argmin(child_fit))
        gen_best_val = float(child_fit[gen_best_idx])
        if gen_best_val < best_f:
            best_f = gen_best_val
            best_x = ui[gen_best_idx, :].copy()

        if return_history and history is not None:
            # Append best-so-far after each evaluated child.
            prev = float(history[-1])
            hist_block = np.minimum.accumulate(np.concatenate(([prev], child_fit)))[1:]
            history = np.concatenate([history, hist_block])

        # 2.9) Greedy selection (minimization). If equal, parent survives.
        improved = child_fit < fitness
        if np.any(improved):
            popold[improved, :] = ui[improved, :]
            fitness[improved] = child_fit[improved]

    if budget.exhausted():
        stop_reason = "budget_exhausted"
    elif budget.remaining() < NP:
        stop_reason = "budget_insufficient_for_step"
    else:
        # Fallback: should not normally occur under the standard CEC2017 budget.
        stop_reason = "stopped"

    result = GSKResult(
        best_x=best_x,
        best_f=best_f,
        nfes_used=budget.nfes_used,
        max_nfes=max_nfes,
        stop_reason=stop_reason,
    )

    if return_history:
        assert history is not None
        return result, history
    return result
