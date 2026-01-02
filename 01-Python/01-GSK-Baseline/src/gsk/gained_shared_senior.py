"""
Senior Gaining-Sharing Knowledge Operator
=========================================

This module implements the index selection for the Senior (exploitation) phase
of the GSK algorithm.

Corresponds to MATLAB file: Gained_Shared_Senior_R1R2R3.m

Algorithm Description
---------------------
The Senior phase models how experienced individuals learn from stratified
knowledge groups. This promotes exploitation of promising regions by
leveraging information from different quality levels of the population.

The population is divided into three groups based on fitness ranking:

    ┌─────────────────────────────────────────────────────────────┐
    │  BEST 10%   │      MIDDLE 80%       │     WORST 10%         │
    │  (Elite)    │   (Average performers) │  (Poor performers)   │
    │             │                        │                       │
    │  R1 source  │      R2 source         │     R3 source        │
    └─────────────────────────────────────────────────────────────┘
    
For each individual i:
- R1: Random individual from top 10% (elite guidance)
- R2: Random individual from middle 80% (baseline reference)
- R3: Random individual from bottom 10% (diversity/contrast)

The mutation equation (applied in gsk.py) is:
    If fitness[i] > fitness[R2]:  (i is worse than middle, needs help)
        x_new = x_i + KF * (x_R1 - x_i + x_R2 - x_R3)
        → Strong pull toward elite (R1), away from worst (R3)
        
    Else:  (i is competitive with middle)
        x_new = x_i + KF * (x_R1 - x_R2 + x_i - x_R3)
        → Balanced exploration using group differentials

This creates a directed exploitation mechanism where:
- Elite individuals (R1) provide target directions
- Middle individuals (R2) serve as reference points
- Worst individuals (R3) indicate regions to avoid
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .rand_matlab import rand_matlab


def gained_shared_senior_r1r2r3(
    ind_best: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate index sets (R1, R2, R3) for the Senior gaining-sharing operator.
    
    Parameters
    ----------
    ind_best : np.ndarray
        Sorted indices array of shape (pop_size,).
        ind_best[0] is the index of the best individual (lowest fitness).
        ind_best[pop_size-1] is the index of the worst individual.
        
    rng : np.random.RandomState
        Random number generator for stratified sampling.
        
    Returns
    -------
    R1 : np.ndarray
        Elite group indices, shape (pop_size,).
        Each R1[i] is randomly sampled from top 10% of population.
        
    R2 : np.ndarray
        Middle group indices, shape (pop_size,).
        Each R2[i] is randomly sampled from middle 80% of population.
        
    R3 : np.ndarray
        Worst group indices, shape (pop_size,).
        Each R3[i] is randomly sampled from bottom 10% of population.
        
    Notes
    -----
    Group boundaries (for pop_size = 100):
    
    - Best 10%:   ranks [0, 9]     → indices ind_best[0:10]
    - Middle 80%: ranks [10, 89]   → indices ind_best[10:90]
    - Worst 10%:  ranks [90, 99]   → indices ind_best[90:100]
    
    Sampling method:
    - MATLAB: ceil(length * rand) gives 1-based index in [1, length]
    - Python: floor(length * rand) gives 0-based index in [0, length-1]
    - Both produce valid uniform sampling in the correct range
    
    Examples
    --------
    >>> pop_size = 100
    >>> # Best 10% = ind_best[0:10], Middle 80% = ind_best[10:90], Worst 10% = ind_best[90:100]
    >>> R1, R2, R3 = gained_shared_senior_r1r2r3(ind_best, rng)
    >>> # R1[i] ∈ {ind_best[0], ..., ind_best[9]}   (elite)
    >>> # R2[i] ∈ {ind_best[10], ..., ind_best[89]} (middle)
    >>> # R3[i] ∈ {ind_best[90], ..., ind_best[99]} (worst)
    """
    ind_best = np.asarray(ind_best, dtype=np.int64)
    pop_size = ind_best.shape[0]
    
    # ========================================================================
    # Step 1: Calculate group boundaries
    # ========================================================================
    # Using 10%-80%-10% split as in the original GSK paper
    #
    # MATLAB code:
    #   Kind_best = ceil(0.1 * pop_size);    % Top 10%
    #   Kind_middle = ceil(0.8 * pop_size);  % Middle 80%
    #   Kind_worst = ceil(0.1 * pop_size);   % Bottom 10%
    
    kind_best = int(np.ceil(0.1 * pop_size))    # Size of elite group
    kind_middle = int(np.ceil(0.8 * pop_size))  # Size of middle group
    kind_worst = int(np.ceil(0.1 * pop_size))   # Size of worst group
    
    # Ensure minimum group sizes (at least 1 individual per group)
    kind_best = max(1, kind_best)
    kind_middle = max(1, kind_middle)
    kind_worst = max(1, kind_worst)
    
    # ========================================================================
    # Step 2: Extract group member indices
    # ========================================================================
    # ind_best is sorted: [best, 2nd best, ..., worst]
    #
    # Best group:   ind_best[0 : kind_best]
    # Middle group: ind_best[kind_best : kind_best + kind_middle]
    # Worst group:  ind_best[pop_size - kind_worst : pop_size]
    
    best_group = ind_best[:kind_best]
    middle_group = ind_best[kind_best : kind_best + kind_middle]
    worst_group = ind_best[pop_size - kind_worst:]
    
    # ========================================================================
    # Step 3: Random sampling from each group
    # ========================================================================
    # For each individual, randomly select one member from each group
    #
    # MATLAB: R1(i) = Best(ceil(Kind_best * rand));
    #         R2(i) = Middle(ceil(Kind_middle * rand));
    #         R3(i) = Worst(ceil(Kind_worst * rand));
    #
    # Python equivalent using floor for 0-based indexing:
    #         R1[i] = best_group[floor(kind_best * rand)]
    
    # Generate random indices into each group
    r1_idx = np.floor(rand_matlab(rng, pop_size) * kind_best).astype(np.int64)
    r2_idx = np.floor(rand_matlab(rng, pop_size) * kind_middle).astype(np.int64)
    r3_idx = np.floor(rand_matlab(rng, pop_size) * kind_worst).astype(np.int64)
    
    # Clamp indices to valid range (safety check)
    r1_idx = np.clip(r1_idx, 0, kind_best - 1)
    r2_idx = np.clip(r2_idx, 0, kind_middle - 1)
    r3_idx = np.clip(r3_idx, 0, kind_worst - 1)
    
    # Map group-relative indices to actual population indices
    R1 = best_group[r1_idx]
    R2 = middle_group[r2_idx]
    R3 = worst_group[r3_idx]
    
    return R1, R2, R3
