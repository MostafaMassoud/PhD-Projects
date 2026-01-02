"""
Junior Gaining-Sharing Knowledge Operator
=========================================

This module implements the index selection for the Junior (exploration) phase
of the GSK algorithm.

Corresponds to MATLAB file: Gained_Shared_Junior_R1R2R3.m

Algorithm Description
---------------------
The Junior phase models how less experienced individuals learn from their
immediate peers in a ranking-based neighborhood. This promotes exploration
and maintains population diversity.

For each individual i in the population:
1. Find i's rank in the fitness ordering
2. Select knowledge sources based on rank proximity:
   - R1: Better immediate neighbor (rank - 1)
   - R2: Worse immediate neighbor (rank + 1)
   - R3: Random individual (for diversity)

Special Cases:
- Best individual (rank 0): Uses 2nd and 3rd best as R1, R2
- Worst individual (rank N-1): Uses 2nd and 3rd worst as R1, R2

The mutation equation (applied in gsk.py) is:
    If fitness[i] > fitness[R3]:  (i is worse, learns from R3)
        x_new = x_i + KF * (x_R1 - x_R2 + x_R3 - x_i)
    Else:  (i is better, shares with R3)
        x_new = x_i + KF * (x_R1 - x_R2 + x_i - x_R3)

This creates a balanced exploration mechanism where:
- Individuals learn from their immediate rank neighbors (R1, R2)
- Random selection (R3) introduces diversity
- Knowledge flows bidirectionally based on relative fitness
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .rand_matlab import rand_matlab


def gained_shared_junior_r1r2r3(
    ind_best: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate index sets (R1, R2, R3) for the Junior gaining-sharing operator.
    
    Parameters
    ----------
    ind_best : np.ndarray
        Sorted indices array of shape (pop_size,).
        ind_best[0] is the index of the best individual (lowest fitness).
        ind_best[pop_size-1] is the index of the worst individual.
        
    rng : np.random.RandomState
        Random number generator for R3 sampling.
        
    Returns
    -------
    R1 : np.ndarray
        Better neighbor indices, shape (pop_size,).
        R1[i] is the index of i's better rank neighbor.
        
    R2 : np.ndarray
        Worse neighbor indices, shape (pop_size,).
        R2[i] is the index of i's worse rank neighbor.
        
    R3 : np.ndarray
        Random indices, shape (pop_size,).
        R3[i] is a random index ≠ i, R1[i], R2[i].
        
    Notes
    -----
    Index selection rules:
    
    1. For the BEST individual (rank = 0):
       - R1 = 2nd best (rank 1)
       - R2 = 3rd best (rank 2)
       
    2. For the WORST individual (rank = N-1):
       - R1 = 3rd worst (rank N-3)
       - R2 = 2nd worst (rank N-2)
       
    3. For MIDDLE individuals (0 < rank < N-1):
       - R1 = better neighbor (rank - 1)
       - R2 = worse neighbor (rank + 1)
       
    4. R3 is randomly sampled, ensuring R3 ≠ self, R1, R2
    
    Examples
    --------
    >>> ind_best = np.array([5, 2, 8, 1, 0])  # Sorted by fitness
    >>> # Individual 5 is best (rank 0), individual 0 is worst (rank 4)
    >>> R1, R2, R3 = gained_shared_junior_r1r2r3(ind_best, rng)
    >>> # For individual 5 (best): R1=ind_best[1]=2, R2=ind_best[2]=8
    >>> # For individual 0 (worst): R1=ind_best[2]=8, R2=ind_best[3]=1
    """
    ind_best = np.asarray(ind_best, dtype=np.int64)
    pop_size = ind_best.shape[0]
    
    # ========================================================================
    # Step 1: Build rank lookup table
    # ========================================================================
    # rank_of[i] = rank of individual i in sorted order
    # e.g., if ind_best = [5, 2, 8, ...], then rank_of[5] = 0, rank_of[2] = 1
    
    idx = np.arange(pop_size, dtype=np.int64)
    rank_of = np.empty(pop_size, dtype=np.int64)
    rank_of[ind_best] = idx
    
    # ========================================================================
    # Step 2: Allocate output arrays
    # ========================================================================
    
    R1 = np.empty(pop_size, dtype=np.int64)
    R2 = np.empty(pop_size, dtype=np.int64)
    
    # ========================================================================
    # Step 3: Handle special cases (best and worst individuals)
    # ========================================================================
    
    # Masks for different rank positions
    best_mask = (rank_of == 0)                  # Best individual
    worst_mask = (rank_of == pop_size - 1)      # Worst individual
    middle_mask = ~(best_mask | worst_mask)     # All others
    
    # Best individual: use 2nd and 3rd best as neighbors
    # MATLAB: if(ind==1) R1(i)=indBest(2); R2(i)=indBest(3);
    if np.any(best_mask):
        R1[best_mask] = ind_best[1]  # 2nd best
        R2[best_mask] = ind_best[2]  # 3rd best
    
    # Worst individual: use 3rd-worst and 2nd-worst as neighbors
    # MATLAB: elseif(ind==pop_size) R1(i)=indBest(pop_size-2); R2(i)=indBest(pop_size-1);
    if np.any(worst_mask):
        R1[worst_mask] = ind_best[pop_size - 3]  # 3rd worst
        R2[worst_mask] = ind_best[pop_size - 2]  # 2nd worst
    
    # ========================================================================
    # Step 4: Handle middle individuals (majority of population)
    # ========================================================================
    # Use immediate rank neighbors
    # MATLAB: else R1(i)=indBest(ind-1); R2(i)=indBest(ind+1);
    
    if np.any(middle_mask):
        ranks = rank_of[middle_mask]
        R1[middle_mask] = ind_best[ranks - 1]  # Better neighbor (lower rank)
        R2[middle_mask] = ind_best[ranks + 1]  # Worse neighbor (higher rank)
    
    # ========================================================================
    # Step 5: Generate random R3 indices
    # ========================================================================
    # R3 must be different from self (idx), R1, and R2
    # MATLAB: R3 = floor(rand(1, pop_size) * pop_size) + 1;
    
    R3 = np.floor(rand_matlab(rng, pop_size) * pop_size).astype(np.int64)
    
    # Resolve conflicts: regenerate R3 where it equals self, R1, or R2
    # MATLAB: while loop with pos = ((R3 == R2) | (R3 == R1) | (R3 == R0))
    conflict_mask = (R3 == idx) | (R3 == R1) | (R3 == R2)
    
    max_iterations = 1000
    iteration = 0
    while np.any(conflict_mask):
        n_conflicts = int(conflict_mask.sum())
        R3[conflict_mask] = np.floor(
            rand_matlab(rng, n_conflicts) * pop_size
        ).astype(np.int64)
        
        conflict_mask = (R3 == idx) | (R3 == R1) | (R3 == R2)
        iteration += 1
        
        if iteration > max_iterations:
            raise RuntimeError(
                f"Failed to generate conflict-free R3 in {max_iterations} iterations"
            )
    
    return R1, R2, R3
