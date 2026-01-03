"""
Boundary Constraint Handling for GSK Algorithm
===============================================

This module implements boundary repair for the GSK algorithm using the
L-SHADE midpoint method.

Corresponds to MATLAB file: boundConstraint.m

Problem Context
---------------
During optimization, mutation operators may generate candidate solutions
that violate the search space bounds. For example, with bounds [-100, 100]:

    x_mutant = 150  (violates upper bound)
    x_mutant = -120 (violates lower bound)

These violations must be repaired before fitness evaluation.

Repair Strategy: L-SHADE Midpoint Method
----------------------------------------
The midpoint method (from L-SHADE algorithm) repairs violations by moving
the violating component halfway back toward the feasible region:

    If x_new < lower:
        x_repaired = (x_parent + lower) / 2
        
    If x_new > upper:
        x_repaired = (x_parent + upper) / 2

Visual representation:
    
    lower                  parent                 upper
      │                      │                      │
      ├──────────────────────┼──────────────────────┤
      │                      │                      │
      ▼                      ▼                      ▼
    -100                    x_p                   +100
      
    Case: x_new < lower (say x_new = -150)
    
           repair point
               ▼
      │────────┼──────────────┼──────────────────────│
    -100   (lower+x_p)/2    x_p                    +100
      
    The repair point is ALWAYS feasible because:
    - lower ≤ x_parent ≤ upper (parent is feasible)
    - (lower + x_parent) / 2 is between lower and x_parent
    - Therefore: lower ≤ repair_point ≤ x_parent ≤ upper

Advantages of Midpoint Method
-----------------------------
1. **Preserves search direction**: Repair point lies between bound and parent,
   maintaining the general direction of mutation
   
2. **Avoids clustering at bounds**: Unlike simple clipping (x = max(lower, x)),
   midpoint method distributes repaired solutions away from boundaries

3. **Reduces bias**: Doesn't artificially concentrate population at bounds

4. **Consistent with L-SHADE**: Proven effective in CEC competitions

Implementation Notes
--------------------
- Operates in-place on the mutation array (vi)
- Requires parent array (pop) for midpoint calculation
- Bounds array lu has shape (2, D) where lu[0] = lower, lu[1] = upper
"""

from __future__ import annotations

import numpy as np


def bound_constraint(
    vi: np.ndarray,
    pop: np.ndarray,
    lu: np.ndarray,
) -> None:
    """
    Apply L-SHADE midpoint boundary repair to mutant vectors.
    
    This function modifies vi IN-PLACE to ensure all values are within bounds.
    
    Parameters
    ----------
    vi : np.ndarray
        Mutant vectors to repair, shape (pop_size, dim).
        Modified in-place.
        
    pop : np.ndarray
        Parent vectors (must be feasible), shape (pop_size, dim).
        Used as reference for midpoint calculation.
        
    lu : np.ndarray
        Bounds array, shape (2, dim).
        lu[0, :] = lower bounds for each dimension
        lu[1, :] = upper bounds for each dimension
        
    Returns
    -------
    None
        vi is modified in-place.
        
    Notes
    -----
    MATLAB equivalent (boundConstraint.m, lines 14-20):
    
        % check the lower hepatitis-c bound
        xl = repmat(lu(1, :), pop_size, 1);
        pos = vi < xl;
        vi(pos) = (pop(pos) + xl(pos)) / 2;
        
        % check the upper bound
        xu = repmat(lu(2, :), pop_size, 1);
        pos = vi > xu;
        vi(pos) = (pop(pos) + xu(pos)) / 2;
    
    Mathematical guarantee:
    - If pop[i,j] ∈ [lower[j], upper[j]] (feasible parent)
    - And vi[i,j] < lower[j]
    - Then (pop[i,j] + lower[j]) / 2 ∈ [lower[j], pop[i,j]] ⊂ [lower[j], upper[j]]
    - Same logic applies for upper bound violations
    
    Examples
    --------
    >>> pop = np.array([[0.0, 50.0]])   # Parent at (0, 50)
    >>> vi = np.array([[-150.0, 120.0]]) # Mutant violates both bounds
    >>> lu = np.array([[-100.0, -100.0], [100.0, 100.0]])  # Bounds
    >>> bound_constraint(vi, pop, lu)
    >>> print(vi)
    [[-50.  75.]]  # Repaired: (-100+0)/2 = -50, (50+100)/2 = 75
    """
    # Extract bounds and expand to match vi shape
    # lu[0, :] = lower bounds (shape: (dim,))
    # lu[1, :] = upper bounds (shape: (dim,))
    # Expand to (pop_size, dim) for element-wise comparison
    # This matches MATLAB: xl = repmat(lu(1, :), pop_size, 1)
    
    pop_size = vi.shape[0]
    lower = np.tile(lu[0, :], (pop_size, 1))  # Shape: (pop_size, dim)
    upper = np.tile(lu[1, :], (pop_size, 1))  # Shape: (pop_size, dim)
    
    # ========================================================================
    # Step 1: Repair lower bound violations
    # ========================================================================
    # Find all positions where vi < lower
    # Replace with midpoint between parent and lower bound
    # MATLAB: pos = vi < xl; vi(pos) = (pop(pos) + xl(pos)) / 2;
    
    lower_violation = vi < lower
    if np.any(lower_violation):
        # Midpoint repair: x_repaired = (x_parent + lower) / 2
        vi[lower_violation] = (pop[lower_violation] + lower[lower_violation]) / 2.0
    
    # ========================================================================
    # Step 2: Repair upper bound violations
    # ========================================================================
    # Find all positions where vi > upper
    # Replace with midpoint between parent and upper bound
    # MATLAB: pos = vi > xu; vi(pos) = (pop(pos) + xu(pos)) / 2;
    
    upper_violation = vi > upper
    if np.any(upper_violation):
        # Midpoint repair: x_repaired = (x_parent + upper) / 2
        vi[upper_violation] = (pop[upper_violation] + upper[upper_violation]) / 2.0
