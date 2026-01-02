"""
MATLAB-Compatible Random Number Generation
==========================================

This module provides random number generation that matches MATLAB's behavior
for reproducibility and validation against reference implementations.

Why MATLAB Compatibility Matters
--------------------------------
When validating a Python implementation against a MATLAB reference:

1. **Identical seeds should produce identical sequences**
   - Both use Mersenne Twister (MT19937)
   - Same seed → same internal state → same random numbers

2. **Matrix layouts differ between languages**
   - MATLAB: Column-major (Fortran) order
   - NumPy: Row-major (C) order by default
   
   This affects how random numbers fill a matrix:
   
   MATLAB rand(2,3) fills:      NumPy default fills:
   
   [r1  r3  r5]                 [r1  r2  r3]
   [r2  r4  r6]                 [r4  r5  r6]
   
   Same 6 numbers, different positions!

The Solution
------------
This module generates random numbers in MATLAB-compatible order:

1. Generate N random values as a flat array
2. Reshape using Fortran (column-major) order
3. Result matches MATLAB's `rand(rows, cols)`

This is CRITICAL for:
- Reproducible experiments across MATLAB/Python
- Validating Python implementation against MATLAB reference
- Debugging discrepancies between implementations

Usage Pattern
-------------
Replace NumPy's `rng.random(shape)` with `rand_matlab(rng, *shape)`:

    # Instead of:
    matrix = rng.random((100, 10))
    
    # Use:
    matrix = rand_matlab(rng, 100, 10)

For single random values or 1D arrays, both behave identically.
"""

from __future__ import annotations

from typing import Union

import numpy as np


def rand_matlab(
    rng: np.random.RandomState,
    *size: int,
) -> Union[float, np.ndarray]:
    """
    Generate random numbers in MATLAB-compatible order.
    
    Parameters
    ----------
    rng : np.random.RandomState
        NumPy random state (initialized with same seed as MATLAB).
        
    *size : int
        Output dimensions. Examples:
        - rand_matlab(rng) → single float
        - rand_matlab(rng, 10) → 1D array of 10 values
        - rand_matlab(rng, 100, 10) → 2D array, 100 rows × 10 cols
        
    Returns
    -------
    float or np.ndarray
        Random values in [0, 1) with MATLAB-compatible ordering.
        
    Notes
    -----
    **Scalar case**: rand_matlab(rng)
        Returns a single float, identical to MATLAB's `rand`.
        
    **1D case**: rand_matlab(rng, n)
        Returns array of n values, identical to MATLAB's `rand(1, n)`.
        No ordering difference for 1D arrays.
        
    **2D+ case**: rand_matlab(rng, m, n, ...)
        Returns array filled in Fortran (column-major) order.
        Matches MATLAB's `rand(m, n, ...)`.
        
    Implementation:
    
        # Generate all values as flat array
        total = m * n * ...
        flat = rng.random(total)
        
        # Reshape with Fortran order (columns first)
        result = flat.reshape(shape, order='F')
    
    Examples
    --------
    >>> rng_py = np.random.RandomState(123)
    >>> rng_matlab = np.random.RandomState(123)  # Same seed
    >>> 
    >>> # MATLAB: rand(2, 3) produces sequence [r1, r2, r3, r4, r5, r6]
    >>> # arranged as [[r1, r3, r5], [r2, r4, r6]]
    >>> 
    >>> # This function reproduces that layout:
    >>> result = rand_matlab(rng_py, 2, 3)
    >>> # result[0,0]=r1, result[1,0]=r2, result[0,1]=r3, ...
    
    Validation Example
    ------------------
    To verify MATLAB compatibility:
    
    MATLAB:
        rng(123);
        X = rand(3, 2);
        disp(X);
        
    Python:
        rng = np.random.RandomState(123)
        X = rand_matlab(rng, 3, 2)
        print(X)
        
    Both should produce identical matrices.
    """
    # ========================================================================
    # Case 1: Scalar (no size arguments)
    # ========================================================================
    if len(size) == 0:
        return float(rng.random())
    
    # ========================================================================
    # Case 2: 1D array (single size argument)
    # ========================================================================
    # No ordering difference for 1D - just return flat array
    if len(size) == 1:
        return rng.random(size[0])
    
    # ========================================================================
    # Case 3: 2D+ array (multiple size arguments)
    # ========================================================================
    # Generate flat, then reshape with Fortran (column-major) order
    
    shape = tuple(size)
    total_elements = int(np.prod(shape))
    
    # Generate all random values as 1D array
    flat = rng.random(total_elements)
    
    # Reshape using Fortran order (fill columns first, like MATLAB)
    # This ensures the random values appear in the same matrix positions
    # as they would in MATLAB
    return flat.reshape(shape, order="F")
