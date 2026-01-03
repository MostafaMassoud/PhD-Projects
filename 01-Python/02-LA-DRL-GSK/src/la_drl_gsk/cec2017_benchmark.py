"""
CEC2017 Benchmark Suite Interface
=================================

This module provides an interface to the CEC2017 benchmark functions
for testing LA-DRL-GSK algorithm.

CEC2017 Benchmark Overview:
- 30 scalable test functions (F2 excluded due to instability)
- Categories: Unimodal (F1,F3), Multimodal (F4-F10), Hybrid (F11-F20), Composition (F21-F30)
- Search space: [-100, +100]^D
- Optimal value: f_opt = 100 × function_id

Reference:
    Awad, N. H., et al. (2016). Problem definitions and evaluation criteria 
    for the CEC 2017 special session and competition on single objective 
    real-parameter numerical optimization. Technical Report.

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import numpy as np
import sys
import importlib
from pathlib import Path
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

# Standard CEC2017 function IDs (F2 excluded)
CEC2017_FUNCTIONS = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# Function categories
UNIMODAL = [1, 3]
SIMPLE_MULTIMODAL = [4, 5, 6, 7, 8, 9, 10]
HYBRID = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
COMPOSITION = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# Standard dimensions
DIMENSIONS = [10, 30, 50, 100]

# NFE budgets per CEC2017 rules
def get_max_nfes(dim: int) -> int:
    """Get standard NFE budget for dimension."""
    return 10000 * dim


@dataclass
class BenchmarkFunction:
    """Container for benchmark function information."""
    func_id: int
    dim: int
    bounds: Tuple[float, float]
    f_opt: float
    category: str
    name: str
    
    def get_max_nfes(self) -> int:
        return get_max_nfes(self.dim)


# =============================================================================
# CEC2017 Function Loader
# =============================================================================

_cec2017_module = None
_cec2017_functions = None


def _ensure_cec2017_loaded(cec_path: Optional[str] = None) -> None:
    """
    Ensure CEC2017 functions are loaded.
    
    Searches for CEC2017 implementation in standard locations:
    1. Provided cec_path
    2. ../cec2017/ relative to this file
    3. Environment variable CEC2017_PATH
    """
    global _cec2017_module, _cec2017_functions
    
    if _cec2017_functions is not None:
        return
    
    # Candidate paths
    search_paths = []
    
    if cec_path:
        search_paths.append(Path(cec_path))
    
    # Relative to this file
    this_dir = Path(__file__).parent
    search_paths.extend([
        this_dir / '..' / '..' / 'cec2017',
        this_dir / '..' / '..' / '..' / 'cec2017',
        this_dir / '..' / '..' / '..' / '00-CEC2017',
        this_dir / '..' / '..' / '..' / '00-CEC-Root' / 'cec2017',
    ])
    
    # Try each path
    for path in search_paths:
        path = path.resolve()
        
        # Check for different layouts
        candidates = [
            (path / 'functions.py', path),
            (path / 'cec2017' / 'functions.py', path),
        ]
        
        for func_file, sys_path in candidates:
            if func_file.exists():
                if str(sys_path) not in sys.path:
                    sys.path.insert(0, str(sys_path))
                try:
                    if 'cec2017' in str(func_file):
                        mod = importlib.import_module('cec2017.functions')
                    else:
                        mod = importlib.import_module('functions')
                    
                    _cec2017_module = mod
                    _cec2017_functions = getattr(mod, 'all_functions', None)
                    
                    if _cec2017_functions is not None:
                        return
                except ImportError:
                    continue
    
    # If still not loaded, create synthetic functions for testing
    print("WARNING: CEC2017 functions not found. Using synthetic test functions.")
    _cec2017_functions = _create_synthetic_functions()


def _create_synthetic_functions() -> List[Callable]:
    """Create synthetic test functions when CEC2017 not available."""
    functions = []
    
    for i in range(30):
        f_opt = (i + 1) * 100
        
        if i < 3:  # Unimodal (sphere-like)
            functions.append(_make_sphere(f_opt))
        elif i < 10:  # Multimodal (rastrigin-like)
            functions.append(_make_rastrigin(f_opt))
        else:  # Hybrid/Composition (simplified)
            functions.append(_make_hybrid(f_opt))
    
    return functions


def _make_sphere(offset: float):
    """Create sphere function with offset."""
    def f(x):
        x = np.atleast_2d(x)
        return np.sum(x ** 2, axis=1) + offset
    return f


def _make_rastrigin(offset: float):
    """Create Rastrigin function with offset."""
    def f(x):
        x = np.atleast_2d(x)
        d = x.shape[1]
        return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1) + offset
    return f


def _make_hybrid(offset: float):
    """Create hybrid function with offset."""
    def f(x):
        x = np.atleast_2d(x)
        d = x.shape[1]
        return (np.sum(x ** 2, axis=1) + 
                np.sum(np.abs(x), axis=1) + 
                np.prod(np.abs(x) + 1e-10, axis=1) ** (1/d)) + offset
    return f


def get_cec2017_function(
    func_id: int,
    dim: int,
    cec_path: Optional[str] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], float]:
    """
    Get a CEC2017 benchmark function.
    
    Parameters
    ----------
    func_id : int
        Function ID in [1, 30] (F2 excluded)
    dim : int
        Problem dimension
    cec_path : str, optional
        Path to CEC2017 implementation
        
    Returns
    -------
    objective : callable
        Vectorized objective function f(X) -> y
    f_opt : float
        Known optimal value (100 * func_id)
    """
    _ensure_cec2017_loaded(cec_path)
    
    if not (1 <= func_id <= 30):
        raise ValueError(f"func_id must be in [1, 30], got {func_id}")
    
    if func_id == 2:
        raise ValueError("F2 is excluded from CEC2017 benchmarks due to instability")
    
    # Get function (0-indexed)
    f = _cec2017_functions[func_id - 1]
    f_opt = 100.0 * func_id
    
    def objective(X: np.ndarray) -> np.ndarray:
        """Vectorized objective wrapper."""
        X = np.atleast_2d(X).astype(np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        
        n = X.shape[0]
        
        # Try batch evaluation
        try:
            y = f(X)
            y = np.asarray(y, dtype=np.float64).reshape(-1)
            if y.shape[0] == n:
                return y
        except Exception:
            pass
        
        # Fallback: row by row
        y = np.empty(n, dtype=np.float64)
        for i in range(n):
            yi = f(X[i:i+1, :])
            y[i] = float(np.asarray(yi).flat[0])
        return y
    
    return objective, f_opt


def get_benchmark_info(func_id: int, dim: int) -> BenchmarkFunction:
    """Get benchmark function metadata."""
    if func_id in UNIMODAL:
        category = 'unimodal'
    elif func_id in SIMPLE_MULTIMODAL:
        category = 'multimodal'
    elif func_id in HYBRID:
        category = 'hybrid'
    else:
        category = 'composition'
    
    names = {
        1: 'Shifted and Rotated Bent Cigar',
        3: 'Shifted and Rotated Zakharov',
        4: 'Shifted and Rotated Rosenbrock',
        5: 'Shifted and Rotated Rastrigin',
        6: 'Shifted and Rotated Expanded Scaffer F6',
        7: 'Shifted and Rotated Lunacek Bi-Rastrigin',
        8: 'Shifted and Rotated Non-Continuous Rastrigin',
        9: 'Shifted and Rotated Levy',
        10: 'Shifted and Rotated Schwefel',
    }
    name = names.get(func_id, f'CEC2017 F{func_id}')
    
    return BenchmarkFunction(
        func_id=func_id,
        dim=dim,
        bounds=(-100.0, 100.0),
        f_opt=100.0 * func_id,
        category=category,
        name=name,
    )


# =============================================================================
# Benchmark Suite for Training/Testing
# =============================================================================

class BenchmarkSuite:
    """
    Collection of benchmark functions for training and evaluation.
    
    Parameters
    ----------
    func_ids : list of int, optional
        Function IDs to include (default: all CEC2017)
    dims : list of int, optional
        Dimensions to include (default: [10, 30])
    cec_path : str, optional
        Path to CEC2017 implementation
    """
    
    def __init__(
        self,
        func_ids: Optional[List[int]] = None,
        dims: Optional[List[int]] = None,
        cec_path: Optional[str] = None,
    ):
        self.func_ids = func_ids or CEC2017_FUNCTIONS
        self.dims = dims or [10, 30]
        self.cec_path = cec_path
        
        # Build function list
        self.functions: List[Tuple[int, int, Callable, float]] = []
        for fid in self.func_ids:
            for dim in self.dims:
                try:
                    obj, f_opt = get_cec2017_function(fid, dim, cec_path)
                    self.functions.append((fid, dim, obj, f_opt))
                except ValueError:
                    continue
    
    def __len__(self) -> int:
        return len(self.functions)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, Callable, float]:
        return self.functions[idx]
    
    def sample(self, rng: Optional[np.random.RandomState] = None) -> Tuple[int, int, Callable, float]:
        """Sample a random function from the suite."""
        if rng is None:
            rng = np.random.RandomState()
        idx = rng.randint(len(self.functions))
        return self.functions[idx]
    
    def get_training_suite(self, train_ratio: float = 0.7) -> 'BenchmarkSuite':
        """Get subset for training."""
        n_train = int(len(self.func_ids) * train_ratio)
        train_ids = self.func_ids[:n_train]
        return BenchmarkSuite(func_ids=train_ids, dims=self.dims, cec_path=self.cec_path)
    
    def get_test_suite(self, train_ratio: float = 0.7) -> 'BenchmarkSuite':
        """Get subset for testing."""
        n_train = int(len(self.func_ids) * train_ratio)
        test_ids = self.func_ids[n_train:]
        return BenchmarkSuite(func_ids=test_ids, dims=self.dims, cec_path=self.cec_path)


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_on_function(
    optimizer,
    func_id: int,
    dim: int,
    n_runs: int = 51,
    seed_base: int = 0,
    cec_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Evaluate optimizer on a single CEC2017 function.
    
    Parameters
    ----------
    optimizer : LADRLGSK or callable
        Optimizer instance with optimize(objective) method
    func_id : int
        CEC2017 function ID
    dim : int
        Problem dimension
    n_runs : int
        Number of independent runs
    seed_base : int
        Base seed for runs (run i uses seed_base + i)
    cec_path : str, optional
        Path to CEC2017 implementation
    verbose : bool
        Print progress
        
    Returns
    -------
    dict with keys: errors, best, worst, mean, median, std
    """
    objective, f_opt = get_cec2017_function(func_id, dim, cec_path)
    
    errors = []
    
    for run in range(n_runs):
        seed = seed_base + run
        
        # Create new optimizer instance with seed
        from .la_drl_gsk import LADRLGSKConfig, LADRLGSK
        config = LADRLGSKConfig(
            dim=dim,
            seed=seed,
            use_rl=optimizer.config.use_rl if hasattr(optimizer, 'config') else True,
            ablation_mode=optimizer.config.ablation_mode if hasattr(optimizer, 'config') else None,
        )
        opt = LADRLGSK(config, policy=optimizer.policy if hasattr(optimizer, 'policy') else None)
        
        result = opt.optimize(objective)
        error = abs(result.best_f - f_opt)
        errors.append(error)
        
        if verbose:
            print(f"  Run {run+1:02d}/{n_runs}: error={error:.6e}")
    
    errors = np.array(errors)
    
    return {
        'func_id': func_id,
        'dim': dim,
        'n_runs': n_runs,
        'errors': errors,
        'best': float(np.min(errors)),
        'worst': float(np.max(errors)),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
    }
