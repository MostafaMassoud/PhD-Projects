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
import importlib.util
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Any
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

# Module cache: stores (module, source_type) where source_type is 'bundled', 'external', or 'synthetic'
_cec2017_module = None
_cec2017_loaded = False
_cec2017_source = None  # Track where module came from


def _load_module_from_path(module_path: Path, module_name: str = 'cec2017_functions'):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def _try_load_external_cec(cec_path: str) -> Optional[Any]:
    """
    Try to load CEC2017 from an external path.
    
    Supports layouts:
    A) <cec_root>/cec2017/functions.py  (package cec2017)
    B) <cec_root>/cec2017/cec2017/functions.py (nested package)
    C) <cec_root>/functions.py (flat module)
    
    Returns the module if successful, None otherwise.
    """
    cec_p = Path(cec_path).resolve()
    
    # List of (functions_file, parent_to_add_to_sys_path) tuples to try
    candidates = []
    
    if cec_p.is_dir():
        # Layout A: <cec_root>/cec2017/functions.py
        candidates.append((cec_p / 'cec2017' / 'functions.py', cec_p))
        # Layout B: <cec_root>/cec2017/cec2017/functions.py  
        candidates.append((cec_p / 'cec2017' / 'cec2017' / 'functions.py', cec_p / 'cec2017'))
        # Layout C: <cec_root>/functions.py
        candidates.append((cec_p / 'functions.py', cec_p))
    else:
        # Direct file path
        candidates.append((cec_p, cec_p.parent))
    
    for func_file, sys_path_entry in candidates:
        if not func_file.exists():
            continue
        
        try:
            # Add parent to sys.path temporarily for relative imports to work
            sys_path_entry_str = str(sys_path_entry)
            added_to_path = False
            if sys_path_entry_str not in sys.path:
                sys.path.insert(0, sys_path_entry_str)
                added_to_path = True
            
            try:
                # Try importing as a package first
                if func_file.parent.name == 'cec2017':
                    try:
                        # Clear any cached version
                        if 'cec2017' in sys.modules:
                            del sys.modules['cec2017']
                        if 'cec2017.functions' in sys.modules:
                            del sys.modules['cec2017.functions']
                        
                        # Try package import
                        mod = importlib.import_module('cec2017.functions')
                        if _validate_cec_module(mod):
                            return mod
                    except Exception:
                        pass
                
                # Fallback to direct file loading
                mod = _load_module_from_path(func_file, f'cec2017_ext_{hash(str(func_file))}')
                if mod is not None and _validate_cec_module(mod):
                    return mod
                    
            finally:
                if added_to_path and sys_path_entry_str in sys.path:
                    sys.path.remove(sys_path_entry_str)
                    
        except Exception:
            continue
    
    return None


def _try_load_bundled_cec() -> Optional[Any]:
    """
    Try to load the bundled CEC2017 implementation.
    
    Location: src/cec2017/functions.py (relative to this file)
    """
    this_dir = Path(__file__).parent.resolve()
    bundled_path = this_dir.parent / 'cec2017' / 'functions.py'
    
    if not bundled_path.exists():
        return None
    
    try:
        mod = _load_module_from_path(bundled_path, 'cec2017_bundled')
        if mod is not None and _validate_cec_module(mod):
            return mod
    except Exception:
        pass
    
    return None


def _validate_cec_module(mod: Any) -> bool:
    """Check if a module has valid CEC2017 interface."""
    # Must have either get_cec2017_function or all_functions
    has_getter = hasattr(mod, 'get_cec2017_function')
    has_all = hasattr(mod, 'all_functions')
    has_test = hasattr(mod, 'cec2017_test_func') or hasattr(mod, 'cec2017_func')
    
    return has_getter or has_all or has_test


def _ensure_cec2017_loaded(cec_path: Optional[str] = None) -> None:
    """
    Ensure CEC2017 functions are loaded.
    
    Load priority:
    1. External path (if cec_path provided)
    2. Bundled implementation (src/cec2017/functions.py)
    3. Synthetic fallback (only if both fail)
    
    Note: Warning is only printed if BOTH external and bundled fail.
    """
    global _cec2017_module, _cec2017_loaded, _cec2017_source
    
    if _cec2017_loaded:
        return
    
    # 1. Try external path if provided
    if cec_path:
        mod = _try_load_external_cec(cec_path)
        if mod is not None:
            _cec2017_module = mod
            _cec2017_loaded = True
            _cec2017_source = 'external'
            return
    
    # 2. Try bundled implementation
    mod = _try_load_bundled_cec()
    if mod is not None:
        _cec2017_module = mod
        _cec2017_loaded = True
        _cec2017_source = 'bundled'
        return
    
    # 3. Fallback to synthetic - only warn here when both fail
    print("WARNING: CEC2017 functions not found. Using synthetic test functions.")
    _cec2017_module = None
    _cec2017_loaded = True
    _cec2017_source = 'synthetic'


def _create_synthetic_function(func_id: int, dim: int) -> Tuple[Callable, float]:
    """Create a synthetic test function when CEC2017 not available."""
    f_opt = func_id * 100.0
    
    # Create random shift for this function
    rng = np.random.RandomState(func_id * 1000 + dim)
    shift = rng.uniform(-50, 50, dim)
    
    if func_id <= 3:  # Unimodal (sphere-like)
        def objective(x):
            x = np.atleast_2d(x).astype(np.float64)
            z = x - shift
            return np.sum(z ** 2, axis=1) + f_opt
            
    elif func_id <= 10:  # Multimodal (rastrigin-like)
        def objective(x):
            x = np.atleast_2d(x).astype(np.float64)
            z = x - shift
            return 10 * dim + np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z), axis=1) + f_opt
            
    elif func_id <= 20:  # Hybrid (griewank-like)
        def objective(x):
            x = np.atleast_2d(x).astype(np.float64)
            z = x - shift
            idx = np.arange(1, dim + 1)
            sum_sq = np.sum(z ** 2, axis=1) / 4000
            prod_cos = np.prod(np.cos(z / np.sqrt(idx)), axis=1)
            return sum_sq - prod_cos + 1 + f_opt
            
    else:  # Composition (ackley-like)
        def objective(x):
            x = np.atleast_2d(x).astype(np.float64)
            z = x - shift
            t1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(z ** 2, axis=1)))
            t2 = -np.exp(np.mean(np.cos(2 * np.pi * z), axis=1))
            return t1 + t2 + 20 + np.e + f_opt
    
    return objective, f_opt


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
        Vectorized objective function f(X) -> y where X is (N, D)
    f_opt : float
        Known optimal value (100 * func_id)
    """
    # Reset and reload if cec_path is provided and different from current source
    global _cec2017_loaded
    if cec_path is not None:
        # Force reload to try the new path
        reset_cec2017_cache()
    
    _ensure_cec2017_loaded(cec_path)
    
    if not (1 <= func_id <= 30):
        raise ValueError(f"func_id must be in [1, 30], got {func_id}")
    
    if func_id == 2:
        raise ValueError("F2 is excluded from CEC2017 benchmarks due to instability")
    
    f_opt = 100.0 * func_id
    
    # No module available - use synthetic
    if _cec2017_module is None:
        return _create_synthetic_function(func_id, dim)
    
    # Try module's get_cec2017_function first (bundled implementation has this)
    if hasattr(_cec2017_module, 'get_cec2017_function'):
        try:
            func_obj, mod_f_opt = _cec2017_module.get_cec2017_function(func_id, dim)
            return _wrap_objective(func_obj, dim), mod_f_opt
        except Exception:
            pass
    
    # Try all_functions attribute
    if hasattr(_cec2017_module, 'all_functions'):
        all_funcs = _cec2017_module.all_functions
        
        # Check if it's callable (lambda returning list) or a list
        if callable(all_funcs):
            try:
                # Bundled implementation: lambda dim -> list of funcs (excludes F2)
                funcs = all_funcs(dim)
                if len(funcs) == 29:
                    # F2 is excluded, need to map func_id to index
                    # func_id 1 -> index 0
                    # func_id 3 -> index 1, func_id 4 -> index 2, etc.
                    if func_id == 1:
                        idx = 0
                    elif func_id > 2:
                        idx = func_id - 2  # 3->1, 4->2, ..., 30->28
                    else:
                        raise ValueError(f"Invalid func_id {func_id}")
                    f = funcs[idx]
                else:
                    # Full list of 30
                    f = funcs[func_id - 1]
                return _wrap_objective(f, dim), f_opt
            except Exception:
                pass
        else:
            # It's a list (external 00-CEC-Root format)
            try:
                if len(all_funcs) == 30:
                    f = all_funcs[func_id - 1]
                    return _wrap_objective(f, dim), f_opt
            except Exception:
                pass
    
    # Try cec2017_test_func or cec2017_func (compatibility wrappers)
    for func_name in ['cec2017_test_func', 'cec2017_func']:
        if hasattr(_cec2017_module, func_name):
            test_func = getattr(_cec2017_module, func_name)
            
            def objective(X, _func_id=func_id, _test_func=test_func):
                X = np.atleast_2d(X).astype(np.float64)
                return np.asarray(_test_func(X, _func_id), dtype=np.float64).ravel()
            
            return objective, f_opt
    
    # Fallback to synthetic
    return _create_synthetic_function(func_id, dim)


def _wrap_objective(f: Any, dim: int) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a benchmark function to ensure vectorized (N,D) -> (N,) interface."""
    
    def objective(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")
        
        n = X.shape[0]
        
        # Try batch evaluation first
        try:
            y = f(X)
            y = np.asarray(y, dtype=np.float64).ravel()
            if y.shape[0] == n:
                return y
        except Exception:
            pass
        
        # Fallback: row by row
        y = np.empty(n, dtype=np.float64)
        for i in range(n):
            try:
                yi = f(X[i:i+1, :])
                y[i] = float(np.asarray(yi).flat[0])
            except Exception:
                # Single row evaluation
                yi = f(X[i])
                y[i] = float(np.asarray(yi).flat[0])
        return y
    
    return objective


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
# Reset function for testing
# =============================================================================

def reset_cec2017_cache() -> None:
    """Reset the CEC2017 module cache (useful for testing)."""
    global _cec2017_module, _cec2017_loaded, _cec2017_source
    _cec2017_module = None
    _cec2017_loaded = False
    _cec2017_source = None


def get_cec2017_source() -> Optional[str]:
    """Get the source of the currently loaded CEC2017 module."""
    return _cec2017_source
