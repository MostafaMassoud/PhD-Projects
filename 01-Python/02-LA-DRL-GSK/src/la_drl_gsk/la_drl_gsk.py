"""
LA-DRL-GSK: Q1 Implementation
==============================

Landscape-Aware Deep Reinforcement Learning Controller for GSK.

Q1 Key Features:
- Windowed control: RL decision every W generations
- Action space: {K, kf, kr, p} (absolute parameters)
- State: 25 FLA features from ZeroCostLandscapeAnalyzer
- Configurable senior stratification via p parameter

Performance Strategy:
- NumPy for all population arrays and GSK core
- Optional PyTorch/SB3 for trained policy
- Vectorized operations throughout

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import time
import os
import platform
import sys

if TYPE_CHECKING:
    from .logger import OptimizationLogger

from .landscape_analyzer import ZeroCostLandscapeAnalyzer, FeatureGroupAnalyzer
from .controllers import (
    BaseController,
    FixedController,
    HeuristicController,
    create_controller,
    PARAM_RANGES,
)


# =============================================================================
# Platform Detection and Configuration
# =============================================================================

def get_platform_info() -> Dict[str, str]:
    """Get current platform information."""
    return {
        'system': platform.system(),
        'machine': platform.machine(),
        'python': sys.version,
    }


def get_optimal_device() -> str:
    """
    Get the optimal PyTorch device for current platform.
    
    Returns
    -------
    str
        'mps' for Apple Silicon, 'cpu' for Windows/Linux
    """
    try:
        import torch
        
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return 'mps'
        
        return 'cpu'
    except ImportError:
        return 'cpu'


def configure_threads(n_threads: Optional[int] = None, n_interop: int = 2) -> None:
    """
    Configure optimal thread settings for current platform.
    
    Parameters
    ----------
    n_threads : int, optional
        Number of threads. Auto-detect if None.
    n_interop : int
        Number of inter-op threads
    """
    if n_threads is None:
        cpu_count = os.cpu_count() or 4
        if platform.system() == 'Windows':
            n_threads = max(1, cpu_count // 2)
        elif platform.system() == 'Darwin':
            n_threads = min(8, cpu_count)
        else:
            n_threads = max(1, cpu_count // 2)
    
    os.environ['MKL_NUM_THREADS'] = str(n_threads)
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(n_threads)
    
    try:
        import torch
        torch.set_num_threads(n_threads)
        if hasattr(torch, 'set_num_interop_threads'):
            torch.set_num_interop_threads(n_interop)
    except ImportError:
        pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LADRLGSKConfig:
    """
    LA-DRL-GSK configuration.
    
    Q1 Parameters:
    - K: Knowledge rate exponent [1, 20]
    - KF: Knowledge factor [0.05, 1.0]
    - KR: Knowledge ratio [0.05, 0.99]
    - P: Senior stratification fraction [0.05, 0.20]
    
    Control Parameters:
    - control_window: Number of generations per RL decision
    - controller_backend: "fixed" | "heuristic" | "sb3"
    """
    
    # Problem configuration
    dim: int = 30
    pop_size: int = 100
    bounds: Tuple[float, float] = (-100.0, 100.0)
    max_nfes: Optional[int] = None
    seed: int = 123456
    
    # Base GSK parameters (used as defaults)
    K: float = 10.0   # Knowledge rate exponent
    KF: float = 0.5   # Knowledge factor
    KR: float = 0.9   # Knowledge ratio
    P: float = 0.1    # Senior stratification fraction
    
    # Control settings
    use_rl: bool = False
    controller_backend: str = "heuristic"  # "fixed" | "heuristic" | "sb3"
    policy_path: Optional[str] = None
    control_window: int = 5  # Generations per RL decision
    
    # Ablation
    ablation_mode: Optional[str] = None
    
    def resolved_max_nfes(self) -> int:
        """Get max NFEs, defaulting to 10000 * dim."""
        return self.max_nfes or 10000 * self.dim


@dataclass
class LADRLGSKResult:
    """Optimization result container."""
    best_x: np.ndarray
    best_f: float
    nfes_used: int
    max_nfes: int
    history: np.ndarray
    stop_reason: str
    runtime: float
    actions_taken: List[Dict] = field(default_factory=list)
    
    def error(self, f_opt: float = 0.0) -> float:
        """Compute error from known optimum."""
        return abs(self.best_f - f_opt)


# =============================================================================
# GSK Run State (for step-based API)
# =============================================================================

@dataclass
class GSKRunState:
    """
    State container for step-based GSK execution.
    
    Enables Gym environment integration without duplicating GSK logic.
    """
    # Population state
    pop: np.ndarray           # (NP, D) current population
    fitness: np.ndarray       # (NP,) fitness values
    
    # Best tracking
    best_x: np.ndarray        # Best solution found
    best_f: float             # Best fitness found
    
    # Progress tracking
    nfes: int                 # Number of function evaluations used
    g: int                    # Current generation
    
    # Budget
    max_nfes: int             # Maximum NFEs
    G_max: int                # Maximum generations
    
    # Problem info
    dim: int
    pop_size: int
    bounds: Tuple[float, float]
    
    # Pre-allocated work arrays
    Gained_Shared_Junior: np.ndarray
    Gained_Shared_Senior: np.ndarray
    ui: np.ndarray
    lu: np.ndarray            # Bounds array (2, D)
    
    # History
    history: List[float] = field(default_factory=list)


# =============================================================================
# LA-DRL-GSK Main Class
# =============================================================================

class LADRLGSK:
    """
    LA-DRL-GSK: Landscape-Aware Deep RL Controller for GSK.
    
    Q1 Implementation Features:
    - Windowed control (W generations per RL decision)
    - Action space: {K, kf, kr, p} absolute parameters
    - State: 25 FLA features from ZeroCostLandscapeAnalyzer
    - Controller backends: fixed, heuristic, sb3
    
    Usage
    -----
    # Baseline GSK (fixed parameters)
    config = LADRLGSKConfig(dim=30, use_rl=False)
    optimizer = LADRLGSK(config)
    result = optimizer.optimize(objective)
    
    # LA-DRL-GSK with heuristic controller
    config = LADRLGSKConfig(dim=30, use_rl=True, controller_backend="heuristic")
    optimizer = LADRLGSK(config)
    result = optimizer.optimize(objective)
    
    # LA-DRL-GSK with trained SB3 policy
    config = LADRLGSKConfig(
        dim=30, use_rl=True, 
        controller_backend="sb3",
        policy_path="models/ppo_ladrl_gsk.zip"
    )
    optimizer = LADRLGSK(config)
    result = optimizer.optimize(objective)
    """
    
    def __init__(self, config: LADRLGSKConfig):
        """
        Initialize LA-DRL-GSK.
        
        Parameters
        ----------
        config : LADRLGSKConfig
            Configuration object
        """
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # FLA analyzer
        self.analyzer = ZeroCostLandscapeAnalyzer(
            dim=config.dim,
            pop_size=config.pop_size,
            max_nfes=config.resolved_max_nfes(),
        )
        
        # Controller (lazy initialization)
        self._controller: Optional[BaseController] = None
    
    @property
    def controller(self) -> BaseController:
        """Get or create the controller."""
        if self._controller is None:
            if not self.config.use_rl:
                # Baseline: fixed parameters
                self._controller = FixedController(
                    K=self.config.K,
                    kf=self.config.KF,
                    kr=self.config.KR,
                    p=self.config.P,
                )
            else:
                # RL-controlled
                self._controller = create_controller(
                    backend=self.config.controller_backend,
                    policy_path=self.config.policy_path,
                    K=self.config.K,
                    kf=self.config.KF,
                    kr=self.config.KR,
                    p=self.config.P,
                )
        return self._controller
    
    # =========================================================================
    # Step-based API (for Gym environment)
    # =========================================================================
    
    def reset_run(self, objective: Callable[[np.ndarray], np.ndarray]) -> GSKRunState:
        """
        Initialize a new optimization run.
        
        Parameters
        ----------
        objective : callable
            Function f(X: (N,D)) -> (N,)
        
        Returns
        -------
        GSKRunState
            Initial state with evaluated population
        """
        D = self.config.dim
        NP = self.config.pop_size
        bounds = self.config.bounds
        max_nfes = self.config.resolved_max_nfes()
        
        # Reset analyzer
        self.analyzer.reset()
        
        # Initialize population
        lu = np.array([[bounds[0]] * D, [bounds[1]] * D], dtype=np.float64)
        pop = self.rng.uniform(bounds[0], bounds[1], (NP, D))
        fitness = objective(pop)
        nfes = NP
        
        # Find best
        best_idx = np.argmin(fitness)
        best_f = float(fitness[best_idx])
        best_x = pop[best_idx].copy()
        
        # Pre-allocate work arrays
        Gained_Shared_Junior = np.empty((NP, D), dtype=np.float64)
        Gained_Shared_Senior = np.empty((NP, D), dtype=np.float64)
        ui = np.empty((NP, D), dtype=np.float64)
        
        return GSKRunState(
            pop=pop,
            fitness=fitness,
            best_x=best_x,
            best_f=best_f,
            nfes=nfes,
            g=0,
            max_nfes=max_nfes,
            G_max=max(1, max_nfes // NP),
            dim=D,
            pop_size=NP,
            bounds=bounds,
            Gained_Shared_Junior=Gained_Shared_Junior,
            Gained_Shared_Senior=Gained_Shared_Senior,
            ui=ui,
            lu=lu,
            history=[best_f],
        )
    
    def step_generation(
        self,
        state: GSKRunState,
        objective: Callable[[np.ndarray], np.ndarray],
        K: float,
        kf: float,
        kr: float,
        p: float,
    ) -> GSKRunState:
        """
        Execute one GSK generation.
        
        Parameters
        ----------
        state : GSKRunState
            Current state
        objective : callable
            Objective function
        K, kf, kr, p : float
            GSK parameters for this generation
        
        Returns
        -------
        GSKRunState
            Updated state
        """
        D = state.dim
        NP = state.pop_size
        pop = state.pop
        fitness = state.fitness
        
        state.g += 1
        g = state.g
        G_max = state.G_max
        
        # Compute junior/senior split
        D_junior = int(np.ceil(D * (1 - g / G_max) ** K))
        p_junior = max(0.0, min(1.0, D_junior / D))
        
        # Sort by fitness
        ind_best = np.argsort(fitness)
        
        # Junior phase
        Rg1, Rg2, Rg3 = self._junior_indices(ind_best, NP)
        
        worse = fitness > fitness[Rg3]
        better = ~worse
        
        # Vectorized junior mutation
        state.Gained_Shared_Junior[worse] = (
            pop[worse] + kf * (
                pop[Rg1[worse]] - pop[Rg2[worse]] + 
                pop[Rg3[worse]] - pop[worse]
            )
        )
        state.Gained_Shared_Junior[better] = (
            pop[better] + kf * (
                pop[Rg1[better]] - pop[Rg2[better]] + 
                pop[better] - pop[Rg3[better]]
            )
        )
        
        # Senior phase
        R1, R2, R3 = self._senior_indices(ind_best, NP, p)
        
        worse2 = fitness > fitness[R2]
        better2 = ~worse2
        
        # Vectorized senior mutation
        state.Gained_Shared_Senior[worse2] = (
            pop[worse2] + kf * (
                pop[R1[worse2]] - pop[worse2] + 
                pop[R2[worse2]] - pop[R3[worse2]]
            )
        )
        state.Gained_Shared_Senior[better2] = (
            pop[better2] + kf * (
                pop[R1[better2]] - pop[R2[better2]] + 
                pop[better2] - pop[R3[better2]]
            )
        )
        
        # Boundary handling
        self._bound_constraint(state.Gained_Shared_Junior, pop, state.lu)
        self._bound_constraint(state.Gained_Shared_Senior, pop, state.lu)
        
        # Crossover
        rand_phase = self.rng.random((NP, D))
        rand_j = self.rng.random((NP, D))
        rand_s = self.rng.random((NP, D))
        
        use_junior = (rand_phase <= p_junior) & (rand_j <= kr)
        use_senior = (rand_phase > p_junior) & (rand_s <= kr)
        
        np.copyto(state.ui, pop)
        state.ui[use_junior] = state.Gained_Shared_Junior[use_junior]
        state.ui[use_senior] = state.Gained_Shared_Senior[use_senior]
        
        # Evaluate offspring
        child_fit = objective(state.ui)
        state.nfes += NP
        
        # Update best
        gen_best_idx = np.argmin(child_fit)
        if child_fit[gen_best_idx] < state.best_f:
            state.best_f = float(child_fit[gen_best_idx])
            state.best_x = state.ui[gen_best_idx].copy()
        
        state.history.append(state.best_f)
        
        # Greedy selection
        improved = child_fit < fitness
        state.pop[improved] = state.ui[improved]
        state.fitness[improved] = child_fit[improved]
        
        return state
    
    def run_generations(
        self,
        state: GSKRunState,
        objective: Callable[[np.ndarray], np.ndarray],
        n_gen: int,
        K: float,
        kf: float,
        kr: float,
        p: float,
    ) -> GSKRunState:
        """
        Run multiple generations with fixed parameters.
        
        Parameters
        ----------
        state : GSKRunState
            Current state
        objective : callable
            Objective function
        n_gen : int
            Number of generations to run
        K, kf, kr, p : float
            GSK parameters
        
        Returns
        -------
        GSKRunState
            Updated state
        """
        for _ in range(n_gen):
            if state.nfes + state.pop_size > state.max_nfes:
                break
            state = self.step_generation(state, objective, K, kf, kr, p)
        return state
    
    # =========================================================================
    # Main Optimization Method
    # =========================================================================
    
    def optimize(
        self,
        objective: Callable[[np.ndarray], np.ndarray],
        verbose: bool = False,
        logger: Optional['OptimizationLogger'] = None,
    ) -> LADRLGSKResult:
        """
        Run optimization.
        
        Parameters
        ----------
        objective : callable
            Function f(X: (N,D)) -> (N,)
        verbose : bool
            Print progress
        logger : OptimizationLogger, optional
            Detailed logger
        
        Returns
        -------
        LADRLGSKResult
        """
        start_time = time.time()
        
        # Initialize run
        state = self.reset_run(objective)
        
        # =====================================================================
        # BASELINE FAST PATH (Task D)
        # When use_rl=False, no logger, and no ablation, skip FLA computation
        # =====================================================================
        if not self.config.use_rl and logger is None and self.config.ablation_mode is None:
            # Pure baseline GSK: fixed parameters, no FLA overhead
            K = self.config.K
            kf = self.config.KF
            kr = self.config.KR
            p = self.config.P
            
            # Run all generations with fixed parameters
            while state.nfes + state.pop_size <= state.max_nfes:
                state = self.step_generation(state, objective, K, kf, kr, p)
                
                if verbose and state.g % 100 == 0:
                    print(f"  Gen {state.g}: nfes={state.nfes} best={state.best_f:.6e}")
            
            return LADRLGSKResult(
                best_x=state.best_x,
                best_f=state.best_f,
                nfes_used=state.nfes,
                max_nfes=state.max_nfes,
                history=np.array(state.history),
                stop_reason="budget_exhausted" if state.nfes >= state.max_nfes else "completed",
                runtime=time.time() - start_time,
                actions_taken=[],  # No RL actions in baseline
            )
        
        # =====================================================================
        # RL-CONTROLLED PATH (with optional logging)
        # =====================================================================
        
        # Reset controller's analyzer (NOT the log analyzer)
        self.analyzer.reset()
        
        # Initialize controller
        controller = self.controller
        controller.reset(
            self.config.dim,
            self.config.pop_size,
            self.config.resolved_max_nfes(),
        )
        
        # Create separate analyzer for logging if needed (Task E)
        # This prevents log_analyzer.compute_state() from corrupting controller state
        log_analyzer = None
        if logger is not None and logger.show_features:
            log_analyzer = ZeroCostLandscapeAnalyzer(
                dim=self.config.dim,
                pop_size=self.config.pop_size,
                max_nfes=self.config.resolved_max_nfes(),
            )
        
        # Log header
        if logger is not None:
            logger.print_header({
                'dim': self.config.dim,
                'pop_size': self.config.pop_size,
                'max_nfes': state.max_nfes,
                'bounds': self.config.bounds,
                'seed': self.config.seed,
                'K': self.config.K,
                'KF': self.config.KF,
                'KR': self.config.KR,
                'P': self.config.P,
                'use_rl': self.config.use_rl,
                'controller_backend': self.config.controller_backend,
                'policy_path': self.config.policy_path,
                'control_window': self.config.control_window,
            })
        
        actions_taken = []
        W = self.config.control_window
        
        # Current parameters (start with defaults)
        K = self.config.K
        kf = self.config.KF
        kr = self.config.KR
        p = self.config.P
        
        # Main optimization loop
        while state.nfes + state.pop_size <= state.max_nfes:
            # Get observation for controller (only from self.analyzer, not log_analyzer)
            obs = self.analyzer.compute_state(state.pop, state.fitness, state.nfes)
            
            # Apply ablation if configured
            if self.config.ablation_mode:
                obs = FeatureGroupAnalyzer.mask_feature_group(
                    obs, self.config.ablation_mode
                )
            
            # Get action from controller
            params = controller.act(obs, info={'nfes': state.nfes, 'best_f': state.best_f})
            K = params['K']
            kf = params['kf']
            kr = params['kr']
            p = params['p']
            
            # Record action
            actions_taken.append({
                'gen': state.g,
                'nfes': state.nfes,
                'K': K,
                'kf': kf,
                'kr': kr,
                'p': p,
                'best_f': state.best_f,
            })
            
            # Run W generations with these parameters
            gens_to_run = min(W, (state.max_nfes - state.nfes) // state.pop_size)
            
            for _ in range(gens_to_run):
                if state.nfes + state.pop_size > state.max_nfes:
                    break
                
                state = self.step_generation(state, objective, K, kf, kr, p)
                
                # Log generation (using separate log_analyzer to not corrupt controller state)
                if logger is not None:
                    obs_log = None
                    if log_analyzer is not None:
                        # Use separate analyzer for logging features
                        obs_log = log_analyzer.compute_state(state.pop, state.fitness, state.nfes)
                    
                    logger.log_generation(
                        generation=state.g,
                        nfes=state.nfes,
                        max_nfes=state.max_nfes,
                        fitness=state.fitness,
                        best_f=state.best_f,
                        p_junior=int(np.ceil(self.config.dim * (1 - state.g / state.G_max) ** K)) / self.config.dim,
                        KF=kf,
                        KR=kr,
                        strategy='(Q1)',  # Deprecated in Q1, kept for backward compatibility
                        features=obs_log if logger.show_features else None,
                        K=K,
                        p_senior=p,  # Senior stratification fraction from RL action
                    )
                
                if verbose and state.g % 100 == 0:
                    print(f"  Gen {state.g}: nfes={state.nfes} best={state.best_f:.6e}")
        
        result = LADRLGSKResult(
            best_x=state.best_x,
            best_f=state.best_f,
            nfes_used=state.nfes,
            max_nfes=state.max_nfes,
            history=np.array(state.history),
            stop_reason="budget_exhausted" if state.nfes >= state.max_nfes else "completed",
            runtime=time.time() - start_time,
            actions_taken=actions_taken,
        )
        
        if logger is not None:
            logger.print_summary(result)
        
        return result
    
    # =========================================================================
    # GSK Helper Methods
    # =========================================================================
    
    def _junior_indices(self, ind_best: np.ndarray, NP: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute junior phase indices with proper edge handling.
        
        Matches original GSK MATLAB implementation:
        - BEST (rank 0): R1=2nd best (rank 1), R2=3rd best (rank 2)
        - WORST (rank N-1): R1=3rd worst (rank N-3), R2=2nd worst (rank N-2)
        - MIDDLE: R1=better neighbor (rank-1), R2=worse neighbor (rank+1)
        - R3 must be ≠ self, R1, R2
        
        Returns
        -------
        Rg1 : ndarray
            Better neighbor indices
        Rg2 : ndarray
            Worse neighbor indices  
        Rg3 : ndarray
            Random indices (conflict-free with self, Rg1, and Rg2)
        """
        ind_best = np.asarray(ind_best, dtype=np.int64)
        idx = np.arange(NP, dtype=np.int64)
        
        # Build rank lookup: rank_of[i] = rank of individual i
        rank_of = np.empty(NP, dtype=np.int64)
        rank_of[ind_best] = idx
        
        # Allocate output arrays
        R1 = np.empty(NP, dtype=np.int64)
        R2 = np.empty(NP, dtype=np.int64)
        
        # Handle edge cases
        best_mask = (rank_of == 0)          # Best individual
        worst_mask = (rank_of == NP - 1)    # Worst individual
        middle_mask = ~(best_mask | worst_mask)
        
        # Best individual: use 2nd and 3rd best as neighbors
        if NP >= 3:
            R1[best_mask] = ind_best[1]  # 2nd best
            R2[best_mask] = ind_best[2]  # 3rd best
        else:
            # Very small population fallback
            R1[best_mask] = ind_best[min(1, NP-1)]
            R2[best_mask] = ind_best[min(2, NP-1)]
        
        # Worst individual: use 3rd-worst and 2nd-worst as neighbors
        if NP >= 3:
            R1[worst_mask] = ind_best[NP - 3]  # 3rd worst
            R2[worst_mask] = ind_best[NP - 2]  # 2nd worst
        else:
            R1[worst_mask] = ind_best[max(0, NP-3)]
            R2[worst_mask] = ind_best[max(0, NP-2)]
        
        # Middle individuals: immediate rank neighbors
        if np.any(middle_mask):
            ranks = rank_of[middle_mask]
            R1[middle_mask] = ind_best[ranks - 1]  # Better neighbor
            R2[middle_mask] = ind_best[ranks + 1]  # Worse neighbor
        
        # Generate conflict-free R3
        # R3 must be ≠ self (idx), R1, and R2
        R3 = self.rng.randint(0, NP, NP)
        
        # For very small populations (NP < 4), we may not be able to satisfy all constraints
        if NP >= 4:
            # Resolve conflicts iteratively with hard cap
            max_iterations = 20
            for _ in range(max_iterations):
                conflict_mask = (R3 == idx) | (R3 == R1) | (R3 == R2)
                if not np.any(conflict_mask):
                    break
                n_conflicts = int(conflict_mask.sum())
                R3[conflict_mask] = self.rng.randint(0, NP, n_conflicts)
        else:
            # Small population: just ensure R3 != self
            max_iterations = 20
            for _ in range(max_iterations):
                conflict_mask = (R3 == idx)
                if not np.any(conflict_mask):
                    break
                n_conflicts = int(conflict_mask.sum())
                R3[conflict_mask] = self.rng.randint(0, NP, n_conflicts)
        
        return R1, R2, R3
    
    def _senior_indices(
        self, 
        ind_best: np.ndarray, 
        NP: int, 
        p: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute senior phase indices with configurable stratification.
        
        Parameters
        ----------
        ind_best : np.ndarray
            Indices sorted by fitness (best first)
        NP : int
            Population size
        p : float
            Stratification fraction for top/bottom groups
        
        Returns
        -------
        R1 : ndarray
            Indices from best group
        R2 : ndarray
            Indices from middle group
        R3 : ndarray
            Indices from worst group
        """
        # Calculate group sizes
        n_top = max(1, int(np.floor(NP * p)))
        n_bottom = max(1, int(np.floor(NP * p)))
        
        # Ensure middle group is non-empty
        while NP - n_top - n_bottom < 1 and (n_top > 1 or n_bottom > 1):
            if n_top > 1:
                n_top -= 1
            if n_bottom > 1 and NP - n_top - n_bottom < 1:
                n_bottom -= 1
        
        n_middle = max(1, NP - n_top - n_bottom)
        
        # Define groups
        top_group = ind_best[:n_top]
        mid_start = n_top
        mid_end = NP - n_bottom
        middle_group = ind_best[mid_start:mid_end] if mid_end > mid_start else ind_best[[NP // 2]]
        bottom_group = ind_best[-n_bottom:] if n_bottom > 0 else ind_best[[-1]]
        
        # Random selection from each group
        R1 = top_group[self.rng.randint(0, len(top_group), NP)]
        R2 = middle_group[self.rng.randint(0, len(middle_group), NP)]
        R3 = bottom_group[self.rng.randint(0, len(bottom_group), NP)]
        
        return R1, R2, R3
    
    def _bound_constraint(
        self, 
        vi: np.ndarray, 
        pop: np.ndarray, 
        lu: np.ndarray,
    ) -> None:
        """
        Apply boundary constraint (in-place).
        
        Half-way back reflection: if out of bounds, place halfway
        between bound and parent position.
        """
        lower = lu[0]  # Shape (D,)
        upper = lu[1]  # Shape (D,)
        
        # Use np.where for efficient broadcasting
        # Below lower bound: go halfway back from bound to parent
        vi[:] = np.where(vi < lower, (lower + pop) / 2, vi)
        
        # Above upper bound: go halfway back from bound to parent
        vi[:] = np.where(vi > upper, (upper + pop) / 2, vi)


# =============================================================================
# Factory Functions
# =============================================================================

def create_baseline_gsk(
    dim: int = 30,
    pop_size: int = 100,
    max_nfes: Optional[int] = None,
    seed: int = 42,
    K: float = 10.0,
    KF: float = 0.5,
    KR: float = 0.9,
) -> LADRLGSK:
    """
    Create a baseline GSK optimizer (no RL).
    
    Parameters
    ----------
    dim : int
        Problem dimension
    pop_size : int
        Population size
    max_nfes : int, optional
        Max evaluations (default: 10000*dim)
    seed : int
        Random seed
    K, KF, KR : float
        GSK parameters
    
    Returns
    -------
    LADRLGSK
        Configured optimizer
    """
    config = LADRLGSKConfig(
        dim=dim,
        pop_size=pop_size,
        max_nfes=max_nfes,
        seed=seed,
        K=K,
        KF=KF,
        KR=KR,
        use_rl=False,
    )
    return LADRLGSK(config)


def create_ladrl_gsk(
    dim: int = 30,
    pop_size: int = 100,
    max_nfes: Optional[int] = None,
    seed: int = 42,
    controller_backend: str = "heuristic",
    policy_path: Optional[str] = None,
    control_window: int = 5,
) -> LADRLGSK:
    """
    Create an LA-DRL-GSK optimizer.
    
    Parameters
    ----------
    dim : int
        Problem dimension
    pop_size : int
        Population size
    max_nfes : int, optional
        Max evaluations (default: 10000*dim)
    seed : int
        Random seed
    controller_backend : str
        "heuristic" or "sb3"
    policy_path : str, optional
        Path to SB3 model (required for "sb3" backend)
    control_window : int
        Generations per RL decision
    
    Returns
    -------
    LADRLGSK
        Configured optimizer
    """
    config = LADRLGSKConfig(
        dim=dim,
        pop_size=pop_size,
        max_nfes=max_nfes,
        seed=seed,
        use_rl=True,
        controller_backend=controller_backend,
        policy_path=policy_path,
        control_window=control_window,
    )
    return LADRLGSK(config)
