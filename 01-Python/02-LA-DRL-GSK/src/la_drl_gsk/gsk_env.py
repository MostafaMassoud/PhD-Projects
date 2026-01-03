"""
LA-DRL-GSK Gymnasium Environment
================================

Gymnasium environment for training RL controllers for GSK.

Q1 Environment Specification:
- Observation: 25 FLA features from ZeroCostLandscapeAnalyzer
- Action: 4D continuous [-1, 1] mapped to {K, kf, kr, p}
- Reward: Relative improvement in best fitness
- Episode: One optimization run until budget exhausted

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional, Tuple, List
import warnings

# Lazy gymnasium import
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None
    spaces = None

from .la_drl_gsk import LADRLGSK, LADRLGSKConfig, GSKRunState
from .landscape_analyzer import ZeroCostLandscapeAnalyzer
from .controllers import map_action_to_params, PARAM_RANGES


# =============================================================================
# Benchmark Suite
# =============================================================================

class BenchmarkSuite:
    """
    Collection of benchmark functions for training.
    
    Supports CEC2017 or synthetic fallback functions.
    """
    
    # Function categories for curriculum learning
    UNIMODAL = [1, 2, 3]
    SIMPLE_MULTIMODAL = [4, 5, 6, 7]
    COMPLEX_MULTIMODAL = [8, 9, 10]
    HYBRID = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    COMPOSITION = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    
    ALL_FUNCTIONS = list(range(1, 31))
    
    def __init__(
        self,
        dim: int = 10,
        cec_path: Optional[str] = None,
        functions: Optional[List[int]] = None,
    ):
        """
        Initialize benchmark suite.
        
        Parameters
        ----------
        dim : int
            Problem dimension
        cec_path : str, optional
            Path to CEC2017 data files
        functions : list, optional
            List of function IDs to use (default: all)
        """
        self.dim = dim
        self.cec_path = cec_path
        self.functions = functions or self.ALL_FUNCTIONS
        
        # Try to load CEC2017
        self._cec_available = False
        try:
            from .cec2017_benchmark import get_cec2017_function
            self._get_func = get_cec2017_function
            # Test if it works
            _, _ = self._get_func(1, dim)
            self._cec_available = True
        except Exception as e:
            warnings.warn(f"CEC2017 not available ({e}), using synthetic functions")
            self._cec_available = False
    
    def sample_function(self, rng: np.random.RandomState) -> Tuple[callable, float, int]:
        """
        Sample a random benchmark function.
        
        Returns
        -------
        objective : callable
            Function f(X: (N,D)) -> (N,)
        f_opt : float
            Known optimum value
        func_id : int
            Function ID
        """
        func_id = rng.choice(self.functions)
        
        if self._cec_available:
            try:
                objective, f_opt = self._get_func(func_id, self.dim)
                return objective, f_opt, func_id
            except Exception:
                pass
        
        # Fallback to synthetic functions
        return self._create_synthetic(func_id, self.dim, rng)
    
    def _create_synthetic(
        self, 
        func_id: int, 
        dim: int,
        rng: np.random.RandomState,
    ) -> Tuple[callable, float, int]:
        """Create synthetic test function."""
        f_opt = func_id * 100  # Offset for each function
        
        # Random rotation and shift
        shift = rng.uniform(-50, 50, dim)
        
        if func_id <= 3:
            # Unimodal: Sphere-like
            def objective(x):
                x = np.atleast_2d(x)
                z = x - shift
                return np.sum(z ** 2, axis=1) + f_opt
        elif func_id <= 10:
            # Multimodal: Rastrigin-like
            def objective(x):
                x = np.atleast_2d(x)
                z = x - shift
                return 10 * dim + np.sum(z**2 - 10*np.cos(2*np.pi*z), axis=1) + f_opt
        elif func_id <= 20:
            # Hybrid: Griewank-like
            def objective(x):
                x = np.atleast_2d(x)
                z = x - shift
                sum_sq = np.sum(z**2, axis=1) / 4000
                prod_cos = np.prod(np.cos(z / np.sqrt(np.arange(1, dim+1))), axis=1)
                return sum_sq - prod_cos + 1 + f_opt
        else:
            # Composition: Ackley-like
            def objective(x):
                x = np.atleast_2d(x)
                z = x - shift
                t1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(z**2, axis=1)))
                t2 = -np.exp(np.mean(np.cos(2*np.pi*z), axis=1))
                return t1 + t2 + 20 + np.e + f_opt
        
        return objective, f_opt, func_id
    
    def set_curriculum(self, stage: str):
        """
        Set curriculum stage for training.
        
        Parameters
        ----------
        stage : str
            "easy", "medium", "hard", or "all"
        """
        if stage == "easy":
            self.functions = self.UNIMODAL + self.SIMPLE_MULTIMODAL
        elif stage == "medium":
            self.functions = self.UNIMODAL + self.SIMPLE_MULTIMODAL + self.COMPLEX_MULTIMODAL
        elif stage == "hard":
            self.functions = self.ALL_FUNCTIONS
        else:
            self.functions = self.ALL_FUNCTIONS


# =============================================================================
# GSK Control Environment
# =============================================================================

if HAS_GYMNASIUM:
    
    class GSKControlEnv(gym.Env):
        """
        Gymnasium environment for GSK parameter control.
        
        Observation Space:
            Box(0, 1, shape=(25,)) - 25 FLA features
        
        Action Space:
            Box(-1, 1, shape=(4,)) - [K_norm, kf_norm, kr_norm, p_norm]
        
        Episode:
            One optimization run with control_window generations per step.
            Terminates when NFE budget is exhausted.
        
        Reward:
            Relative improvement in best fitness over control window.
        """
        
        metadata = {"render_modes": []}
        
        def __init__(
            self,
            dim: int = 10,
            pop_size: int = 100,
            max_nfes: Optional[int] = None,
            control_window: int = 5,
            functions: Optional[List[int]] = None,
            cec_path: Optional[str] = None,
            seed: Optional[int] = None,
        ):
            """
            Initialize environment.
            
            Parameters
            ----------
            dim : int
                Problem dimension
            pop_size : int
                Population size
            max_nfes : int, optional
                Maximum NFEs (default: 10000*dim)
            control_window : int
                Generations per RL step
            functions : list, optional
                Benchmark function IDs to use
            cec_path : str, optional
                Path to CEC2017 data
            seed : int, optional
                Random seed
            """
            super().__init__()
            
            self.dim = dim
            self.pop_size = pop_size
            self.max_nfes = max_nfes or 10000 * dim
            self.control_window = control_window
            
            # Observation and action spaces
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(25,),
                dtype=np.float32,
            )
            
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            )
            
            # Benchmark suite
            self.benchmark = BenchmarkSuite(
                dim=dim,
                cec_path=cec_path,
                functions=functions,
            )
            
            # Random state
            self._seed = seed
            self.rng = np.random.RandomState(seed)
            
            # Optimizer and state (initialized in reset)
            self._optimizer: Optional[LADRLGSK] = None
            self._state: Optional[GSKRunState] = None
            self._objective: Optional[callable] = None
            self._f_opt: float = 0.0
            self._func_id: int = 0
            self._best_prev: float = float('inf')
        
        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None,
        ) -> Tuple[np.ndarray, Dict]:
            """
            Reset environment for new episode.
            
            Parameters
            ----------
            seed : int, optional
                Random seed
            options : dict, optional
                Additional options
            
            Returns
            -------
            observation : ndarray
                Initial FLA features
            info : dict
                Episode info
            """
            if seed is not None:
                self.rng = np.random.RandomState(seed)
            
            # Sample benchmark function
            self._objective, self._f_opt, self._func_id = self.benchmark.sample_function(self.rng)
            
            # Create optimizer
            config = LADRLGSKConfig(
                dim=self.dim,
                pop_size=self.pop_size,
                max_nfes=self.max_nfes,
                seed=int(self.rng.randint(0, 2**31)),
                use_rl=False,  # We control externally
            )
            self._optimizer = LADRLGSK(config)
            
            # Initialize run
            self._state = self._optimizer.reset_run(self._objective)
            self._best_prev = self._state.best_f
            
            # Compute initial observation
            obs = self._optimizer.analyzer.compute_state(
                self._state.pop,
                self._state.fitness,
                self._state.nfes,
            )
            
            info = {
                'func_id': self._func_id,
                'dim': self.dim,
                'f_opt': self._f_opt,
                'nfes': self._state.nfes,
                'best_f': self._state.best_f,
            }
            
            return obs.astype(np.float32), info
        
        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            """
            Execute one control step.
            
            Parameters
            ----------
            action : ndarray
                4D action in [-1, 1]
            
            Returns
            -------
            observation : ndarray
                New FLA features
            reward : float
                Relative improvement (clipped)
            terminated : bool
                True if budget exhausted
            truncated : bool
                Always False
            info : dict
                Step info
            """
            # Map action to parameters
            params = map_action_to_params(action)
            K = params['K']
            kf = params['kf']
            kr = params['kr']
            p = params['p']
            
            # Store previous best
            self._best_prev = self._state.best_f
            
            # Run control_window generations
            gens_run = 0
            for _ in range(self.control_window):
                if self._state.nfes + self._state.pop_size > self._state.max_nfes:
                    break
                self._state = self._optimizer.step_generation(
                    self._state,
                    self._objective,
                    K, kf, kr, p,
                )
                gens_run += 1
            
            # Compute reward
            best_now = self._state.best_f
            if abs(self._best_prev) > 1e-12:
                reward = (self._best_prev - best_now) / abs(self._best_prev)
            else:
                reward = 0.0 if best_now >= self._best_prev else 1.0
            
            # Clip reward
            reward = float(np.clip(reward, -1.0, 1.0))
            
            # Compute observation
            obs = self._optimizer.analyzer.compute_state(
                self._state.pop,
                self._state.fitness,
                self._state.nfes,
            )
            
            # Check termination
            terminated = self._state.nfes + self._state.pop_size > self._state.max_nfes
            truncated = False
            
            info = {
                'func_id': self._func_id,
                'dim': self.dim,
                'f_opt': self._f_opt,
                'nfes': self._state.nfes,
                'best_f': self._state.best_f,
                'error': abs(self._state.best_f - self._f_opt),
                'params': params,
                'gens_run': gens_run,
            }
            
            return obs.astype(np.float32), reward, terminated, truncated, info
        
        def set_curriculum(self, stage: str):
            """Set curriculum stage."""
            self.benchmark.set_curriculum(stage)
    
    
    # =========================================================================
    # Environment Registration
    # =========================================================================
    
    def make_gsk_env(
        dim: int = 10,
        pop_size: int = 100,
        max_nfes: Optional[int] = None,
        control_window: int = 5,
        functions: Optional[List[int]] = None,
        cec_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> GSKControlEnv:
        """
        Factory function to create GSK control environment.
        
        Parameters
        ----------
        dim : int
            Problem dimension
        pop_size : int
            Population size
        max_nfes : int, optional
            Maximum NFEs
        control_window : int
            Generations per RL step
        functions : list, optional
            Benchmark function IDs
        cec_path : str, optional
            CEC2017 data path
        seed : int, optional
            Random seed
        
        Returns
        -------
        GSKControlEnv
        """
        return GSKControlEnv(
            dim=dim,
            pop_size=pop_size,
            max_nfes=max_nfes,
            control_window=control_window,
            functions=functions,
            cec_path=cec_path,
            seed=seed,
        )

else:
    # Gymnasium not installed - provide stub
    GSKControlEnv = None
    make_gsk_env = None
