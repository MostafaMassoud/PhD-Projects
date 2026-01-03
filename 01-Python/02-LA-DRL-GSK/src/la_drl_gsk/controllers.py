"""
LA-DRL-GSK Controllers
======================

Controller abstraction for GSK parameter control.
Provides Fixed, Heuristic, and SB3PPO controllers.

Q1 Action Space (absolute parameters):
- K:  Knowledge rate exponent [1, 20] - controls junior/senior transition
- kf: Knowledge factor [0.05, 1.0] - step size magnitude
- kr: Knowledge ratio [0.05, 0.99] - per-dimension update probability
- p:  Senior stratification [0.05, 0.20] - top/bottom group fraction

Author: LA-DRL-GSK Research Team
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass


# =============================================================================
# Parameter Ranges (Single Source of Truth)
# =============================================================================

@dataclass(frozen=True)
class ParamRanges:
    """Q1 parameter ranges for GSK control."""
    
    # Knowledge rate exponent (controls junior/senior dimension split)
    K_MIN: float = 1.0
    K_MAX: float = 20.0
    K_DEFAULT: float = 10.0
    
    # Knowledge factor (step size magnitude)
    KF_MIN: float = 0.05
    KF_MAX: float = 1.0
    KF_DEFAULT: float = 0.5
    
    # Knowledge ratio (per-dimension update probability)
    KR_MIN: float = 0.05
    KR_MAX: float = 0.99
    KR_DEFAULT: float = 0.9
    
    # Senior stratification fraction
    P_MIN: float = 0.05
    P_MAX: float = 0.20
    P_DEFAULT: float = 0.1


PARAM_RANGES = ParamRanges()


# =============================================================================
# Action Mapping Utilities
# =============================================================================

def map_action_to_params(action: np.ndarray) -> Dict[str, float]:
    """
    Map normalized action [-1, 1]^4 to GSK parameters.
    
    Parameters
    ----------
    action : np.ndarray
        Shape (4,) with values in [-1, 1]:
        - action[0] -> K (log-scaled)
        - action[1] -> kf (linear)
        - action[2] -> kr (linear)
        - action[3] -> p (linear)
    
    Returns
    -------
    dict
        {"K": float, "kf": float, "kr": float, "p": float}
    """
    R = PARAM_RANGES
    
    # Normalize action to [0, 1]
    a = (np.clip(action, -1, 1) + 1) / 2
    
    # K: log-scaled mapping for better coverage
    # log(K) in [log(K_MIN), log(K_MAX)]
    log_K_min = np.log(R.K_MIN)
    log_K_max = np.log(R.K_MAX)
    K = np.exp(log_K_min + a[0] * (log_K_max - log_K_min))
    
    # kf, kr, p: linear mapping
    kf = R.KF_MIN + a[1] * (R.KF_MAX - R.KF_MIN)
    kr = R.KR_MIN + a[2] * (R.KR_MAX - R.KR_MIN)
    p = R.P_MIN + a[3] * (R.P_MAX - R.P_MIN)
    
    return {
        "K": float(np.clip(K, R.K_MIN, R.K_MAX)),
        "kf": float(np.clip(kf, R.KF_MIN, R.KF_MAX)),
        "kr": float(np.clip(kr, R.KR_MIN, R.KR_MAX)),
        "p": float(np.clip(p, R.P_MIN, R.P_MAX)),
    }


def params_to_action(K: float, kf: float, kr: float, p: float) -> np.ndarray:
    """
    Map GSK parameters to normalized action [-1, 1]^4.
    
    Inverse of map_action_to_params.
    """
    R = PARAM_RANGES
    
    # K: log-scaled inverse
    log_K_min = np.log(R.K_MIN)
    log_K_max = np.log(R.K_MAX)
    a0 = (np.log(np.clip(K, R.K_MIN, R.K_MAX)) - log_K_min) / (log_K_max - log_K_min)
    
    # Linear inverse
    a1 = (np.clip(kf, R.KF_MIN, R.KF_MAX) - R.KF_MIN) / (R.KF_MAX - R.KF_MIN)
    a2 = (np.clip(kr, R.KR_MIN, R.KR_MAX) - R.KR_MIN) / (R.KR_MAX - R.KR_MIN)
    a3 = (np.clip(p, R.P_MIN, R.P_MAX) - R.P_MIN) / (R.P_MAX - R.P_MIN)
    
    # Convert from [0,1] to [-1,1]
    return np.array([a0 * 2 - 1, a1 * 2 - 1, a2 * 2 - 1, a3 * 2 - 1], dtype=np.float32)


# =============================================================================
# Base Controller
# =============================================================================

class BaseController(ABC):
    """Abstract base class for GSK controllers."""
    
    def __init__(self):
        self.dim: int = 0
        self.pop_size: int = 0
        self.max_nfes: int = 0
        self.G_max: int = 1
        self.g: int = 0  # Generation counter for windowed control
    
    def reset(self, dim: int, pop_size: int, max_nfes: int) -> None:
        """
        Reset controller for a new optimization run.
        
        Parameters
        ----------
        dim : int
            Problem dimension
        pop_size : int
            Population size
        max_nfes : int
            Maximum number of function evaluations
        """
        self.dim = dim
        self.pop_size = pop_size
        self.max_nfes = max_nfes
        self.G_max = max(1, max_nfes // pop_size)
        self.g = 0
    
    @abstractmethod
    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get GSK parameters for the current state.
        
        Parameters
        ----------
        obs : np.ndarray
            Observation/state vector (25 FLA features)
        info : dict, optional
            Additional info (nfes, best_f, etc.)
        
        Returns
        -------
        dict
            {"K": float, "kf": float, "kr": float, "p": float}
        """
        pass


# =============================================================================
# Fixed Controller
# =============================================================================

class FixedController(BaseController):
    """
    Fixed parameter controller (baseline GSK).
    
    Always returns the same configured parameters.
    """
    
    def __init__(
        self,
        K: float = PARAM_RANGES.K_DEFAULT,
        kf: float = PARAM_RANGES.KF_DEFAULT,
        kr: float = PARAM_RANGES.KR_DEFAULT,
        p: float = PARAM_RANGES.P_DEFAULT,
    ):
        super().__init__()
        self.K = K
        self.kf = kf
        self.kr = kr
        self.p = p
    
    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> Dict[str, float]:
        """Return fixed parameters."""
        return {
            "K": self.K,
            "kf": self.kf,
            "kr": self.kr,
            "p": self.p,
        }


# =============================================================================
# Heuristic Controller
# =============================================================================

class HeuristicController(BaseController):
    """
    Landscape-aware heuristic controller.
    
    Uses FLA features to adapt GSK parameters without RL training.
    No external dependencies (pure NumPy).
    
    Conservative adaptation strategy:
    - Only make small adjustments from baseline
    - Focus on escaping stagnation
    - Maintain stability by defaulting to baseline values
    """
    
    def __init__(
        self,
        base_K: float = PARAM_RANGES.K_DEFAULT,
        base_kf: float = PARAM_RANGES.KF_DEFAULT,
        base_kr: float = PARAM_RANGES.KR_DEFAULT,
        base_p: float = PARAM_RANGES.P_DEFAULT,
    ):
        super().__init__()
        self.base_K = base_K
        self.base_kf = base_kf
        self.base_kr = base_kr
        self.base_p = base_p
        
        # Tracking
        self._prev_best = float('inf')
        self._stagnation_window = 0
    
    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute adaptive parameters based on landscape features.
        
        Conservative strategy: only adjust when clearly needed.
        
        FLA Feature Layout (25 features):
        - 0-4:   Population (diversity, spread, centroid_offset, variance, elite_clustering)
        - 5-9:   Fitness (range, cv, skewness, kurtosis, elite_gap)
        - 10-14: Correlation (fdc, fdc_centroid, tau, separability, neighbor_corr)
        - 15-19: Temporal (improvement_rate, stagnation, div_trend, mean_trend, consistency)
        - 20-24: Progress (progress, convergence_ratio, exploitation, improvement_potential, quality)
        """
        R = PARAM_RANGES
        
        # Extract key features
        diversity = obs[0]           # [0,1] population spread
        stagnation = obs[16]         # [0,1] how stuck we are
        progress = obs[20]           # [0,1] NFE progress
        
        # Start with baseline values
        K = self.base_K
        kf = self.base_kf
        kr = self.base_kr
        p = self.base_p
        
        # =====================================================================
        # Only adjust in extreme stagnation situations
        # =====================================================================
        
        # Severe stagnation with low diversity -> need escape
        if stagnation > 0.9 and diversity < 0.05:
            # Reduce K to favor junior (exploration) phase
            K = max(R.K_MIN, self.base_K * 0.7)
            # Increase step size slightly
            kf = min(R.KF_MAX, self.base_kf * 1.15)
        
        # Moderate stagnation
        elif stagnation > 0.7 and diversity < 0.1:
            K = max(R.K_MIN, self.base_K * 0.85)
            kf = min(R.KF_MAX, self.base_kf * 1.05)
        
        # Late stage with good progress -> slight exploitation boost
        elif progress > 0.9 and stagnation < 0.3:
            K = min(R.K_MAX, self.base_K * 1.1)
        
        return {
            "K": float(K),
            "kf": float(kf),
            "kr": float(kr),
            "p": float(p),
        }


# =============================================================================
# SB3 PPO Controller
# =============================================================================

class SB3PPOController(BaseController):
    """
    Stable-Baselines3 PPO policy controller.
    
    Loads a trained PPO model and uses it for parameter control.
    Lazy imports SB3 to keep it optional.
    """
    
    def __init__(self, policy_path: str, device: str = "auto"):
        """
        Initialize SB3 controller.
        
        Parameters
        ----------
        policy_path : str
            Path to saved PPO model (.zip)
        device : str
            Device for inference ("auto", "cpu", "cuda", "mps")
        """
        super().__init__()
        self.policy_path = policy_path
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Lazy load the SB3 model."""
        if self._model is None:
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(self.policy_path, device=self.device)
            except ImportError:
                raise ImportError(
                    "stable-baselines3 is required for SB3PPOController. "
                    "Install with: pip install stable-baselines3"
                )
    
    def reset(self, dim: int, pop_size: int, max_nfes: int) -> None:
        """Reset and ensure model is loaded."""
        super().reset(dim, pop_size, max_nfes)
        self._load_model()
    
    def act(self, obs: np.ndarray, info: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get action from trained PPO policy.
        
        Parameters
        ----------
        obs : np.ndarray
            25-dim FLA feature vector
        info : dict, optional
            Additional info (not used by PPO)
        
        Returns
        -------
        dict
            {"K": float, "kf": float, "kr": float, "p": float}
        """
        self._load_model()
        
        # PPO expects float32
        obs = np.asarray(obs, dtype=np.float32)
        
        # Get action from policy (deterministic for evaluation)
        action, _ = self._model.predict(obs, deterministic=True)
        
        # Map action to parameters
        return map_action_to_params(action)


# =============================================================================
# Controller Factory
# =============================================================================

def create_controller(
    backend: str = "fixed",
    policy_path: Optional[str] = None,
    K: float = PARAM_RANGES.K_DEFAULT,
    kf: float = PARAM_RANGES.KF_DEFAULT,
    kr: float = PARAM_RANGES.KR_DEFAULT,
    p: float = PARAM_RANGES.P_DEFAULT,
    device: str = "auto",
) -> BaseController:
    """
    Create a controller based on the specified backend.
    
    Parameters
    ----------
    backend : str
        "fixed" | "heuristic" | "sb3"
    policy_path : str, optional
        Path to SB3 model (required for "sb3" backend)
    K, kf, kr, p : float
        Default/base parameters
    device : str
        Device for SB3 inference
    
    Returns
    -------
    BaseController
    """
    if backend == "fixed":
        return FixedController(K=K, kf=kf, kr=kr, p=p)
    
    elif backend == "heuristic":
        return HeuristicController(base_K=K, base_kf=kf, base_kr=kr, base_p=p)
    
    elif backend == "sb3":
        if policy_path is None:
            raise ValueError("policy_path required for sb3 backend")
        return SB3PPOController(policy_path=policy_path, device=device)
    
    else:
        raise ValueError(f"Unknown controller backend: {backend}")
