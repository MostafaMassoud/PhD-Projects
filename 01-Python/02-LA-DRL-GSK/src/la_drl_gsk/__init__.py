"""
LA-DRL-GSK: Landscape-Aware Deep Reinforcement Learning GSK Algorithm
======================================================================

Q1 Implementation:
- Windowed control: RL decision every W generations
- Action space: {K, kf, kr, p} absolute parameters
- State: 25 FLA features from ZeroCostLandscapeAnalyzer
- Controllers: fixed, heuristic, sb3 (stable-baselines3)

Usage (Baseline GSK - no RL):
    from la_drl_gsk import LADRLGSK, LADRLGSKConfig
    
    config = LADRLGSKConfig(dim=30, use_rl=False)
    optimizer = LADRLGSK(config)
    result = optimizer.optimize(objective_function)

Usage (Heuristic Controller - no external deps):
    config = LADRLGSKConfig(dim=30, use_rl=True, controller_backend="heuristic")
    optimizer = LADRLGSK(config)
    result = optimizer.optimize(objective_function)

Usage (SB3 PPO Policy - requires stable-baselines3):
    config = LADRLGSKConfig(
        dim=30, 
        use_rl=True, 
        controller_backend="sb3",
        policy_path='models/ppo_gsk.zip'
    )
    optimizer = LADRLGSK(config)
    result = optimizer.optimize(objective_function)

Author: LA-DRL-GSK Research Team
Date: 2025
"""

__version__ = '2.0.0'
__author__ = 'LA-DRL-GSK Research Team'

# Core (NumPy only)
from .la_drl_gsk import (
    LADRLGSK,
    LADRLGSKConfig,
    LADRLGSKResult,
    GSKRunState,
    create_baseline_gsk,
    create_ladrl_gsk,
    configure_threads,
    get_optimal_device,
    get_platform_info,
)

from .landscape_analyzer import (
    ZeroCostLandscapeAnalyzer,
    LandscapeFeatures,
    FeatureGroupAnalyzer,
)

from .controllers import (
    BaseController,
    FixedController,
    HeuristicController,
    create_controller,
    map_action_to_params,
    params_to_action,
    PARAM_RANGES,
)

from .logger import (
    OptimizationLogger,
    Colors,
    GenerationStats,
    OptimizationLog,
    format_scientific,
    format_time,
    progress_bar,
    print_info,
    print_warning,
    print_error,
    print_success,
)

from .cec2017_benchmark import (
    get_cec2017_function,
    get_benchmark_info,
    CEC2017_FUNCTIONS,
    DIMENSIONS,
)

__all__ = [
    # Main optimizer (NumPy)
    'LADRLGSK',
    'LADRLGSKConfig',
    'LADRLGSKResult',
    'GSKRunState',
    'create_baseline_gsk',
    'create_ladrl_gsk',
    'configure_threads',
    'get_optimal_device',
    'get_platform_info',
    
    # Controllers
    'BaseController',
    'FixedController',
    'HeuristicController',
    'create_controller',
    'map_action_to_params',
    'params_to_action',
    'PARAM_RANGES',
    
    # Landscape analysis (NumPy)
    'ZeroCostLandscapeAnalyzer',
    'LandscapeFeatures',
    'FeatureGroupAnalyzer',
    
    # Logging
    'OptimizationLogger',
    'Colors',
    
    # Benchmarks
    'get_cec2017_function',
    'get_benchmark_info',
    'CEC2017_FUNCTIONS',
    'DIMENSIONS',
]

# Optional Gymnasium environment (lazy import)
def get_gsk_env():
    """Import GSK control environment (requires gymnasium)."""
    from .gsk_env import GSKControlEnv, make_gsk_env, BenchmarkSuite
    return GSKControlEnv, make_gsk_env, BenchmarkSuite

# Optional SB3 controller (lazy import)
def get_sb3_controller():
    """Import SB3 PPO controller (requires stable-baselines3)."""
    from .controllers import SB3PPOController
    return SB3PPOController

# Keep backward compatibility for visualization
try:
    from .visualization import *
except ImportError:
    pass
