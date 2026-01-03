"""
LA-DRL-GSK Logging and Display Utilities
========================================

Eye-friendly console output with detailed parameter tracking.

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import numpy as np
import time
import platform
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


# =============================================================================
# Console Colors (Cross-platform)
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    
    # Check if colors are supported
    ENABLED = True  # Enable by default, user can disable
    
    # Colors
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    
    # Foreground
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Background
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    
    @classmethod
    def enable(cls):
        """Force enable colors."""
        cls.ENABLED = True
        cls._update_codes(True)
    
    @classmethod
    def disable(cls):
        """Force disable colors."""
        cls.ENABLED = False
        cls._update_codes(False)
    
    @classmethod
    def _update_codes(cls, enabled: bool):
        """Update color codes based on enabled flag."""
        e = enabled
        cls.RESET = '\033[0m' if e else ''
        cls.BOLD = '\033[1m' if e else ''
        cls.DIM = '\033[2m' if e else ''
        cls.UNDERLINE = '\033[4m' if e else ''
        cls.RED = '\033[91m' if e else ''
        cls.GREEN = '\033[92m' if e else ''
        cls.YELLOW = '\033[93m' if e else ''
        cls.BLUE = '\033[94m' if e else ''
        cls.MAGENTA = '\033[95m' if e else ''
        cls.CYAN = '\033[96m' if e else ''
        cls.WHITE = '\033[97m' if e else ''
        cls.GRAY = '\033[90m' if e else ''


# =============================================================================
# Formatting Helpers
# =============================================================================

def format_scientific(value: float, precision: int = 4) -> str:
    """Format number in scientific notation."""
    if abs(value) < 1e-10:
        return f"{0.0:.{precision}e}"
    return f"{value:.{precision}e}"


def format_float(value: float, precision: int = 4) -> str:
    """Format float with fixed precision."""
    return f"{value:.{precision}f}"


def format_percent(value: float, precision: int = 1) -> str:
    """Format as percentage."""
    return f"{value * 100:.{precision}f}%"


def format_time(seconds: float) -> str:
    """Format time duration."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def progress_bar(current: int, total: int, width: int = 30, 
                 fill: str = '█', empty: str = '░') -> str:
    """Create a progress bar string."""
    progress = current / max(1, total)
    filled = int(width * progress)
    bar = fill * filled + empty * (width - filled)
    return f"[{bar}] {progress*100:5.1f}%"


def feature_bar(value: float, width: int = 10) -> str:
    """Create a mini bar for feature visualization."""
    C = Colors
    filled = int(width * min(1.0, max(0.0, value)))
    
    # Color based on value
    if value > 0.8:
        color = C.RED
    elif value > 0.6:
        color = C.YELLOW
    elif value > 0.4:
        color = C.WHITE
    elif value > 0.2:
        color = C.CYAN
    else:
        color = C.GREEN
    
    bar = '▓' * filled + '░' * (width - filled)
    return f"{color}{bar}{C.RESET}"


def box_line(char: str = '─', width: int = 80) -> str:
    """Create a box line."""
    return char * width


def box_header(title: str, width: int = 80, char: str = '═') -> str:
    """Create a box header with title."""
    padding = (width - len(title) - 2) // 2
    return f"{char * padding} {title} {char * (width - padding - len(title) - 2)}"


def section_header(title: str, width: int = 80) -> str:
    """Create a section header."""
    C = Colors
    return f"{C.CYAN}{C.BOLD}{'─' * 3} {title} {'─' * (width - len(title) - 5)}{C.RESET}"


# =============================================================================
# Statistics Tracker
# =============================================================================

@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    generation: int
    nfes: int
    best_f: float
    mean_f: float
    std_f: float
    min_f: float
    max_f: float
    improvement: float
    improvement_abs: float
    p_junior: float
    KF: float
    KR: float
    K: float
    strategy: str
    features: Optional[Dict[str, float]] = None
    runtime: float = 0.0
    
    # Population stats
    pop_diversity: float = 0.0
    pop_spread: float = 0.0
    
    # Stagnation tracking
    stagnation_count: int = 0


@dataclass 
class OptimizationLog:
    """Complete optimization run log."""
    config: Dict[str, Any] = field(default_factory=dict)
    generations: List[GenerationStats] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    final_best_f: float = float('inf')
    final_best_x: Optional[np.ndarray] = None
    total_nfes: int = 0
    
    def add_generation(self, stats: GenerationStats):
        """Add generation statistics."""
        self.generations.append(stats)
        self.final_best_f = stats.best_f
        self.total_nfes = stats.nfes
    
    @property
    def total_runtime(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def convergence_history(self) -> np.ndarray:
        return np.array([g.best_f for g in self.generations])


# =============================================================================
# Feature Names and Groups
# =============================================================================

FEATURE_NAMES = [
    # Group 1: Population (0-4)
    'diversity', 'spread', 'centroid_offset', 'variance', 'elite_clustering',
    # Group 2: Fitness (5-9)
    'fit_range', 'fit_cv', 'skewness', 'kurtosis', 'elite_gap',
    # Group 3: Correlation (10-14)
    'fdc', 'fdc_centroid', 'kendall_tau', 'separability', 'neighbor_corr',
    # Group 4: Temporal (15-19)
    'improvement_rate', 'stagnation', 'diversity_trend', 'mean_trend', 'consistency',
    # Group 5: Progress (20-24)
    'progress', 'convergence_ratio', 'exploitation', 'improvement_potential', 'quality',
]

FEATURE_GROUPS = {
    'Population': (0, 5),
    'Fitness': (5, 10),
    'Correlation': (10, 15),
    'Temporal': (15, 20),
    'Progress': (20, 25),
}

FEATURE_DESCRIPTIONS = {
    'diversity': 'Population spread in search space',
    'spread': 'Range of population positions',
    'centroid_offset': 'Distance of centroid from origin',
    'variance': 'Population position variance',
    'elite_clustering': 'How clustered are top solutions',
    'fit_range': 'Fitness value range (normalized)',
    'fit_cv': 'Coefficient of variation of fitness',
    'skewness': 'Fitness distribution skewness',
    'kurtosis': 'Fitness distribution kurtosis',
    'elite_gap': 'Gap between best and median fitness',
    'fdc': 'Fitness-distance correlation',
    'fdc_centroid': 'FDC to population centroid',
    'kendall_tau': 'Rank correlation measure',
    'separability': 'Problem separability estimate',
    'neighbor_corr': 'Nearest neighbor correlation',
    'improvement_rate': 'Recent improvement speed',
    'stagnation': 'Generations without improvement',
    'diversity_trend': 'Diversity change direction',
    'mean_trend': 'Mean fitness change direction',
    'consistency': 'Improvement consistency',
    'progress': 'Optimization progress (NFEs used)',
    'convergence_ratio': 'Convergence speed estimate',
    'exploitation': 'Exploitation vs exploration',
    'improvement_potential': 'Remaining improvement estimate',
    'quality': 'Current solution quality',
}


# =============================================================================
# Logger Class
# =============================================================================

class OptimizationLogger:
    """
    Eye-friendly logging for LA-DRL-GSK optimization.
    
    Verbosity levels:
    - 0: Silent
    - 1: Summary only
    - 2: Generation progress
    - 3: Detailed with features
    - 4: Full debug with all parameters
    """
    
    def __init__(
        self,
        verbosity: int = 2,
        log_interval: int = 10,
        show_features: bool = True,
        show_params: bool = True,
        show_population: bool = False,
        width: int = 90,
        use_colors: bool = True,
    ):
        """
        Initialize logger.
        
        Parameters
        ----------
        verbosity : int
            0=silent, 1=summary, 2=generations, 3=detailed, 4=debug
        log_interval : int
            Log every N generations (for verbosity >= 2)
        show_features : bool
            Show FLA features
        show_params : bool
            Show GSK parameters
        show_population : bool
            Show population statistics
        width : int
            Console width
        use_colors : bool
            Use colored output
        """
        self.verbosity = verbosity
        self.log_interval = log_interval
        self.show_features = show_features
        self.show_params = show_params
        self.show_population = show_population
        self.width = width
        
        if use_colors:
            Colors.enable()
        else:
            Colors.disable()
        
        self.log = OptimizationLog()
        self._last_best_f = float('inf')
        self._stagnation_count = 0
        self._best_improvement_gen = 0
        self._total_improvements = 0
    
    def print_header(self, config: Dict[str, Any]):
        """Print optimization header with full configuration."""
        self.log.config = config
        self.log.start_time = time.time()
        
        if self.verbosity < 1:
            return
        
        C = Colors
        W = self.width
        
        print()
        print(f"{C.CYAN}{C.BOLD}{box_header('LA-DRL-GSK OPTIMIZATION', W)}{C.RESET}")
        print()
        
        # ─────────────────────────────────────────────────────────────────
        # Platform Information
        # ─────────────────────────────────────────────────────────────────
        print(section_header("PLATFORM", W))
        print()
        
        info = [
            ("System", f"{platform.system()} ({platform.machine()})"),
            ("Python", platform.python_version()),
        ]
        
        try:
            import torch
            info.append(("PyTorch", torch.__version__))
            
            if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                if torch.backends.mps.is_available():
                    info.append(("Backend", f"{C.GREEN}MPS (Metal){C.RESET}"))
                else:
                    info.append(("Backend", "CPU"))
            else:
                if torch.backends.mkl.is_available():
                    info.append(("Backend", f"{C.GREEN}MKL (Intel){C.RESET}"))
                else:
                    info.append(("Backend", "CPU"))
            info.append(("Threads", str(torch.get_num_threads())))
        except ImportError:
            info.append(("PyTorch", f"{C.YELLOW}Not installed{C.RESET}"))
        
        for label, value in info:
            print(f"  {C.BOLD}{label:12}{C.RESET} {value}")
        
        print()
        
        # ─────────────────────────────────────────────────────────────────
        # Problem Configuration
        # ─────────────────────────────────────────────────────────────────
        print(section_header("PROBLEM CONFIGURATION", W))
        print()
        
        dim = config.get('dim', '?')
        pop_size = config.get('pop_size', 100)
        max_nfes = config.get('max_nfes', 10000 * dim if isinstance(dim, int) else '?')
        bounds = config.get('bounds', (-100, 100))
        seed = config.get('seed', '?')
        
        print(f"  {C.BOLD}{'Dimension':<16}{C.RESET} {dim}")
        print(f"  {C.BOLD}{'Population':<16}{C.RESET} {pop_size}")
        print(f"  {C.BOLD}{'Max NFEs':<16}{C.RESET} {max_nfes:,}")
        print(f"  {C.BOLD}{'Max Generations':<16}{C.RESET} {max_nfes // pop_size if isinstance(max_nfes, int) else '?'}")
        print(f"  {C.BOLD}{'Bounds':<16}{C.RESET} [{bounds[0]}, {bounds[1]}]")
        print(f"  {C.BOLD}{'Seed':<16}{C.RESET} {seed}")
        
        print()
        
        # ─────────────────────────────────────────────────────────────────
        # GSK Parameters
        # ─────────────────────────────────────────────────────────────────
        print(section_header("GSK PARAMETERS", W))
        print()
        
        KF = config.get('KF', 0.5)
        KR = config.get('KR', 0.9)
        K = config.get('K', 10.0)
        
        print(f"  {C.BOLD}{'KF (step size)':<20}{C.RESET} {KF:.4f}  {C.DIM}[0.1 - 0.9]{C.RESET}")
        print(f"  {C.BOLD}{'KR (crossover)':<20}{C.RESET} {KR:.4f}  {C.DIM}[0.5 - 1.0]{C.RESET}")
        print(f"  {C.BOLD}{'K (decay rate)':<20}{C.RESET} {K:.4f}  {C.DIM}[1.0 - 20.0]{C.RESET}")
        
        # Show decay formula
        print()
        print(f"  {C.DIM}Decay formula: D_junior = ceil(D × (1 - g/G_max)^K){C.RESET}")
        
        print()
        
        # ─────────────────────────────────────────────────────────────────
        # Mode
        # ─────────────────────────────────────────────────────────────────
        print(section_header("OPTIMIZATION MODE", W))
        print()
        
        use_rl = config.get('use_rl', False)
        policy_path = config.get('policy_path')
        
        if use_rl:
            if policy_path:
                print(f"  {C.BOLD}Mode:{C.RESET}         {C.GREEN}Neural Policy (Trained){C.RESET}")
                print(f"  {C.BOLD}Policy:{C.RESET}       {policy_path}")
            else:
                print(f"  {C.BOLD}Mode:{C.RESET}         {C.CYAN}Adaptive Heuristic{C.RESET}")
                print(f"  {C.DIM}  Uses FLA features to adapt parameters{C.RESET}")
        else:
            print(f"  {C.BOLD}Mode:{C.RESET}         {C.YELLOW}Baseline GSK{C.RESET}")
            print(f"  {C.DIM}  Standard GSK with fixed decay schedule{C.RESET}")
        
        print()
        print(f"{C.DIM}{box_line('─', W)}{C.RESET}")
        print()
        
        if self.verbosity >= 2:
            self._print_generation_header()
    
    def _print_generation_header(self):
        """Print header for generation logs."""
        C = Colors
        
        # Main header
        cols = [
            ('Gen', 5),
            ('NFEs', 9),
            ('Best', 13),
            ('Mean±Std', 22),
            ('Improv', 10),
        ]
        
        if self.show_params:
            cols.extend([
                ('p_jun', 6),
                ('KF', 5),
                ('KR', 5),
                ('Strategy', 10),
            ])
        
        header = ' │ '.join(f"{name:>{width}}" for name, width in cols)
        print(f"{C.BOLD}{header}{C.RESET}")
        
        separator = '─┼─'.join('─' * width for _, width in cols)
        print(f"{C.DIM}{separator}{C.RESET}")
    
    def log_generation(
        self,
        generation: int,
        nfes: int,
        max_nfes: int,
        fitness: np.ndarray,
        best_f: float,
        p_junior: float,
        KF: float,
        KR: float,
        strategy: str,
        features: Optional[np.ndarray] = None,
        action: Optional[Dict] = None,
        K: float = 10.0,
    ):
        """Log a generation with full statistics."""
        C = Colors
        
        # Calculate stats
        mean_f = np.mean(fitness)
        std_f = np.std(fitness)
        min_f = np.min(fitness)
        max_f = np.max(fitness)
        
        # Improvement tracking
        if best_f < self._last_best_f:
            improvement_abs = self._last_best_f - best_f
            improvement = improvement_abs / max(abs(self._last_best_f), 1e-10)
            self._stagnation_count = 0
            self._best_improvement_gen = generation
            self._total_improvements += 1
        else:
            improvement = 0.0
            improvement_abs = 0.0
            self._stagnation_count += 1
        self._last_best_f = best_f
        
        # Feature dict
        feature_dict = None
        if features is not None:
            feature_dict = {FEATURE_NAMES[i]: float(features[i]) for i in range(min(len(features), len(FEATURE_NAMES)))}
        
        # Store stats
        stats = GenerationStats(
            generation=generation,
            nfes=nfes,
            best_f=best_f,
            mean_f=mean_f,
            std_f=std_f,
            min_f=min_f,
            max_f=max_f,
            improvement=improvement,
            improvement_abs=improvement_abs,
            p_junior=p_junior,
            KF=KF,
            KR=KR,
            K=K,
            strategy=strategy,
            features=feature_dict,
            stagnation_count=self._stagnation_count,
            pop_diversity=features[0] if features is not None else 0.0,
            pop_spread=features[1] if features is not None else 0.0,
        )
        self.log.add_generation(stats)
        
        # Print based on verbosity and interval
        should_print = (
            self.verbosity >= 2 and 
            (generation % self.log_interval == 0 or generation == 1)
        )
        
        if should_print:
            self._print_generation(stats, nfes, max_nfes, features)
    
    def _print_generation(
        self, 
        stats: GenerationStats, 
        nfes: int, 
        max_nfes: int,
        features: Optional[np.ndarray] = None
    ):
        """Print generation line with all details."""
        C = Colors
        
        # Color based on improvement/stagnation
        if stats.improvement > 0.01:
            best_color = C.GREEN
        elif self._stagnation_count > 50:
            best_color = C.RED
        elif self._stagnation_count > 20:
            best_color = C.YELLOW
        else:
            best_color = ''
        
        # Format improvement
        if stats.improvement > 0:
            improv_str = f"{C.GREEN}↓{format_percent(stats.improvement):>7}{C.RESET}"
        elif self._stagnation_count > 5:
            improv_str = f"{C.YELLOW}→{self._stagnation_count:>4}gen{C.RESET}"
        else:
            improv_str = f"{C.DIM}   ─────{C.RESET}"
        
        # Mean ± Std
        mean_std = f"{format_scientific(stats.mean_f)} ± {format_scientific(stats.std_f)}"
        
        # Strategy color
        strat_colors = {
            'standard': C.WHITE,
            'aggressive': C.RED,
            'conservative': C.BLUE,
        }
        strat_color = strat_colors.get(stats.strategy, '')
        
        # Build line
        line_parts = [
            f"{stats.generation:>5}",
            f"{nfes:>9,}",
            f"{best_color}{format_scientific(stats.best_f):>13}{C.RESET}",
            f"{mean_std:>22}",
            f"{improv_str:>10}",
        ]
        
        if self.show_params:
            line_parts.extend([
                f"{stats.p_junior:>6.3f}",
                f"{stats.KF:>5.2f}",
                f"{stats.KR:>5.2f}",
                f"{strat_color}{stats.strategy:>10}{C.RESET}",
            ])
        
        print(' │ '.join(line_parts))
        
        # Show features if enabled and verbosity >= 3
        if self.show_features and features is not None and self.verbosity >= 3:
            self._print_features_detailed(features)
    
    def _print_features_detailed(self, features: np.ndarray):
        """Print FLA features in detailed format with visual bars."""
        C = Colors
        
        print()
        for group_name, (start, end) in FEATURE_GROUPS.items():
            print(f"  {C.CYAN}{group_name:12}{C.RESET} ", end='')
            
            for i in range(start, end):
                if i < len(features):
                    val = features[i]
                    name = FEATURE_NAMES[i][:8]
                    bar = feature_bar(val, 8)
                    print(f"{name:>8}:{bar}{val:.2f} ", end='')
            print()
        print()
    
    def print_summary(self, result=None):
        """Print comprehensive optimization summary."""
        self.log.end_time = time.time()
        
        if self.verbosity < 1:
            return
        
        C = Colors
        W = self.width
        
        print()
        print(f"{C.DIM}{box_line('─', W)}{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}{box_header('OPTIMIZATION COMPLETE', W)}{C.RESET}")
        print()
        
        # ─────────────────────────────────────────────────────────────────
        # Final Results
        # ─────────────────────────────────────────────────────────────────
        print(section_header("RESULTS", W))
        print()
        
        print(f"  {C.BOLD}{'Best Fitness':<20}{C.RESET} {C.GREEN}{format_scientific(self.log.final_best_f)}{C.RESET}")
        print(f"  {C.BOLD}{'Total NFEs':<20}{C.RESET} {self.log.total_nfes:,}")
        print(f"  {C.BOLD}{'Generations':<20}{C.RESET} {len(self.log.generations)}")
        print(f"  {C.BOLD}{'Runtime':<20}{C.RESET} {format_time(self.log.total_runtime)}")
        print(f"  {C.BOLD}{'NFEs/second':<20}{C.RESET} {self.log.total_nfes / max(0.001, self.log.total_runtime):,.0f}")
        
        if result is not None and hasattr(result, 'best_x') and result.best_x is not None:
            x = result.best_x
            if len(x) <= 5:
                x_str = ', '.join(f"{xi:.4f}" for xi in x)
            else:
                x_str = f"{x[0]:.4f}, {x[1]:.4f}, ..., {x[-1]:.4f}"
            print(f"  {C.BOLD}{'Best Solution':<20}{C.RESET} [{x_str}]")
        
        print()
        
        # ─────────────────────────────────────────────────────────────────
        # Convergence Analysis
        # ─────────────────────────────────────────────────────────────────
        if len(self.log.generations) > 1:
            print(section_header("CONVERGENCE ANALYSIS", W))
            print()
            
            history = self.log.convergence_history
            
            print(f"  {C.BOLD}{'Initial Fitness':<20}{C.RESET} {format_scientific(history[0])}")
            print(f"  {C.BOLD}{'Final Fitness':<20}{C.RESET} {format_scientific(history[-1])}")
            
            total_improv = history[0] - history[-1]
            if abs(history[0]) > 1e-10:
                rel_improv = total_improv / abs(history[0]) * 100
                print(f"  {C.BOLD}{'Improvement':<20}{C.RESET} {C.GREEN}{rel_improv:.2f}%{C.RESET}")
            
            print(f"  {C.BOLD}{'Total Improvements':<20}{C.RESET} {self._total_improvements}")
            print(f"  {C.BOLD}{'Best Found at Gen':<20}{C.RESET} {self._best_improvement_gen}")
            
            # Find convergence milestones
            if total_improv > 0:
                for pct in [50, 90, 99]:
                    threshold = history[0] - (pct / 100) * total_improv
                    idx = np.argmax(history <= threshold)
                    if idx > 0 or history[0] <= threshold:
                        gen_pct = idx / len(history) * 100
                        print(f"  {C.BOLD}{f'{pct}% at Gen':<20}{C.RESET} {idx} ({gen_pct:.1f}% of budget)")
            
            print()
        
        # ─────────────────────────────────────────────────────────────────
        # Parameter Statistics
        # ─────────────────────────────────────────────────────────────────
        if self.log.generations and self.show_params:
            print(section_header("PARAMETER STATISTICS", W))
            print()
            
            p_juniors = [g.p_junior for g in self.log.generations]
            kfs = [g.KF for g in self.log.generations]
            krs = [g.KR for g in self.log.generations]
            strategies = [g.strategy for g in self.log.generations]
            
            # Parameter table
            print(f"  {'Parameter':<12} │ {'Min':>8} │ {'Max':>8} │ {'Mean':>8} │ {'Std':>8}")
            print(f"  {'─'*12}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")
            
            for name, values in [('p_junior', p_juniors), ('KF', kfs), ('KR', krs)]:
                print(f"  {name:<12} │ {min(values):>8.4f} │ {max(values):>8.4f} │ {np.mean(values):>8.4f} │ {np.std(values):>8.4f}")
            
            print()
            
            # Strategy distribution
            print(f"  {C.BOLD}Strategy Distribution:{C.RESET}")
            strat_counts = {}
            for s in strategies:
                strat_counts[s] = strat_counts.get(s, 0) + 1
            
            for strat, count in sorted(strat_counts.items(), key=lambda x: -x[1]):
                pct = count / len(strategies) * 100
                bar_width = int(30 * pct / 100)
                bar = '█' * bar_width + '░' * (30 - bar_width)
                strat_color = {'standard': C.WHITE, 'aggressive': C.RED, 'conservative': C.BLUE}.get(strat, '')
                print(f"    {strat_color}{strat:<12}{C.RESET} [{bar}] {pct:>5.1f}% ({count})")
            
            print()
        
        # ─────────────────────────────────────────────────────────────────
        # Feature Statistics (if available)
        # ─────────────────────────────────────────────────────────────────
        if self.log.generations and self.show_features and self.log.generations[0].features:
            print(section_header("FLA FEATURE STATISTICS", W))
            print()
            
            # Collect feature stats
            feature_data = {name: [] for name in FEATURE_NAMES}
            for gen in self.log.generations:
                if gen.features:
                    for name, val in gen.features.items():
                        if name in feature_data:
                            feature_data[name].append(val)
            
            print(f"  {'Feature':<18} │ {'Min':>6} │ {'Max':>6} │ {'Mean':>6} │ {'Visual'}")
            print(f"  {'─'*18}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*6}─┼─{'─'*20}")
            
            for name in FEATURE_NAMES:
                if feature_data[name]:
                    vals = feature_data[name]
                    mean_val = np.mean(vals)
                    bar = feature_bar(mean_val, 15)
                    print(f"  {name:<18} │ {min(vals):>6.3f} │ {max(vals):>6.3f} │ {mean_val:>6.3f} │ {bar}")
            
            print()
        
        print(f"{C.DIM}{box_line('═', W)}{C.RESET}")
        print()
    
    def print_comparison(
        self,
        name1: str, results1: List[float],
        name2: str, results2: List[float],
        f_opt: float = 0.0,
    ):
        """Print comparison between two algorithms."""
        C = Colors
        W = self.width
        
        errors1 = [abs(r - f_opt) for r in results1]
        errors2 = [abs(r - f_opt) for r in results2]
        
        mean1, std1 = np.mean(errors1), np.std(errors1)
        mean2, std2 = np.mean(errors2), np.std(errors2)
        
        print()
        print(f"{C.CYAN}{C.BOLD}{box_header('ALGORITHM COMPARISON', W)}{C.RESET}")
        print()
        
        # Header
        print(f"  {'Algorithm':<20} │ {'Mean Error':>14} │ {'Std':>12} │ {'Best':>12} │ {'Worst':>12}")
        print(f"  {'─'*20}─┼─{'─'*14}─┼─{'─'*12}─┼─{'─'*12}─┼─{'─'*12}")
        
        # Results with colors
        better1 = mean1 < mean2
        c1 = C.GREEN if better1 else ''
        c2 = C.GREEN if not better1 else ''
        
        print(f"  {c1}{name1:<20}{C.RESET} │ {format_scientific(mean1):>14} │ {format_scientific(std1):>12} │ {format_scientific(min(errors1)):>12} │ {format_scientific(max(errors1)):>12}")
        print(f"  {c2}{name2:<20}{C.RESET} │ {format_scientific(mean2):>14} │ {format_scientific(std2):>12} │ {format_scientific(min(errors2)):>12} │ {format_scientific(max(errors2)):>12}")
        
        print()
        
        # Statistical comparison
        if mean1 > 1e-10:
            rel_diff = (mean1 - mean2) / mean1 * 100
            if abs(rel_diff) < 1:
                print(f"  {C.YELLOW}⚖ Results are equivalent (difference < 1%){C.RESET}")
            elif rel_diff > 0:
                print(f"  {C.GREEN}✓ {name2} is {abs(rel_diff):.1f}% better than {name1}{C.RESET}")
            else:
                print(f"  {C.GREEN}✓ {name1} is {abs(rel_diff):.1f}% better than {name2}{C.RESET}")
        
        print()


# =============================================================================
# Quick Logging Functions
# =============================================================================

def print_run_header(run: int, total: int, seed: int):
    """Print header for a single run."""
    C = Colors
    print(f"{C.BOLD}Run {run}/{total}{C.RESET} (seed={seed})")


def print_run_result(run: int, error: float, nfes: int, runtime: float):
    """Print result of a single run."""
    C = Colors
    print(f"  {C.GREEN}✓{C.RESET} error={format_scientific(error)}, NFEs={nfes:,}, time={format_time(runtime)}")


def print_error(message: str):
    """Print error message."""
    C = Colors
    print(f"{C.RED}✗ Error: {message}{C.RESET}")


def print_warning(message: str):
    """Print warning message."""
    C = Colors
    print(f"{C.YELLOW}⚠ Warning: {message}{C.RESET}")


def print_info(message: str):
    """Print info message."""
    C = Colors
    print(f"{C.CYAN}ℹ {message}{C.RESET}")


def print_success(message: str):
    """Print success message."""
    C = Colors
    print(f"{C.GREEN}✓ {message}{C.RESET}")
