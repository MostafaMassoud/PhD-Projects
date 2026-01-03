"""
Zero-Cost Fitness Landscape Analysis for LA-DRL-GSK
====================================================

This module implements fitness landscape analysis using ONLY existing 
population data - NO additional function evaluations are performed.

All 25 features are computed from:
- Current population positions (N × D matrix)
- Current fitness values (N vector)  
- Historical data from previous generations

Reference:
    Malan, K. M., & Engelbrecht, A. P. (2013). A survey of techniques for 
    characterising fitness landscapes and some possible ways forward.
    Information Sciences, 241, 148-163.

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class LandscapeFeatures:
    """Container for computed landscape features with named access."""
    # Population Distribution (5)
    diversity: float
    spread: float
    centroid_offset: float
    pop_variance: float
    elite_clustering: float
    
    # Fitness Distribution (5)
    fit_range: float
    fit_cv: float
    skewness: float
    kurtosis: float
    elite_gap: float
    
    # Correlation Features (5)
    fdc: float
    fdc_centroid: float
    tau: float
    separability: float
    neighbor_correlation: float
    
    # Temporal Features (5)
    improvement_rate: float
    stagnation: float
    div_trend: float
    mean_trend: float
    consistency: float
    
    # Progress Features (5)
    progress: float
    convergence_ratio: float
    exploitation_mode: float
    improvement_potential: float
    pop_quality: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for neural network input."""
        return np.array([
            # Population Distribution
            self.diversity, self.spread, self.centroid_offset,
            self.pop_variance, self.elite_clustering,
            # Fitness Distribution
            self.fit_range, self.fit_cv, self.skewness,
            self.kurtosis, self.elite_gap,
            # Correlation Features
            self.fdc, self.fdc_centroid, self.tau,
            self.separability, self.neighbor_correlation,
            # Temporal Features
            self.improvement_rate, self.stagnation, self.div_trend,
            self.mean_trend, self.consistency,
            # Progress Features
            self.progress, self.convergence_ratio, self.exploitation_mode,
            self.improvement_potential, self.pop_quality,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Return ordered list of feature names."""
        return [
            'diversity', 'spread', 'centroid_offset', 'pop_variance', 'elite_clustering',
            'fit_range', 'fit_cv', 'skewness', 'kurtosis', 'elite_gap',
            'fdc', 'fdc_centroid', 'tau', 'separability', 'neighbor_correlation',
            'improvement_rate', 'stagnation', 'div_trend', 'mean_trend', 'consistency',
            'progress', 'convergence_ratio', 'exploitation_mode', 'improvement_potential', 'pop_quality',
        ]


class ZeroCostLandscapeAnalyzer:
    """
    Fitness Landscape Analysis using ONLY existing population data.
    
    NO additional function evaluations are performed.
    All features are derived from:
    - Current population positions
    - Current fitness values
    - Historical data from previous generations
    
    Parameters
    ----------
    dim : int
        Problem dimensionality
    pop_size : int
        Population size
    bounds : tuple
        (lower, upper) bounds for search space
    max_nfes : int
        Maximum function evaluations (for progress calculation)
    history_length : int
        Number of generations to keep in history buffer
    """
    
    def __init__(
        self, 
        dim: int, 
        pop_size: int, 
        bounds: Tuple[float, float] = (-100.0, 100.0),
        max_nfes: Optional[int] = None,
        history_length: int = 50
    ):
        self.dim = dim
        self.pop_size = pop_size
        self.bounds = bounds
        self.range = bounds[1] - bounds[0]
        self.max_nfes = max_nfes or 10000 * dim
        self.history_length = history_length
        
        # Historical storage (no NFE cost - just memory)
        self.fitness_history: deque = deque(maxlen=history_length)
        self.best_history: deque = deque(maxlen=history_length)
        self.diversity_history: deque = deque(maxlen=history_length)
        self.mean_history: deque = deque(maxlen=history_length)
        self.improvement_history: deque = deque(maxlen=history_length)
        
        # Generation counter
        self.generation = 0
        self.nfes_used = 0
        
        # Cache for expensive computations
        self._dist_matrix_cache: Optional[np.ndarray] = None
        self._last_pop_hash: Optional[int] = None
        
    def compute_state(
        self, 
        population: np.ndarray, 
        fitness: np.ndarray,
        nfes_used: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute 25 zero-cost FLA features.
        
        Parameters
        ----------
        population : np.ndarray
            Current population positions, shape (pop_size, dim)
        fitness : np.ndarray
            Current fitness values, shape (pop_size,)
        nfes_used : int, optional
            Current NFE count for progress calculation
            
        Returns
        -------
        np.ndarray
            25 normalized features as float32 array
        """
        self.generation += 1
        if nfes_used is not None:
            self.nfes_used = nfes_used
        
        # Compute distance matrix (cached if population unchanged)
        pop_hash = hash(population.tobytes())
        if pop_hash != self._last_pop_hash:
            if len(population) > 1:
                self._dist_matrix_cache = squareform(pdist(population))
            else:
                self._dist_matrix_cache = np.zeros((1, 1))
            self._last_pop_hash = pop_hash
        
        features = []
        
        # === GROUP 1: Population Distribution Features (5) ===
        features.extend(self._population_features(population, fitness))
        
        # === GROUP 2: Fitness Distribution Features (5) ===
        features.extend(self._fitness_features(fitness))
        
        # === GROUP 3: Correlation Features (5) ===
        features.extend(self._correlation_features(population, fitness))
        
        # === GROUP 4: Temporal/Historical Features (5) ===
        features.extend(self._temporal_features(fitness))
        
        # === GROUP 5: Search Progress Features (5) ===
        features.extend(self._progress_features(population, fitness))
        
        # Update history for next generation
        self._update_history(population, fitness)
        
        return np.array(features, dtype=np.float32)
    
    def compute_features_structured(
        self, 
        population: np.ndarray, 
        fitness: np.ndarray,
        nfes_used: Optional[int] = None
    ) -> LandscapeFeatures:
        """Compute features and return as structured dataclass."""
        arr = self.compute_state(population, fitness, nfes_used)
        return LandscapeFeatures(*arr)
    
    # =========================================================================
    # GROUP 1: Population Distribution Features (Zero Cost)
    # =========================================================================
    
    def _population_features(self, pop: np.ndarray, fit: np.ndarray) -> List[float]:
        """Features from population spatial distribution."""
        n = len(pop)
        dist_matrix = self._dist_matrix_cache
        
        # 1. Population diversity (mean pairwise distance)
        if n > 1:
            distances = pdist(pop)
            diversity = float(np.mean(distances) / (self.range * np.sqrt(self.dim)))
            diversity = np.clip(diversity, 0, 1)
        else:
            diversity = 0.0
        
        # 2. Population spread (max pairwise distance)
        if n > 1:
            spread = float(np.max(distances) / (self.range * np.sqrt(self.dim)))
            spread = np.clip(spread, 0, 1)
        else:
            spread = 0.0
        
        # 3. Population centroid distance from center
        centroid = np.mean(pop, axis=0)
        center = (self.bounds[0] + self.bounds[1]) / 2
        centroid_offset = float(np.linalg.norm(centroid - center) / (self.range * np.sqrt(self.dim)))
        centroid_offset = np.clip(centroid_offset, 0, 1)
        
        # 4. Population variance (average per-dimension variance)
        pop_variance = float(np.mean(np.var(pop, axis=0)) / (self.range ** 2))
        pop_variance = np.clip(pop_variance, 0, 1)
        
        # 5. Elite clustering (how close are top 20% to best)
        n_elite = max(1, n // 5)
        elite_idx = np.argsort(fit)[:n_elite]
        best_pos = pop[elite_idx[0]]
        if n_elite > 1:
            elite_distances = np.linalg.norm(pop[elite_idx[1:]] - best_pos, axis=1)
            elite_clustering = float(1.0 / (1.0 + np.mean(elite_distances) / self.range))
        else:
            elite_clustering = 1.0
        elite_clustering = np.clip(elite_clustering, 0, 1)
        
        return [diversity, spread, centroid_offset, pop_variance, elite_clustering]
    
    # =========================================================================
    # GROUP 2: Fitness Distribution Features (Zero Cost)
    # =========================================================================
    
    def _fitness_features(self, fit: np.ndarray) -> List[float]:
        """Features from fitness value distribution."""
        
        # 1. Fitness range (normalized)
        fit_range = (np.max(fit) - np.min(fit)) / (np.abs(np.mean(fit)) + 1e-10)
        fit_range = float(np.clip(fit_range, 0, 10) / 10)
        
        # 2. Fitness coefficient of variation
        fit_cv = np.std(fit) / (np.abs(np.mean(fit)) + 1e-10)
        fit_cv = float(np.clip(fit_cv, 0, 5) / 5)
        
        # 3. Fitness skewness
        fit_centered = fit - np.mean(fit)
        fit_std = np.std(fit) + 1e-10
        skewness = np.mean((fit_centered / fit_std) ** 3)
        skewness = float(np.clip(skewness, -3, 3) / 6 + 0.5)  # Normalize to [0, 1]
        
        # 4. Fitness kurtosis
        kurtosis = np.mean((fit_centered / fit_std) ** 4) - 3
        kurtosis = float(np.clip(kurtosis, -3, 10) / 13 + 3/13)  # Normalize to [0, 1]
        
        # 5. Elite-to-mean gap
        n_elite = max(1, len(fit) // 10)
        elite_mean = np.mean(np.sort(fit)[:n_elite])
        elite_gap = (np.mean(fit) - elite_mean) / (np.abs(np.mean(fit)) + 1e-10)
        elite_gap = float(np.clip(elite_gap, 0, 2) / 2)
        
        return [fit_range, fit_cv, skewness, kurtosis, elite_gap]
    
    # =========================================================================
    # GROUP 3: Correlation Features (Zero Cost)
    # =========================================================================
    
    def _correlation_features(self, pop: np.ndarray, fit: np.ndarray) -> List[float]:
        """Features from fitness-distance correlations."""
        n = len(fit)
        
        # 1. Fitness-Distance Correlation (FDC) to best
        best_idx = np.argmin(fit)
        best_pos = pop[best_idx]
        distances_to_best = np.linalg.norm(pop - best_pos, axis=1)
        
        mask = np.arange(n) != best_idx
        if np.sum(mask) > 2:
            fdc, _ = spearmanr(fit[mask], distances_to_best[mask])
            fdc = 0.0 if np.isnan(fdc) else float(fdc)
        else:
            fdc = 0.0
        fdc = (fdc + 1) / 2  # Normalize from [-1,1] to [0,1]
        
        # 2. Fitness-Distance Correlation to centroid
        centroid = np.mean(pop, axis=0)
        distances_to_centroid = np.linalg.norm(pop - centroid, axis=1)
        fdc_centroid, _ = spearmanr(fit, distances_to_centroid)
        fdc_centroid = 0.0 if np.isnan(fdc_centroid) else float(fdc_centroid)
        fdc_centroid = (fdc_centroid + 1) / 2
        
        # 3. Kendall's tau correlation
        if np.sum(mask) > 2:
            tau, _ = kendalltau(fit[mask], distances_to_best[mask])
            tau = 0.0 if np.isnan(tau) else float(tau)
        else:
            tau = 0.0
        tau = (tau + 1) / 2
        
        # 4. Dimension-fitness correlation (separability indicator)
        dim_correlations = []
        sample_dims = min(self.dim, 20)  # Cap for efficiency
        for d in range(sample_dims):
            corr, _ = spearmanr(pop[:, d], fit)
            if not np.isnan(corr):
                dim_correlations.append(abs(corr))
        separability = float(np.mean(dim_correlations)) if dim_correlations else 0.0
        
        # 5. Neighbor fitness correlation
        dist_matrix = self._dist_matrix_cache
        np.fill_diagonal(dist_matrix, np.inf)
        nearest_idx = np.argmin(dist_matrix, axis=1)
        neighbor_fit_diff = np.abs(fit - fit[nearest_idx])
        neighbor_correlation = float(1.0 / (1.0 + np.mean(neighbor_fit_diff) / (np.std(fit) + 1e-10)))
        neighbor_correlation = np.clip(neighbor_correlation, 0, 1)
        
        return [fdc, fdc_centroid, tau, separability, neighbor_correlation]
    
    # =========================================================================
    # GROUP 4: Temporal/Historical Features (Zero Cost)
    # =========================================================================
    
    def _temporal_features(self, fit: np.ndarray) -> List[float]:
        """Features from optimization history."""
        
        current_best = float(np.min(fit))
        
        # 1. Best fitness improvement rate (recent)
        if len(self.best_history) >= 3:
            recent_best = list(self.best_history)[-3:]
            improvement = (recent_best[0] - recent_best[-1]) / (np.abs(recent_best[0]) + 1e-10)
            improvement_rate = float(np.clip(improvement / 3, -1, 1))
            improvement_rate = (improvement_rate + 1) / 2  # Normalize to [0, 1]
        else:
            improvement_rate = 0.5
        
        # 2. Stagnation indicator
        if len(self.best_history) >= 10:
            recent_10 = list(self.best_history)[-10:]
            rel_improvement = (recent_10[0] - recent_10[-1]) / (np.abs(recent_10[0]) + 1e-10)
            stagnation = float(1.0 - np.clip(rel_improvement * 10, 0, 1))
        else:
            stagnation = 0.0
        
        # 3. Diversity trend
        if len(self.diversity_history) >= 3:
            recent_div = list(self.diversity_history)[-3:]
            div_trend = (recent_div[-1] - recent_div[0]) / (recent_div[0] + 1e-10)
            div_trend = float(np.clip(div_trend, -1, 1))
            div_trend = (div_trend + 1) / 2
        else:
            div_trend = 0.5
        
        # 4. Mean fitness trend
        if len(self.mean_history) >= 3:
            recent_mean = list(self.mean_history)[-3:]
            mean_improvement = (recent_mean[0] - recent_mean[-1]) / (np.abs(recent_mean[0]) + 1e-10)
            mean_trend = float(np.clip(mean_improvement, -1, 1))
            mean_trend = (mean_trend + 1) / 2
        else:
            mean_trend = 0.5
        
        # 5. Improvement consistency
        if len(self.improvement_history) >= 5:
            recent_imp = list(self.improvement_history)[-5:]
            positive_improvements = sum(1 for x in recent_imp if x > 0)
            consistency = float(positive_improvements / 5.0)
        else:
            consistency = 0.5
        
        return [improvement_rate, stagnation, div_trend, mean_trend, consistency]
    
    # =========================================================================
    # GROUP 5: Search Progress Features (Zero Cost)
    # =========================================================================
    
    def _progress_features(self, pop: np.ndarray, fit: np.ndarray) -> List[float]:
        """Features indicating search progress/stage."""
        n = len(pop)
        
        # 1. Normalized generation progress
        progress = float(min(1.0, self.nfes_used / self.max_nfes))
        
        # 2. Convergence ratio (current diversity / initial diversity)
        if len(self.diversity_history) > 0:
            initial_div = self.diversity_history[0]
            if n > 1:
                current_div = np.mean(pdist(pop)) / (self.range * np.sqrt(self.dim))
            else:
                current_div = 0
            convergence_ratio = float(current_div / (initial_div + 1e-10))
            convergence_ratio = np.clip(convergence_ratio, 0, 2) / 2
        else:
            convergence_ratio = 1.0
        
        # 3. Exploitation indicator
        if n > 1:
            pop_diameter = np.max(pdist(pop))
        else:
            pop_diameter = 0
        search_radius = pop_diameter / (self.range * np.sqrt(self.dim))
        exploitation_mode = float(1.0 - np.clip(search_radius, 0, 1))
        
        # 4. Best improvement potential
        sorted_fit = np.sort(fit)
        median_fit = sorted_fit[len(fit) // 2]
        best_fit = sorted_fit[0]
        improvement_potential = (median_fit - best_fit) / (np.abs(median_fit) + 1e-10)
        improvement_potential = float(np.clip(improvement_potential, 0, 2) / 2)
        
        # 5. Population quality
        if len(self.mean_history) > 0:
            initial_mean = self.mean_history[0]
            fraction_better = float(np.mean(fit < initial_mean))
        else:
            fraction_better = 0.5
        
        return [progress, convergence_ratio, exploitation_mode, improvement_potential, fraction_better]
    
    # =========================================================================
    # History Management (Zero Cost - Just Memory)
    # =========================================================================
    
    def _update_history(self, pop: np.ndarray, fit: np.ndarray) -> None:
        """Update historical records for temporal features."""
        current_best = float(np.min(fit))
        current_mean = float(np.mean(fit))
        if len(pop) > 1:
            current_diversity = float(np.mean(pdist(pop)) / (self.range * np.sqrt(self.dim)))
        else:
            current_diversity = 0.0
        
        # Compute improvement from last generation
        if len(self.best_history) > 0:
            improvement = self.best_history[-1] - current_best
        else:
            improvement = 0.0
        
        self.best_history.append(current_best)
        self.mean_history.append(current_mean)
        self.diversity_history.append(current_diversity)
        self.improvement_history.append(improvement)
    
    def reset(self) -> None:
        """Reset analyzer for new optimization run."""
        self.fitness_history.clear()
        self.best_history.clear()
        self.diversity_history.clear()
        self.mean_history.clear()
        self.improvement_history.clear()
        self.generation = 0
        self.nfes_used = 0
        self._dist_matrix_cache = None
        self._last_pop_hash = None


# =============================================================================
# Feature Importance Analysis (for ablation studies)
# =============================================================================

class FeatureGroupAnalyzer:
    """Analyze contribution of different feature groups."""
    
    FEATURE_GROUPS = {
        'population': slice(0, 5),
        'fitness': slice(5, 10),
        'correlation': slice(10, 15),
        'temporal': slice(15, 20),
        'progress': slice(20, 25),
    }
    
    @staticmethod
    def mask_feature_group(features: np.ndarray, group_name: str, fill_value: float = 0.5) -> np.ndarray:
        """
        Mask a feature group for ablation study.
        
        Parameters
        ----------
        features : np.ndarray
            Full 25-dimensional feature vector
        group_name : str
            Name of group to mask: 'population', 'fitness', 'correlation', 'temporal', 'progress'
        fill_value : float
            Value to fill masked features with
            
        Returns
        -------
        np.ndarray
            Features with specified group masked
        """
        masked = features.copy()
        if group_name in FeatureGroupAnalyzer.FEATURE_GROUPS:
            masked[FeatureGroupAnalyzer.FEATURE_GROUPS[group_name]] = fill_value
        return masked
    
    @staticmethod
    def get_group_features(features: np.ndarray, group_name: str) -> np.ndarray:
        """Extract features for a specific group."""
        if group_name in FeatureGroupAnalyzer.FEATURE_GROUPS:
            return features[FeatureGroupAnalyzer.FEATURE_GROUPS[group_name]]
        return np.array([])
