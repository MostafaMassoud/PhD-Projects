"""
Experiment Runner for LA-DRL-GSK
================================

This module provides comprehensive experiment functionality including:
1. Full benchmark evaluation on CEC2017
2. Ablation studies for feature groups
3. Statistical analysis (Wilcoxon, Friedman tests)
4. Comparison with baseline GSK and other algorithms
5. Result visualization and export

Usage:
    python experiments.py --mode full --dims 10 30 --runs 51
    python experiments.py --mode ablation --dim 30
    python experiments.py --mode compare --algorithms GSK LA-DRL-GSK

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Import from la_drl_gsk module (these imports are verified to exist)
from la_drl_gsk.la_drl_gsk import LADRLGSK, LADRLGSKConfig
from la_drl_gsk.landscape_analyzer import FeatureGroupAnalyzer
from la_drl_gsk.cec2017_benchmark import (
    get_cec2017_function, get_benchmark_info,
    CEC2017_FUNCTIONS, DIMENSIONS
)


# =============================================================================
# Ablation Configurations  
# Uses FeatureGroupAnalyzer group names: population, fitness, correlation, temporal, progress
# =============================================================================

ABLATION_CONFIGS = {
    'full': None,  # No ablation - use all features
    'no_population': 'population',
    'no_fitness': 'fitness',
    'no_correlation': 'correlation',
    'no_temporal': 'temporal',
    'no_progress': 'progress',
}


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    # Functions and dimensions
    func_ids: List[int] = field(default_factory=lambda: CEC2017_FUNCTIONS)
    dims: List[int] = field(default_factory=lambda: [10, 30])
    
    # Runs
    n_runs: int = 51
    seed_base: int = 0
    
    # Model
    model_path: Optional[str] = None
    
    # Output
    output_dir: str = 'results'
    experiment_name: str = 'experiment'
    
    # Parallelization
    n_workers: int = 1
    
    # CEC2017
    cec_path: Optional[str] = None
    
    # GSK parameters
    pop_size: int = 100
    control_window: int = 5


# =============================================================================
# Single Run Evaluation
# =============================================================================

def run_single_evaluation(
    func_id: int,
    dim: int,
    seed: int,
    use_rl: bool = True,
    ablation_mode: Optional[str] = None,
    model_path: Optional[str] = None,
    cec_path: Optional[str] = None,
    controller_backend: str = "heuristic",
    control_window: int = 5,
    pop_size: int = 100,
) -> Dict:
    """
    Run single optimization and return results.
    
    Parameters
    ----------
    func_id : int
        CEC2017 function ID
    dim : int
        Problem dimension
    seed : int
        Random seed
    use_rl : bool
        Whether to use RL controller
    ablation_mode : str, optional
        Feature group to ablate (population, fitness, correlation, temporal, progress)
    model_path : str, optional
        Path to trained policy (SB3 .zip file)
    cec_path : str, optional
        Path to CEC2017 implementation
    controller_backend : str
        Controller type: "fixed", "heuristic" or "sb3"
    control_window : int
        Generations per control decision
    pop_size : int
        Population size
        
    Returns
    -------
    dict with: func_id, dim, seed, best_f, f_opt, error, nfes, runtime, success
    """
    try:
        # Load function
        objective, f_opt = get_cec2017_function(func_id, dim, cec_path)
        
        # Determine controller backend
        if use_rl:
            if model_path and Path(model_path).exists():
                backend = "sb3"
            else:
                backend = controller_backend if controller_backend != "fixed" else "heuristic"
        else:
            backend = "fixed"
        
        # Create optimizer config
        config = LADRLGSKConfig(
            dim=dim,
            pop_size=pop_size,
            max_nfes=10000 * dim,
            seed=seed,
            use_rl=use_rl,
            controller_backend=backend,
            policy_path=model_path if backend == "sb3" else None,
            control_window=control_window,
            ablation_mode=ablation_mode,
        )
        
        # Create and run optimizer
        optimizer = LADRLGSK(config)
        result = optimizer.optimize(objective)
        
        error = abs(result.best_f - f_opt)
        
        return {
            'func_id': func_id,
            'dim': dim,
            'seed': seed,
            'best_f': float(result.best_f),
            'f_opt': float(f_opt),
            'error': float(error),
            'nfes': result.nfes_used,
            'runtime': result.runtime,
            'success': error < 1e-8,
            'algorithm': 'LA-DRL-GSK' if use_rl else 'GSK-Baseline',
        }
        
    except Exception as e:
        return {
            'func_id': func_id,
            'dim': dim,
            'seed': seed,
            'best_f': float('inf'),
            'f_opt': float(func_id * 100),
            'error': float('inf'),
            'nfes': 0,
            'runtime': 0.0,
            'success': False,
            'exception': str(e),
        }


# =============================================================================
# Full Benchmark Evaluation
# =============================================================================

class BenchmarkEvaluator:
    """
    Run full CEC2017 benchmark evaluation.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_full_evaluation(
        self,
        use_rl: bool = True,
        algorithm_name: str = 'LA-DRL-GSK',
        controller_backend: str = "heuristic",
    ) -> pd.DataFrame:
        """
        Run full benchmark evaluation.
        
        Returns DataFrame with results for all functions/dims/runs.
        """
        print(f"\n{'='*60}")
        print(f"Running Full Evaluation: {algorithm_name}")
        print(f"Functions: {len(self.config.func_ids)}, Dims: {self.config.dims}")
        print(f"Runs per config: {self.config.n_runs}")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for dim in self.config.dims:
            print(f"\n=== Dimension: {dim} ===\n")
            
            for func_id in self.config.func_ids:
                print(f"[F{func_id:02d}] (D={dim}): running {self.config.n_runs} trials...")
                
                func_results = []
                
                for run in range(self.config.n_runs):
                    seed = self.config.seed_base + run
                    
                    result = run_single_evaluation(
                        func_id=func_id,
                        dim=dim,
                        seed=seed,
                        use_rl=use_rl,
                        model_path=self.config.model_path,
                        cec_path=self.config.cec_path,
                        controller_backend=controller_backend,
                        control_window=self.config.control_window,
                        pop_size=self.config.pop_size,
                    )
                    result['algorithm'] = algorithm_name
                    result['run'] = run + 1
                    func_results.append(result)
                
                # Print summary
                errors = [r['error'] for r in func_results if 'error' in r and np.isfinite(r['error'])]
                if errors:
                    print(f"  Best: {np.min(errors):.4e}, "
                          f"Mean: {np.mean(errors):.4e}, "
                          f"Worst: {np.max(errors):.4e}")
                
                all_results.extend(func_results)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f'{algorithm_name.replace(" ", "_")}_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        return df
    
    def generate_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics table."""
        # Filter out rows with infinite errors
        df_valid = df[np.isfinite(df['error'])]
        
        summary = df_valid.groupby(['dim', 'func_id', 'algorithm'])['error'].agg([
            ('Best', 'min'),
            ('Median', 'median'),
            ('Mean', 'mean'),
            ('Worst', 'max'),
            ('Std', 'std'),
        ]).reset_index()
        
        return summary


# =============================================================================
# Ablation Study
# =============================================================================

class AblationStudy:
    """
    Run ablation study to analyze feature group importance.
    """
    
    ABLATION_NAMES = list(ABLATION_CONFIGS.keys())
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_ablation(self) -> pd.DataFrame:
        """
        Run ablation study for all feature groups.
        
        Returns DataFrame with results for each ablation configuration.
        """
        print(f"\n{'='*60}")
        print("Running Ablation Study")
        print(f"Configurations: {self.ABLATION_NAMES}")
        print(f"Functions: {len(self.config.func_ids)}, Dims: {self.config.dims}")
        print(f"{'='*60}\n")
        
        all_results = []
        
        for ablation_name in self.ABLATION_NAMES:
            ablation_mode = ABLATION_CONFIGS[ablation_name]
            print(f"\n=== Ablation: {ablation_name} ===")
            
            for dim in self.config.dims:
                for func_id in self.config.func_ids:
                    print(f"  F{func_id:02d} (D={dim})...", end=' ')
                    
                    errors = []
                    for run in range(self.config.n_runs):
                        seed = self.config.seed_base + run
                        
                        result = run_single_evaluation(
                            func_id=func_id,
                            dim=dim,
                            seed=seed,
                            use_rl=True,
                            ablation_mode=ablation_mode,
                            model_path=self.config.model_path,
                            cec_path=self.config.cec_path,
                            control_window=self.config.control_window,
                            pop_size=self.config.pop_size,
                        )
                        result['ablation'] = ablation_name
                        if np.isfinite(result['error']):
                            errors.append(result['error'])
                        all_results.append(result)
                    
                    if errors:
                        print(f"mean={np.mean(errors):.4e}")
                    else:
                        print("all failed")
        
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f'ablation_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        return df
    
    def analyze_ablation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze ablation results to determine feature importance.
        
        Returns dict with importance rankings and statistical tests.
        """
        # Filter valid data
        df = df[np.isfinite(df['error'])]
        
        # Get baseline (full model) results
        full_results = df[df['ablation'] == 'full']
        if full_results.empty:
            return {'importance': {}, 'ranking': []}
            
        full_mean = full_results.groupby(['dim', 'func_id'])['error'].mean()
        
        importance = {}
        
        for ablation_name in self.ABLATION_NAMES:
            if ablation_name == 'full':
                continue
            
            ablation_results = df[df['ablation'] == ablation_name]
            if ablation_results.empty:
                continue
                
            ablation_mean = ablation_results.groupby(['dim', 'func_id'])['error'].mean()
            
            # Align indices
            common_idx = full_mean.index.intersection(ablation_mean.index)
            if len(common_idx) == 0:
                continue
            
            # Compute degradation (higher = more important feature)
            degradation = (ablation_mean.loc[common_idx] - full_mean.loc[common_idx]) / (full_mean.loc[common_idx] + 1e-10)
            importance[ablation_name] = {
                'mean_degradation': float(degradation.mean()),
                'median_degradation': float(degradation.median()),
                'max_degradation': float(degradation.max()),
            }
        
        # Rank by importance
        ranked = sorted(
            importance.items(),
            key=lambda x: x[1]['mean_degradation'],
            reverse=True
        )
        
        return {
            'importance': importance,
            'ranking': [name for name, _ in ranked],
        }


# =============================================================================
# Statistical Analysis
# =============================================================================

class StatisticalAnalyzer:
    """
    Perform statistical tests for algorithm comparison.
    """
    
    @staticmethod
    def wilcoxon_test(
        errors1: np.ndarray,
        errors2: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Perform Wilcoxon signed-rank test.
        
        Returns dict with statistic, p-value, and significance.
        """
        # Remove NaN and inf values
        mask = np.isfinite(errors1) & np.isfinite(errors2)
        e1 = errors1[mask]
        e2 = errors2[mask]
        
        if len(e1) < 5:
            return {
                'statistic': 0.0,
                'pvalue': 1.0,
                'significant': False,
                'better': 'tie',
            }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, pvalue = stats.wilcoxon(e1, e2)
            except ValueError:
                # All differences are zero
                stat, pvalue = 0.0, 1.0
        
        return {
            'statistic': float(stat),
            'pvalue': float(pvalue),
            'significant': pvalue < alpha,
            'better': 'first' if np.mean(e1) < np.mean(e2) else 'second',
        }
    
    @staticmethod
    def friedman_test(
        *error_arrays: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Perform Friedman test for multiple algorithm comparison.
        """
        # Stack and remove problems with NaN in any algorithm
        stacked = np.vstack(error_arrays).T  # Shape: (n_problems, n_algos)
        mask = np.all(np.isfinite(stacked), axis=1)
        stacked = stacked[mask]
        
        if stacked.shape[0] < 5:
            return {
                'statistic': 0.0,
                'pvalue': 1.0,
                'significant': False,
                'average_ranks': [],
            }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, pvalue = stats.friedmanchisquare(*stacked.T)
            except ValueError:
                stat, pvalue = 0.0, 1.0
        
        # Compute average ranks
        n_algos = stacked.shape[1]
        n_problems = stacked.shape[0]
        
        ranks = np.zeros((n_problems, n_algos))
        for i in range(n_problems):
            ranks[i] = stats.rankdata(stacked[i])
        
        avg_ranks = ranks.mean(axis=0)
        
        return {
            'statistic': float(stat),
            'pvalue': float(pvalue),
            'significant': pvalue < alpha,
            'average_ranks': avg_ranks.tolist(),
        }
    
    @staticmethod
    def holm_posthoc(
        pvalues: List[float],
        alpha: float = 0.05,
    ) -> List[bool]:
        """
        Holm-Bonferroni post-hoc correction.
        
        Returns list of booleans indicating significance.
        """
        n = len(pvalues)
        sorted_indices = np.argsort(pvalues)
        sorted_pvalues = np.array(pvalues)[sorted_indices]
        
        significant = [False] * n
        
        for i, (idx, pval) in enumerate(zip(sorted_indices, sorted_pvalues)):
            adjusted_alpha = alpha / (n - i)
            if pval < adjusted_alpha:
                significant[idx] = True
            else:
                break
        
        return significant
    
    @staticmethod
    def compare_algorithms(
        results_df: pd.DataFrame,
        algorithm_col: str = 'algorithm',
        error_col: str = 'error',
    ) -> Dict:
        """
        Compare all algorithms in results DataFrame.
        
        Returns comprehensive comparison statistics.
        """
        # Filter valid data
        results_df = results_df[np.isfinite(results_df[error_col])]
        
        algorithms = results_df[algorithm_col].unique()
        n_algos = len(algorithms)
        
        if n_algos < 2:
            return {
                'summary': {},
                'pairwise_wilcoxon': {},
                'friedman': {'statistic': 0.0, 'pvalue': 1.0, 'significant': False},
            }
        
        # Aggregate errors per (dim, func_id)
        pivot = results_df.groupby(['dim', 'func_id', algorithm_col])[error_col].mean().unstack()
        
        # Pairwise Wilcoxon tests
        pairwise_results = {}
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                if alg1 not in pivot.columns or alg2 not in pivot.columns:
                    continue
                errors1 = pivot[alg1].values
                errors2 = pivot[alg2].values
                
                result = StatisticalAnalyzer.wilcoxon_test(errors1, errors2)
                pairwise_results[f'{alg1}_vs_{alg2}'] = result
        
        # Friedman test
        error_arrays = [pivot[alg].values for alg in algorithms if alg in pivot.columns]
        friedman_result = StatisticalAnalyzer.friedman_test(*error_arrays)
        friedman_result['algorithms'] = list(algorithms)
        
        # Summary statistics
        summary = {}
        for alg in algorithms:
            alg_df = results_df[results_df[algorithm_col] == alg]
            valid_errors = alg_df[error_col][np.isfinite(alg_df[error_col])]
            if len(valid_errors) > 0:
                summary[alg] = {
                    'mean': float(valid_errors.mean()),
                    'median': float(valid_errors.median()),
                    'std': float(valid_errors.std()),
                    'best': float(valid_errors.min()),
                    'worst': float(valid_errors.max()),
                }
        
        return {
            'summary': summary,
            'pairwise_wilcoxon': pairwise_results,
            'friedman': friedman_result,
        }


# =============================================================================
# Result Export
# =============================================================================

def export_latex_table(df: pd.DataFrame, output_path: str) -> None:
    """Export results as LaTeX table."""
    df = df[np.isfinite(df['error'])]
    
    summary = df.groupby(['dim', 'func_id', 'algorithm'])['error'].agg([
        'mean', 'std'
    ]).reset_index()
    
    # Format as mean ± std
    summary['result'] = summary.apply(
        lambda x: f"{x['mean']:.2e} ± {x['std']:.2e}", axis=1
    )
    
    pivot = summary.pivot_table(
        index=['func_id'],
        columns=['algorithm', 'dim'],
        values='result',
        aggfunc='first'
    )
    
    latex = pivot.to_latex(escape=False)
    
    with open(output_path, 'w') as f:
        f.write(latex)


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_experiments(
    mode: str,
    config: ExperimentConfig,
    compare_algorithms: Optional[List[str]] = None,
) -> None:
    """
    Main experiment runner.
    
    Parameters
    ----------
    mode : str
        'full', 'ablation', 'compare', or 'all'
    config : ExperimentConfig
        Experiment configuration
    compare_algorithms : list of str, optional
        Algorithms to compare (for 'compare' mode)
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode in ['full', 'all']:
        print("\n" + "="*60)
        print("FULL BENCHMARK EVALUATION")
        print("="*60)
        
        evaluator = BenchmarkEvaluator(config)
        
        # Run LA-DRL-GSK
        df_ladrl = evaluator.run_full_evaluation(
            use_rl=True, algorithm_name='LA-DRL-GSK'
        )
        
        # Run baseline GSK
        df_baseline = evaluator.run_full_evaluation(
            use_rl=False, algorithm_name='GSK-Baseline'
        )
        
        # Combine and compare
        df_combined = pd.concat([df_ladrl, df_baseline])
        
        # Statistical comparison
        analyzer = StatisticalAnalyzer()
        comparison = analyzer.compare_algorithms(df_combined)
        
        # Save comparison
        comparison_path = output_dir / 'statistical_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"\nComparison saved to: {comparison_path}")
        print("\nFriedman test results:")
        print(f"  Statistic: {comparison['friedman']['statistic']:.4f}")
        print(f"  P-value: {comparison['friedman']['pvalue']:.6f}")
        print(f"  Significant: {comparison['friedman']['significant']}")
    
    if mode in ['ablation', 'all']:
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)
        
        ablation = AblationStudy(config)
        df_ablation = ablation.run_ablation()
        
        # Analyze importance
        importance = ablation.analyze_ablation(df_ablation)
        
        # Save analysis
        importance_path = output_dir / 'feature_importance.json'
        with open(importance_path, 'w') as f:
            json.dump(importance, f, indent=2)
        
        print(f"\nFeature importance ranking:")
        for i, name in enumerate(importance['ranking'], 1):
            if name in importance['importance']:
                deg = importance['importance'][name]['mean_degradation']
                print(f"  {i}. {name}: {deg:.4f} degradation")
    
    if mode == 'compare':
        print("\n" + "="*60)
        print("ALGORITHM COMPARISON")
        print("="*60)
        
        evaluator = BenchmarkEvaluator(config)
        
        # Run both algorithms
        df_ladrl = evaluator.run_full_evaluation(
            use_rl=True, algorithm_name='LA-DRL-GSK'
        )
        df_baseline = evaluator.run_full_evaluation(
            use_rl=False, algorithm_name='GSK-Baseline'
        )
        
        # Combine
        df_combined = pd.concat([df_ladrl, df_baseline])
        
        # Compare
        analyzer = StatisticalAnalyzer()
        comparison = analyzer.compare_algorithms(df_combined)
        
        # Print summary
        print("\n=== Summary ===")
        for alg, stats in comparison['summary'].items():
            print(f"\n{alg}:")
            print(f"  Mean error: {stats['mean']:.4e}")
            print(f"  Median error: {stats['median']:.4e}")
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run LA-DRL-GSK experiments')
    
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'ablation', 'compare', 'all'],
                        help='Experiment mode')
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 30],
                        help='Dimensions to evaluate')
    parser.add_argument('--funcs', type=int, nargs='+', default=None,
                        help='Function IDs (default: all)')
    parser.add_argument('--runs', type=int, default=51,
                        help='Number of runs per configuration')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--cec-path', type=str, default=None,
                        help='Path to CEC2017 implementation')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    config = ExperimentConfig(
        func_ids=args.funcs or CEC2017_FUNCTIONS,
        dims=args.dims,
        n_runs=args.runs,
        model_path=args.model,
        output_dir=args.output,
        cec_path=args.cec_path,
        n_workers=args.workers,
    )
    
    run_experiments(args.mode, config)


if __name__ == '__main__':
    main()
