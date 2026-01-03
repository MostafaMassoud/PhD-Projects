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
    python experiments.py --mode compare --algorithms GSK LSHADE

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from la_drl_gsk.la_drl_gsk import (
    LADRLGSK, LADRLGSKConfig, create_baseline_gsk, 
    create_ablation_optimizer, ABLATION_CONFIGS
)
from la_drl_gsk.policy_network import GSKPolicyNetwork
from la_drl_gsk.cec2017_benchmark import (
    get_cec2017_function, get_benchmark_info,
    CEC2017_FUNCTIONS, DIMENSIONS
)


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
        Feature group to ablate
    model_path : str, optional
        Path to trained policy
    cec_path : str, optional
        Path to CEC2017 implementation
        
    Returns
    -------
    dict with: func_id, dim, seed, best_f, error, nfes, runtime
    """
    try:
        # Load function
        objective, f_opt = get_cec2017_function(func_id, dim, cec_path)
        
        # Load policy if using RL
        policy = None
        if use_rl and model_path and Path(model_path).exists():
            policy = GSKPolicyNetwork.load(model_path)
        
        # Create optimizer
        config = LADRLGSKConfig(
            dim=dim,
            seed=seed,
            use_rl=use_rl,
            ablation_mode=ablation_mode,
        )
        optimizer = LADRLGSK(config, policy=policy)
        
        # Run optimization
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
        }
        
    except Exception as e:
        return {
            'func_id': func_id,
            'dim': dim,
            'seed': seed,
            'error': float('inf'),
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
                    )
                    result['algorithm'] = algorithm_name
                    result['run'] = run + 1
                    func_results.append(result)
                
                # Print summary
                errors = [r['error'] for r in func_results if 'error' in r]
                if errors:
                    print(f"  Best: {np.min(errors):.4e}, "
                          f"Mean: {np.mean(errors):.4e}, "
                          f"Worst: {np.max(errors):.4e}")
                
                all_results.extend(func_results)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f'{algorithm_name}_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        return df
    
    def generate_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics table."""
        summary = df.groupby(['dim', 'func_id', 'algorithm'])['error'].agg([
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
                        )
                        result['ablation'] = ablation_name
                        errors.append(result['error'])
                        all_results.append(result)
                    
                    print(f"mean={np.mean(errors):.4e}")
        
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_path = self.output_dir / f'ablation_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        
        return df
    
    def analyze_ablation(self, df: pd.DataFrame) -> Dict:
        """
        Analyze ablation results to determine feature importance.
        
        Returns dict with importance rankings and statistical tests.
        """
        # Get baseline (full model) results
        full_results = df[df['ablation'] == 'full']
        full_mean = full_results.groupby(['dim', 'func_id'])['error'].mean()
        
        importance = {}
        
        for ablation_name in self.ABLATION_NAMES:
            if ablation_name == 'full':
                continue
            
            ablation_results = df[df['ablation'] == ablation_name]
            ablation_mean = ablation_results.groupby(['dim', 'func_id'])['error'].mean()
            
            # Compute degradation (higher = more important feature)
            degradation = (ablation_mean - full_mean) / (full_mean + 1e-10)
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, pvalue = stats.wilcoxon(errors1, errors2)
            except ValueError:
                # All differences are zero
                stat, pvalue = 0.0, 1.0
        
        return {
            'statistic': float(stat),
            'pvalue': float(pvalue),
            'significant': pvalue < alpha,
            'better': 'first' if np.mean(errors1) < np.mean(errors2) else 'second',
        }
    
    @staticmethod
    def friedman_test(
        *error_arrays: np.ndarray,
        alpha: float = 0.05,
    ) -> Dict:
        """
        Perform Friedman test for multiple algorithm comparison.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, pvalue = stats.friedmanchisquare(*error_arrays)
            except ValueError:
                stat, pvalue = 0.0, 1.0
        
        # Compute average ranks
        n_algos = len(error_arrays)
        n_problems = len(error_arrays[0])
        
        ranks = np.zeros((n_problems, n_algos))
        for i in range(n_problems):
            problem_errors = [arr[i] for arr in error_arrays]
            ranks[i] = stats.rankdata(problem_errors)
        
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
        algorithms = results_df[algorithm_col].unique()
        n_algos = len(algorithms)
        
        # Aggregate errors per (dim, func_id)
        pivot = results_df.groupby(['dim', 'func_id', algorithm_col])[error_col].mean().unstack()
        
        # Pairwise Wilcoxon tests
        pairwise_results = {}
        for i, alg1 in enumerate(algorithms):
            for alg2 in algorithms[i+1:]:
                errors1 = pivot[alg1].values
                errors2 = pivot[alg2].values
                
                # Remove NaN
                mask = ~(np.isnan(errors1) | np.isnan(errors2))
                result = StatisticalAnalyzer.wilcoxon_test(
                    errors1[mask], errors2[mask]
                )
                pairwise_results[f'{alg1}_vs_{alg2}'] = result
        
        # Friedman test
        error_arrays = [pivot[alg].values for alg in algorithms]
        friedman_result = StatisticalAnalyzer.friedman_test(*error_arrays)
        friedman_result['algorithms'] = list(algorithms)
        
        # Summary statistics
        summary = {}
        for alg in algorithms:
            alg_df = results_df[results_df[algorithm_col] == alg]
            summary[alg] = {
                'mean': float(alg_df[error_col].mean()),
                'median': float(alg_df[error_col].median()),
                'std': float(alg_df[error_col].std()),
                'best': float(alg_df[error_col].min()),
                'worst': float(alg_df[error_col].max()),
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


def export_convergence_data(df: pd.DataFrame, output_path: str) -> None:
    """Export convergence history for plotting."""
    # This would require storing convergence history in results
    pass


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
            deg = importance['importance'][name]['mean_degradation']
            print(f"  {i}. {name}: {deg:.4f} degradation")
    
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
