"""
Visualization Utilities for LA-DRL-GSK
======================================

Provides plotting functions for:
- Convergence curves
- Feature importance (ablation)
- Statistical comparison tables
- Landscape feature visualization

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import LogLocator, LogFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


# =============================================================================
# Style Configuration
# =============================================================================

def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (6, 4),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 1.5,
    })


# =============================================================================
# Convergence Plots
# =============================================================================

def plot_convergence_curve(
    histories: Dict[str, np.ndarray],
    title: str = 'Convergence Curve',
    xlabel: str = 'Generation',
    ylabel: str = 'Best Fitness (log)',
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """
    Plot convergence curves for multiple algorithms.
    
    Parameters
    ----------
    histories : dict
        Algorithm name -> convergence history array
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    setup_publication_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(histories)))
    
    for (name, history), color in zip(histories.items(), colors):
        generations = np.arange(len(history))
        ax.semilogy(generations, history, label=name, color=color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_convergence_with_confidence(
    histories_list: Dict[str, List[np.ndarray]],
    title: str = 'Convergence with Confidence Intervals',
    save_path: Optional[str] = None,
    confidence: float = 0.95,
) -> None:
    """
    Plot convergence with confidence intervals from multiple runs.
    
    Parameters
    ----------
    histories_list : dict
        Algorithm name -> list of convergence histories
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    setup_publication_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(histories_list)))
    
    for (name, histories), color in zip(histories_list.items(), colors):
        # Align histories to same length
        min_len = min(len(h) for h in histories)
        aligned = np.array([h[:min_len] for h in histories])
        
        mean = np.mean(aligned, axis=0)
        std = np.std(aligned, axis=0)
        
        # Confidence interval
        z = 1.96 if confidence == 0.95 else 2.58
        ci = z * std / np.sqrt(len(histories))
        
        generations = np.arange(min_len)
        
        ax.semilogy(generations, mean, label=name, color=color)
        ax.fill_between(generations, mean - ci, mean + ci, color=color, alpha=0.2)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness (log)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Ablation Study Visualization
# =============================================================================

def plot_ablation_results(
    ablation_df: pd.DataFrame,
    metric: str = 'error',
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ablation study results as bar chart.
    
    Parameters
    ----------
    ablation_df : pd.DataFrame
        DataFrame with 'ablation' and metric columns
    metric : str
        Column name for metric to plot
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    setup_publication_style()
    
    # Aggregate by ablation
    summary = ablation_df.groupby('ablation')[metric].agg(['mean', 'std']).reset_index()
    summary = summary.sort_values('mean')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(summary))
    bars = ax.bar(x, summary['mean'], yerr=summary['std'], capsize=3, 
                  color='steelblue', edgecolor='black', alpha=0.8)
    
    # Highlight full model
    full_idx = summary[summary['ablation'] == 'full'].index
    if len(full_idx) > 0:
        idx = list(summary['ablation']).index('full')
        bars[idx].set_color('forestgreen')
    
    ax.set_xticks(x)
    ax.set_xticklabels(summary['ablation'], rotation=45, ha='right')
    ax.set_ylabel(f'Mean {metric.capitalize()}')
    ax.set_title('Ablation Study: Feature Group Importance')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(
    importance_dict: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot feature group importance as horizontal bar chart.
    
    Parameters
    ----------
    importance_dict : dict
        From AblationStudy.analyze_ablation()
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    setup_publication_style()
    
    # Extract data
    names = []
    degradations = []
    for name, data in importance_dict['importance'].items():
        names.append(name.replace('no_', ''))
        degradations.append(data['mean_degradation'])
    
    # Sort by importance
    sorted_idx = np.argsort(degradations)[::-1]
    names = [names[i] for i in sorted_idx]
    degradations = [degradations[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))
    y = np.arange(len(names))
    
    ax.barh(y, degradations, color=colors, edgecolor='black', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('Mean Degradation (higher = more important)')
    ax.set_title('Feature Group Importance')
    ax.invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Statistical Comparison Tables
# =============================================================================

def create_comparison_table(
    results_df: pd.DataFrame,
    algorithms: List[str],
    output_format: str = 'latex',
) -> str:
    """
    Create comparison table in LaTeX or markdown format.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with 'algorithm', 'func_id', 'dim', 'error' columns
    algorithms : list
        Algorithm names to include
    output_format : str
        'latex' or 'markdown'
        
    Returns
    -------
    str : Formatted table
    """
    # Filter algorithms
    df = results_df[results_df['algorithm'].isin(algorithms)]
    
    # Pivot table
    pivot = df.pivot_table(
        index=['dim', 'func_id'],
        columns='algorithm',
        values='error',
        aggfunc=['mean', 'std']
    )
    
    # Format as mean ± std
    formatted = pd.DataFrame(index=pivot.index)
    for alg in algorithms:
        mean = pivot['mean'][alg]
        std = pivot['std'][alg]
        formatted[alg] = [f"{m:.2e} ± {s:.2e}" for m, s in zip(mean, std)]
    
    if output_format == 'latex':
        return formatted.to_latex(escape=False)
    else:
        return formatted.to_markdown()


def create_wilcoxon_table(
    comparison_results: Dict,
    output_format: str = 'latex',
) -> str:
    """
    Create Wilcoxon test results table.
    
    Parameters
    ----------
    comparison_results : dict
        From StatisticalAnalyzer.compare_algorithms()
    """
    pairwise = comparison_results['pairwise_wilcoxon']
    
    rows = []
    for pair, result in pairwise.items():
        alg1, alg2 = pair.split('_vs_')
        sig = '✓' if result['significant'] else '✗'
        better = result['better']
        
        rows.append({
            'Comparison': f"{alg1} vs {alg2}",
            'p-value': f"{result['pvalue']:.4e}",
            'Significant': sig,
            'Better': better,
        })
    
    df = pd.DataFrame(rows)
    
    if output_format == 'latex':
        return df.to_latex(index=False, escape=False)
    return df.to_markdown(index=False)


# =============================================================================
# Landscape Feature Visualization
# =============================================================================

def plot_landscape_features(
    features_history: List[np.ndarray],
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot evolution of landscape features over generations.
    
    Parameters
    ----------
    features_history : list
        List of 25-dimensional feature vectors per generation
    feature_names : list, optional
        Names for each feature
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    if feature_names is None:
        from la_drl_gsk.landscape_analyzer import LandscapeFeatures
        feature_names = LandscapeFeatures.feature_names()
    
    features = np.array(features_history)
    n_gen, n_feat = features.shape
    
    setup_publication_style()
    fig, axes = plt.subplots(5, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.plot(features[:, i], color='steelblue', alpha=0.8)
        ax.set_title(name, fontsize=8)
        ax.set_ylim(0, 1)
        if i >= 20:
            ax.set_xlabel('Generation')
    
    plt.suptitle('Landscape Features Evolution', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_action_distribution(
    actions_history: List[Dict],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot distribution of RL actions over optimization.
    
    Parameters
    ----------
    actions_history : list
        List of action dictionaries
    """
    if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
        return
    
    setup_publication_style()
    
    df = pd.DataFrame(actions_history)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # p_junior distribution
    sns.histplot(df['p_junior'], ax=axes[0, 0], kde=True, color='steelblue')
    axes[0, 0].set_title('Junior Phase Probability')
    axes[0, 0].set_xlabel('p_junior')
    
    # delta_kf distribution
    sns.histplot(df['delta_kf'], ax=axes[0, 1], kde=True, color='forestgreen')
    axes[0, 1].set_title('KF Adjustment')
    axes[0, 1].set_xlabel('Δ KF')
    
    # delta_kr distribution
    sns.histplot(df['delta_kr'], ax=axes[1, 0], kde=True, color='coral')
    axes[1, 0].set_title('KR Adjustment')
    axes[1, 0].set_xlabel('Δ KR')
    
    # Strategy selection
    strategy_counts = df['strategy'].value_counts()
    axes[1, 1].bar(['Standard', 'Aggressive', 'Conservative'], 
                   [strategy_counts.get(i, 0) for i in range(3)],
                   color=['steelblue', 'coral', 'forestgreen'])
    axes[1, 1].set_title('Strategy Selection')
    axes[1, 1].set_ylabel('Count')
    
    plt.suptitle('RL Action Distribution', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Heatmap Visualizations
# =============================================================================

def plot_performance_heatmap(
    results_df: pd.DataFrame,
    algorithm: str,
    metric: str = 'error',
    save_path: Optional[str] = None,
) -> None:
    """
    Plot performance heatmap (functions × dimensions).
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results with columns: algorithm, func_id, dim, error
    algorithm : str
        Algorithm to visualize
    metric : str
        Metric to plot
    """
    if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
        return
    
    setup_publication_style()
    
    df = results_df[results_df['algorithm'] == algorithm]
    pivot = df.pivot_table(index='func_id', columns='dim', values=metric, aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Log scale for errors
    log_data = np.log10(pivot.values + 1e-10)
    
    sns.heatmap(log_data, ax=ax, cmap='RdYlGn_r', 
                xticklabels=pivot.columns, yticklabels=pivot.index,
                cbar_kws={'label': f'log10({metric})'})
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Function ID')
    ax.set_title(f'{algorithm} Performance Heatmap')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Export Functions
# =============================================================================

def export_all_figures(
    results_df: pd.DataFrame,
    output_dir: str,
    algorithms: Optional[List[str]] = None,
) -> None:
    """
    Export all standard figures for a paper.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Complete results DataFrame
    output_dir : str
        Directory to save figures
    algorithms : list, optional
        Algorithms to include
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if algorithms is None:
        algorithms = results_df['algorithm'].unique().tolist()
    
    print(f"Exporting figures to {output_dir}...")
    
    # Performance heatmaps
    for alg in algorithms:
        plot_performance_heatmap(
            results_df, alg, 
            save_path=str(output_dir / f'heatmap_{alg}.png')
        )
    
    print("Figures exported successfully!")
