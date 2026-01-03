#!/usr/bin/env python3
"""
LA-DRL-GSK Main Runner
======================

Unified interface for running LA-DRL-GSK experiments.

Commands:
    train       Train the RL policy network
    evaluate    Evaluate on CEC2017 benchmarks
    ablation    Run ablation studies
    compare     Compare with baseline algorithms
    demo        Quick demonstration

Supported Platforms:
    - Windows (Intel CPU with MKL)
    - macOS (Apple Silicon with MPS, Intel with MKL)
    - Linux (CPU with MKL)

Examples:
    python run.py demo
    python run.py train --epochs 500 --dims 10 30
    python run.py evaluate --dims 10 30 --runs 51
    python run.py ablation --dim 30 --runs 25
    python run.py compare --algorithms GSK LA-DRL-GSK

Author: LA-DRL-GSK Research Team
Date: 2025
"""

import argparse
import sys
import time
import platform
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np


def print_platform_info():
    """Print platform information."""
    print(f"\nPlatform: {platform.system()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        
        # Check backends
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            if torch.backends.mps.is_available():
                print("Backend: MPS (Metal) available")
            else:
                print("Backend: CPU (MPS not available)")
        else:
            if torch.backends.mkl.is_available():
                print("Backend: MKL available")
            else:
                print("Backend: CPU")
        print(f"Threads: {torch.get_num_threads()}")
    except ImportError:
        print("PyTorch: Not installed (using NumPy-only mode)")


def run_demo(detailed: bool = False):
    """Run quick demonstration of LA-DRL-GSK."""
    from la_drl_gsk import (
        LADRLGSK, LADRLGSKConfig, create_baseline_gsk, configure_threads,
        OptimizationLogger
    )
    from la_drl_gsk.cec2017_benchmark import get_cec2017_function
    
    # Configure threads
    configure_threads()
    
    # Test parameters
    dim = 10
    func_id = 5  # Rastrigin
    n_runs = 3 if detailed else 5
    
    try:
        objective, f_opt = get_cec2017_function(func_id, dim)
    except Exception as e:
        print(f"Warning: CEC2017 not available ({e})")
        print("Using synthetic test function...")
        f_opt = func_id * 100
        def objective(x):
            x = np.atleast_2d(x)
            return np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10, axis=1) + f_opt
    
    if detailed:
        # Run single detailed optimization
        print("\n" + "="*70)
        print("DETAILED OPTIMIZATION RUN")
        print("="*70)
        
        config = LADRLGSKConfig(dim=dim, seed=42, use_rl=True, max_nfes=50000)
        optimizer = LADRLGSK(config)
        
        logger = OptimizationLogger(
            verbosity=2,
            log_interval=50,
            show_features=True,
            show_params=True,
        )
        
        result = optimizer.optimize(objective, logger=logger)
        
    else:
        # Quick comparison
        print("\n" + "="*60)
        print("LA-DRL-GSK DEMONSTRATION")
        print("="*60)
        
        print_platform_info()
        
        print(f"\nBenchmark: CEC2017 F{func_id} (D={dim})")
        print(f"Running {n_runs} trials each for baseline and LA-DRL-GSK...")
        
        # Run baseline GSK
        print("\n--- Baseline GSK ---")
        baseline_results = []
        for run in range(n_runs):
            config = LADRLGSKConfig(dim=dim, seed=run, use_rl=False)
            optimizer = LADRLGSK(config)
            result = optimizer.optimize(objective)
            baseline_results.append(result.best_f)
            print(f"  Run {run+1}: error={abs(result.best_f - f_opt):.6e}, nfes={result.nfes_used}")
        
        baseline_errors = [abs(r - f_opt) for r in baseline_results]
        print(f"  Mean error: {np.mean(baseline_errors):.6e}")
        
        # Run LA-DRL-GSK
        print("\n--- LA-DRL-GSK ---")
        ladrl_results = []
        for run in range(n_runs):
            config = LADRLGSKConfig(dim=dim, seed=run, use_rl=True)
            optimizer = LADRLGSK(config)
            result = optimizer.optimize(objective)
            ladrl_results.append(result.best_f)
            print(f"  Run {run+1}: error={abs(result.best_f - f_opt):.6e}, nfes={result.nfes_used}")
        
        ladrl_errors = [abs(r - f_opt) for r in ladrl_results]
        print(f"  Mean error: {np.mean(ladrl_errors):.6e}")
        
        # Compare using logger
        from la_drl_gsk import OptimizationLogger
        logger = OptimizationLogger(verbosity=1)
        logger.print_comparison(
            "Baseline GSK", baseline_results,
            "LA-DRL-GSK", ladrl_results,
            f_opt=f_opt
        )
    
    print("Demo complete!")


def run_train(args):
    """Run SB3 PPO training."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    except ImportError:
        print("Error: stable-baselines3 required for training.")
        print("Install with: pip install stable-baselines3")
        return
    
    try:
        from la_drl_gsk.gsk_env import make_gsk_env
    except ImportError:
        print("Error: gymnasium required for training.")
        print("Install with: pip install gymnasium")
        return
    
    from la_drl_gsk import configure_threads
    import os
    
    configure_threads()
    
    # Training parameters
    dims = args.dims
    total_timesteps = args.timesteps
    n_envs = args.n_envs
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for dim in dims:
        print(f"\n{'='*60}")
        print(f"Training PPO for D={dim}")
        print(f"{'='*60}")
        
        # Create vectorized environments
        def make_env(seed):
            def _init():
                return make_gsk_env(
                    dim=dim,
                    pop_size=100,
                    max_nfes=10000 * dim,
                    control_window=5,
                    seed=seed,
                )
            return _init
        
        # Use DummyVecEnv on Windows for stability
        if platform.system() == 'Windows':
            env = DummyVecEnv([make_env(i) for i in range(n_envs)])
        else:
            env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        
        # Create eval environment
        eval_env = DummyVecEnv([make_env(999)])
        
        # PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            device=args.device,
        )
        
        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(save_dir / f"checkpoints_D{dim}"),
            name_prefix="ppo_gsk",
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(save_dir),
            log_path=str(save_dir / "logs"),
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
        )
        
        # Train
        print(f"Training for {total_timesteps:,} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
        
        # Save final model
        model_path = save_dir / f"ppo_gsk_D{dim}.zip"
        model.save(str(model_path))
        print(f"Model saved to: {model_path}")
        
        env.close()
        eval_env.close()
    
    print("\nTraining complete!")


def run_evaluate(args):
    """Run evaluation."""
    from la_drl_gsk.experiments import BenchmarkEvaluator, ExperimentConfig
    
    config = ExperimentConfig(
        dims=args.dims,
        n_runs=args.runs,
        model_path=args.model,
        output_dir=args.output,
        cec_path=args.cec_path,
    )
    
    evaluator = BenchmarkEvaluator(config)
    
    # Run LA-DRL-GSK
    df = evaluator.run_full_evaluation(use_rl=True, algorithm_name='LA-DRL-GSK')
    
    # Generate summary
    summary = evaluator.generate_summary_table(df)
    print("\n=== Summary Statistics ===")
    print(summary.to_string())


def run_ablation(args):
    """Run ablation study."""
    from la_drl_gsk.experiments import AblationStudy, ExperimentConfig
    
    config = ExperimentConfig(
        dims=[args.dim],
        n_runs=args.runs,
        model_path=args.model,
        output_dir=args.output,
        cec_path=args.cec_path,
    )
    
    study = AblationStudy(config)
    df = study.run_ablation()
    
    # Analyze
    importance = study.analyze_ablation(df)
    
    print("\n=== Feature Importance Ranking ===")
    for i, name in enumerate(importance['ranking'], 1):
        deg = importance['importance'][name]['mean_degradation']
        print(f"  {i}. {name}: {deg:.4f} mean degradation")


def run_compare(args):
    """Run algorithm comparison."""
    from la_drl_gsk.experiments import run_experiments, ExperimentConfig
    
    config = ExperimentConfig(
        dims=args.dims,
        n_runs=args.runs,
        model_path=args.model,
        output_dir=args.output,
        cec_path=args.cec_path,
    )
    
    run_experiments('full', config, compare_algorithms=args.algorithms)


def main():
    parser = argparse.ArgumentParser(
        description='LA-DRL-GSK: Landscape-Aware Deep RL GSK Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Quick demonstration')
    demo_parser.add_argument('--detailed', '-d', action='store_true',
                             help='Show detailed logging with features and parameters')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train SB3 PPO policy')
    train_parser.add_argument('--timesteps', type=int, default=200000,
                              help='Total timesteps for training')
    train_parser.add_argument('--n-envs', type=int, default=8,
                              help='Number of parallel environments')
    train_parser.add_argument('--dims', type=int, nargs='+', default=[10],
                              help='Dimensions to train on')
    train_parser.add_argument('--lr', type=float, default=3e-4,
                              help='Learning rate')
    train_parser.add_argument('--save-dir', type=str, default='models',
                              help='Directory to save models')
    train_parser.add_argument('--device', type=str, default='cpu',
                              help='Device for training (cpu/cuda)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on benchmarks')
    eval_parser.add_argument('--dims', type=int, nargs='+', default=[10, 30])
    eval_parser.add_argument('--runs', type=int, default=51)
    eval_parser.add_argument('--model', type=str, default=None)
    eval_parser.add_argument('--output', type=str, default='results')
    eval_parser.add_argument('--cec-path', type=str, default=None)
    
    # Ablation command
    ablation_parser = subparsers.add_parser('ablation', help='Run ablation study')
    ablation_parser.add_argument('--dim', type=int, default=30)
    ablation_parser.add_argument('--runs', type=int, default=25)
    ablation_parser.add_argument('--model', type=str, default=None)
    ablation_parser.add_argument('--output', type=str, default='results')
    ablation_parser.add_argument('--cec-path', type=str, default=None)
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare algorithms')
    compare_parser.add_argument('--algorithms', type=str, nargs='+', 
                                default=['GSK', 'LA-DRL-GSK'])
    compare_parser.add_argument('--dims', type=int, nargs='+', default=[10, 30])
    compare_parser.add_argument('--runs', type=int, default=51)
    compare_parser.add_argument('--model', type=str, default=None)
    compare_parser.add_argument('--output', type=str, default='results')
    compare_parser.add_argument('--cec-path', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'demo':
        run_demo(detailed=args.detailed)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'evaluate':
        run_evaluate(args)
    elif args.command == 'ablation':
        run_ablation(args)
    elif args.command == 'compare':
        run_compare(args)


if __name__ == '__main__':
    main()
