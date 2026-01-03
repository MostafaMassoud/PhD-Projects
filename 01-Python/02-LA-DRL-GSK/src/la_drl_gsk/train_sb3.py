"""
LA-DRL-GSK SB3 PPO Training Script
==================================

Train PPO controllers for GSK parameter control using stable-baselines3.

Usage:
    python -m la_drl_gsk.train_sb3 --dim 30 --timesteps 200000 --n-envs 8

Requirements:
    pip install gymnasium stable-baselines3

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import argparse
import platform
import pickle
from pathlib import Path
from typing import Optional, List
import numpy as np


def train_ppo(
    dim: int = 30,
    pop_size: int = 100,
    max_nfes: Optional[int] = None,
    control_window: int = 5,
    total_timesteps: int = 200000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    save_dir: str = "models",
    cec_path: Optional[str] = None,
    device: str = "auto",
    seed: int = 42,
    verbose: int = 1,
) -> str:
    """
    Train PPO controller for GSK.
    
    Parameters
    ----------
    dim : int
        Problem dimension
    pop_size : int
        Population size for GSK
    max_nfes : int, optional
        Max function evaluations (default: 10000*dim)
    control_window : int
        Generations per RL decision
    total_timesteps : int
        Total training timesteps
    n_envs : int
        Number of parallel environments
    learning_rate : float
        PPO learning rate
    n_steps : int
        Steps per environment per update
    batch_size : int
        Minibatch size for PPO
    gamma : float
        Discount factor
    gae_lambda : float
        GAE lambda
    ent_coef : float
        Entropy coefficient
    save_dir : str
        Directory to save models
    cec_path : str, optional
        Path to CEC2017 implementation
    device : str
        Device for training
    seed : int
        Random seed
    verbose : int
        Verbosity level
        
    Returns
    -------
    str
        Path to saved model
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.utils import set_random_seed
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for training. "
            "Install with: pip install stable-baselines3"
        )
    
    try:
        from .gsk_env import make_gsk_env
    except ImportError:
        raise ImportError(
            "gymnasium is required for training. "
            "Install with: pip install gymnasium"
        )
    
    if max_nfes is None:
        max_nfes = 10000 * dim
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training PPO for LA-DRL-GSK (D={dim})")
    print(f"{'='*60}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Environments: {n_envs}")
    print(f"  Control window: {control_window}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    # Create environment factory
    def make_env(env_seed: int):
        def _init():
            env = make_gsk_env(
                dim=dim,
                pop_size=pop_size,
                max_nfes=max_nfes,
                control_window=control_window,
                cec_path=cec_path,
                seed=env_seed,
            )
            return env
        return _init
    
    # Create vectorized environments
    if platform.system() == 'Windows' or n_envs == 1:
        env = DummyVecEnv([make_env(seed + i) for i in range(n_envs)])
    else:
        env = SubprocVecEnv([make_env(seed + i) for i in range(n_envs)])
    
    # Wrap with VecNormalize for observation normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(seed + 1000)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Set random seed
    set_random_seed(seed)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        device=device,
        seed=seed,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1000),
        save_path=str(save_path / f"checkpoints_D{dim}"),
        name_prefix="ppo_gsk",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / f"best_D{dim}"),
        log_path=str(save_path / f"logs_D{dim}"),
        eval_freq=max(5000 // n_envs, 500),
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
    model_path = save_path / f"ppo_gsk_D{dim}.zip"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Save VecNormalize statistics
    vec_normalize_path = save_path / f"vecnormalize_D{dim}.pkl"
    with open(vec_normalize_path, 'wb') as f:
        pickle.dump({
            'obs_rms': env.obs_rms,
            'ret_rms': env.ret_rms if hasattr(env, 'ret_rms') else None,
        }, f)
    print(f"VecNormalize stats saved to: {vec_normalize_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return str(model_path)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train PPO controller for LA-DRL-GSK',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--dim', type=int, default=30,
                        help='Problem dimension')
    parser.add_argument('--pop-size', type=int, default=100,
                        help='Population size')
    parser.add_argument('--max-nfes', type=int, default=None,
                        help='Max function evaluations (default: 10000*dim)')
    parser.add_argument('--control-window', type=int, default=5,
                        help='Generations per RL decision')
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=8,
                        help='Number of parallel environments')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Steps per environment per update')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Minibatch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--cec-path', type=str, default=None,
                        help='Path to CEC2017 implementation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for training (cpu/cuda/mps/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    
    args = parser.parse_args()
    
    train_ppo(
        dim=args.dim,
        pop_size=args.pop_size,
        max_nfes=args.max_nfes,
        control_window=args.control_window,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        save_dir=args.save_dir,
        cec_path=args.cec_path,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
