"""
Training Script for LA-DRL-GSK
==============================

This script trains the RL policy network for LA-DRL-GSK using PPO
on CEC2017 benchmark functions.

Training Strategy:
1. Pre-train on diverse functions (D=10, D=30)
2. Curriculum: start with easier functions, progress to harder
3. Periodic validation and model checkpointing
4. Transfer learning capability for new dimensions

Supported Platforms:
- Windows: CPU with MKL backend
- macOS: MPS (Metal) on Apple Silicon, CPU fallback

Usage:
    python train.py --dim 10 --epochs 1000 --save-dir models/
    
Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import argparse
import json
import time
import platform
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

from la_drl_gsk.policy_network import GSKPolicyNetwork, PolicyConfig, PPOTrainer
from la_drl_gsk.la_drl_gsk import LADRLGSK, LADRLGSKConfig, get_optimal_device, configure_threads
from la_drl_gsk.cec2017_benchmark import (
    BenchmarkSuite, get_cec2017_function, CEC2017_FUNCTIONS,
    UNIMODAL, SIMPLE_MULTIMODAL, HYBRID, COMPOSITION
)


def get_device(requested: Optional[str] = None) -> str:
    """
    Get the best available device for training.
    
    Parameters
    ----------
    requested : str, optional
        Requested device ('cpu', 'mps', 'cuda', 'auto')
        
    Returns
    -------
    str
        Device string
    """
    if requested and requested != 'auto':
        return requested
    
    # Auto-detect
    return get_optimal_device()


# =============================================================================
# Training Configuration
# =============================================================================

class TrainingConfig:
    """Training hyperparameters."""
    
    def __init__(
        self,
        # Training schedule
        n_epochs: int = 500,
        episodes_per_epoch: int = 10,
        
        # Curriculum
        use_curriculum: bool = True,
        curriculum_stages: int = 4,
        
        # PPO hyperparameters
        ppo_epochs: int = 10,
        batch_size: int = 64,
        lr: float = 3e-4,
        
        # Validation
        val_interval: int = 50,
        val_runs: int = 5,
        
        # Checkpointing
        save_interval: int = 100,
        save_dir: str = 'models',
        
        # Dimensions
        train_dims: List[int] = None,
        
        # Device (auto-detect if not specified)
        device: str = 'auto',
    ):
        self.n_epochs = n_epochs
        self.episodes_per_epoch = episodes_per_epoch
        self.use_curriculum = use_curriculum
        self.curriculum_stages = curriculum_stages
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.val_interval = val_interval
        self.val_runs = val_runs
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)
        self.train_dims = train_dims or [10, 30]
        self.device = device
        
    def to_dict(self) -> Dict:
        return {
            'n_epochs': self.n_epochs,
            'episodes_per_epoch': self.episodes_per_epoch,
            'use_curriculum': self.use_curriculum,
            'ppo_epochs': self.ppo_epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'train_dims': self.train_dims,
        }


# =============================================================================
# Curriculum Learning
# =============================================================================

def get_curriculum_functions(stage: int, n_stages: int = 4) -> List[int]:
    """
    Get function IDs for curriculum stage.
    
    Stage 0: Unimodal only (easiest)
    Stage 1: Unimodal + Simple Multimodal
    Stage 2: Add Hybrid functions
    Stage 3: All functions (hardest)
    """
    if stage == 0:
        return UNIMODAL
    elif stage == 1:
        return UNIMODAL + SIMPLE_MULTIMODAL
    elif stage == 2:
        return UNIMODAL + SIMPLE_MULTIMODAL + HYBRID[:5]
    else:  # stage >= 3
        return CEC2017_FUNCTIONS


# =============================================================================
# Trainer Class
# =============================================================================

class LADRLGSKTrainer:
    """
    Trainer for LA-DRL-GSK policy network.
    
    Implements curriculum learning and periodic validation.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        policy: Optional[GSKPolicyNetwork] = None,
        cec_path: Optional[str] = None,
    ):
        self.config = config
        self.cec_path = cec_path
        
        # Create directories
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize policy
        if policy is None:
            policy_config = PolicyConfig(
                state_dim=25,
                hidden_dim=256,
                lr=config.lr,
            )
            self.policy = GSKPolicyNetwork(policy_config)
        else:
            self.policy = policy
        
        self.policy = self.policy.to(config.device)
        
        # Initialize trainer
        self.ppo_trainer = PPOTrainer(
            self.policy,
            config=self.policy.config,
            device=config.device,
        )
        
        # Training state
        self.epoch = 0
        self.best_val_error = float('inf')
        self.training_history: List[Dict] = []
        self.rng = np.random.RandomState(42)
        
    def train(self) -> GSKPolicyNetwork:
        """
        Run full training loop.
        
        Returns
        -------
        GSKPolicyNetwork : Trained policy
        """
        print("=" * 60)
        print("LA-DRL-GSK Training")
        print("=" * 60)
        print(f"Config: {self.config.to_dict()}")
        print()
        
        start_time = time.time()
        
        for epoch in range(self.config.n_epochs):
            self.epoch = epoch
            
            # Determine curriculum stage
            if self.config.use_curriculum:
                stage_progress = epoch / self.config.n_epochs
                stage = min(
                    int(stage_progress * self.config.curriculum_stages),
                    self.config.curriculum_stages - 1
                )
                func_ids = get_curriculum_functions(stage)
            else:
                func_ids = CEC2017_FUNCTIONS
            
            # Run training episodes
            epoch_stats = self._train_epoch(func_ids)
            
            # Validation
            if (epoch + 1) % self.config.val_interval == 0:
                val_stats = self._validate()
                epoch_stats.update(val_stats)
                
                # Save best model
                if val_stats['val_mean_error'] < self.best_val_error:
                    self.best_val_error = val_stats['val_mean_error']
                    self._save_checkpoint('best_model.pt')
                    print(f"  New best model! Val error: {self.best_val_error:.6e}")
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Log progress
            self.training_history.append(epoch_stats)
            self._log_epoch(epoch_stats)
        
        # Save final model
        self._save_checkpoint('final_model.pt')
        
        # Save training history
        history_path = self.config.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/3600:.2f} hours")
        print(f"Best validation error: {self.best_val_error:.6e}")
        
        return self.policy
    
    def _train_epoch(self, func_ids: List[int]) -> Dict:
        """Run one training epoch."""
        self.policy.train()
        
        epoch_rewards = []
        epoch_errors = []
        
        for episode in range(self.config.episodes_per_epoch):
            # Sample function and dimension
            func_id = self.rng.choice(func_ids)
            dim = self.rng.choice(self.config.train_dims)
            
            try:
                objective, f_opt = get_cec2017_function(func_id, dim, self.cec_path)
            except ValueError:
                continue
            
            # Create optimizer for this episode
            config = LADRLGSKConfig(
                dim=dim,
                seed=self.rng.randint(0, 1000000),
                use_rl=True,
            )
            optimizer = LADRLGSK(config, policy=self.policy, device=self.config.device)
            
            # Run optimization with training
            result = optimizer.optimize(
                objective,
                training=True,
                trainer=self.ppo_trainer,
            )
            
            error = abs(result.best_f - f_opt)
            epoch_errors.append(error)
        
        # PPO update
        if len(self.ppo_trainer.buffer) >= self.config.batch_size:
            ppo_stats = self.ppo_trainer.update(
                n_epochs=self.config.ppo_epochs,
                batch_size=self.config.batch_size,
            )
        else:
            ppo_stats = {}
        
        return {
            'epoch': self.epoch,
            'mean_error': float(np.mean(epoch_errors)) if epoch_errors else 0,
            'min_error': float(np.min(epoch_errors)) if epoch_errors else 0,
            'n_episodes': len(epoch_errors),
            **ppo_stats,
        }
    
    def _validate(self) -> Dict:
        """Run validation on held-out functions."""
        self.policy.eval()
        
        val_errors = []
        
        # Use a few functions from each category
        val_funcs = [1, 5, 10, 15, 25]  # Representative functions
        
        for func_id in val_funcs:
            for dim in [10, 30]:
                try:
                    objective, f_opt = get_cec2017_function(func_id, dim, self.cec_path)
                except ValueError:
                    continue
                
                for run in range(self.config.val_runs):
                    config = LADRLGSKConfig(
                        dim=dim,
                        seed=10000 + run,
                        use_rl=True,
                    )
                    optimizer = LADRLGSK(config, policy=self.policy, device=self.config.device)
                    result = optimizer.optimize(objective)
                    
                    error = abs(result.best_f - f_opt)
                    val_errors.append(error)
        
        return {
            'val_mean_error': float(np.mean(val_errors)) if val_errors else float('inf'),
            'val_min_error': float(np.min(val_errors)) if val_errors else float('inf'),
            'val_max_error': float(np.max(val_errors)) if val_errors else float('inf'),
        }
    
    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.config.save_dir / filename
        torch.save({
            'epoch': self.epoch,
            'policy_state_dict': self.policy.state_dict(),
            'policy_config': self.policy.config,
            'optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'best_val_error': self.best_val_error,
            'training_config': self.config.to_dict(),
        }, path)
    
    def _log_epoch(self, stats: Dict) -> None:
        """Print epoch statistics."""
        epoch = stats.get('epoch', 0)
        mean_err = stats.get('mean_error', 0)
        policy_loss = stats.get('policy_loss', 0)
        val_err = stats.get('val_mean_error', None)
        
        msg = f"Epoch {epoch+1:4d}: mean_err={mean_err:.4e}"
        if policy_loss:
            msg += f" policy_loss={policy_loss:.4f}"
        if val_err is not None:
            msg += f" val_err={val_err:.4e}"
        
        print(msg)
    
    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        config: Optional[TrainingConfig] = None,
    ) -> 'LADRLGSKTrainer':
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        policy_config = checkpoint.get('policy_config', PolicyConfig())
        policy = GSKPolicyNetwork(policy_config)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        
        if config is None:
            config = TrainingConfig()
        
        trainer = cls(config, policy=policy)
        trainer.epoch = checkpoint.get('epoch', 0)
        trainer.best_val_error = checkpoint.get('best_val_error', float('inf'))
        
        return trainer


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train LA-DRL-GSK')
    
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Episodes per epoch')
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 30],
                        help='Training dimensions')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--curriculum', action='store_true', default=True,
                        help='Use curriculum learning')
    parser.add_argument('--cec-path', type=str, default=None,
                        help='Path to CEC2017 implementation')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        n_epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        train_dims=args.dims,
        lr=args.lr,
        save_dir=args.save_dir,
        device=args.device,
        use_curriculum=args.curriculum,
    )
    
    if args.resume:
        trainer = LADRLGSKTrainer.load_checkpoint(args.resume, config)
        print(f"Resumed from {args.resume} at epoch {trainer.epoch}")
    else:
        trainer = LADRLGSKTrainer(config, cec_path=args.cec_path)
    
    trainer.train()


if __name__ == '__main__':
    main()
