"""
Deep Reinforcement Learning Policy Networks for LA-DRL-GSK
==========================================================

This module implements the actor-critic neural networks and PPO training
algorithm for controlling GSK parameters based on landscape features.

Key Components:
- GSKPolicyNetwork: Actor-critic network with continuous + discrete outputs
- PPOTrainer: Proximal Policy Optimization training algorithm
- ReplayBuffer: Experience storage for training

Reference:
    Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
    arXiv preprint arXiv:1707.06347.

Author: LA-DRL-GSK Research Team
Date: 2025
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import random


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PolicyConfig:
    """Configuration for policy network architecture."""
    state_dim: int = 25          # FLA features
    hidden_dim: int = 256        # Hidden layer size
    n_hidden_layers: int = 2     # Number of hidden layers
    activation: str = 'relu'     # Activation function
    
    # Action bounds
    kf_range: Tuple[float, float] = (0.1, 0.9)
    kr_range: Tuple[float, float] = (0.5, 1.0)
    k_range: Tuple[float, float] = (1.0, 20.0)
    
    # Training hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    

@dataclass
class Action:
    """Container for RL agent actions."""
    p_junior: float     # Probability of junior phase [0, 1]
    delta_kf: float     # KF adjustment [-0.2, +0.2]
    delta_kr: float     # KR adjustment [-0.2, +0.2]
    delta_k: float      # K adjustment [-5, +5]
    strategy: int       # Mutation strategy index [0, 1, 2]
    
    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            'p_junior': self.p_junior,
            'delta_kf': self.delta_kf,
            'delta_kr': self.delta_kr,
            'delta_k': self.delta_k,
            'strategy': self.strategy,
        }


# =============================================================================
# Neural Network Architecture
# =============================================================================

class GSKPolicyNetwork(nn.Module):
    """
    Actor-Critic network for GSK parameter control.
    
    Architecture:
    - Shared backbone: MLP with layer normalization
    - Actor heads: Gaussian distributions for continuous actions + categorical for strategy
    - Critic head: State value estimation
    
    Input: 25 FLA features (normalized to [0, 1])
    
    Output:
    - p_junior: Beta distribution parameters for junior phase probability
    - delta_kf: Gaussian distribution for KF adjustment
    - delta_kr: Gaussian distribution for KR adjustment  
    - delta_k: Gaussian distribution for K adjustment
    - strategy: Categorical distribution over 3 mutation strategies
    - value: Scalar state value estimate
    """
    
    def __init__(self, config: Optional[PolicyConfig] = None):
        super().__init__()
        self.config = config or PolicyConfig()
        
        # Build shared backbone
        layers = []
        in_dim = self.config.state_dim
        for i in range(self.config.n_hidden_layers):
            layers.extend([
                nn.Linear(in_dim, self.config.hidden_dim),
                nn.LayerNorm(self.config.hidden_dim),
                nn.ReLU() if self.config.activation == 'relu' else nn.Tanh(),
            ])
            in_dim = self.config.hidden_dim
        self.shared = nn.Sequential(*layers)
        
        # Policy heads (Actor)
        # p_junior: [0, 1] - use sigmoid
        self.p_junior_mean = nn.Linear(self.config.hidden_dim, 1)
        self.p_junior_logstd = nn.Parameter(torch.zeros(1))
        
        # delta_kf: [-0.2, +0.2] - use tanh * 0.2
        self.delta_kf_mean = nn.Linear(self.config.hidden_dim, 1)
        self.delta_kf_logstd = nn.Parameter(torch.zeros(1))
        
        # delta_kr: [-0.2, +0.2]
        self.delta_kr_mean = nn.Linear(self.config.hidden_dim, 1)
        self.delta_kr_logstd = nn.Parameter(torch.zeros(1))
        
        # delta_k: [-5, +5] - use tanh * 5
        self.delta_k_mean = nn.Linear(self.config.hidden_dim, 1)
        self.delta_k_logstd = nn.Parameter(torch.zeros(1))
        
        # strategy: categorical over 3 options
        self.strategy_logits = nn.Linear(self.config.hidden_dim, 3)
        
        # Value head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Smaller initialization for output heads
        for head in [self.p_junior_mean, self.delta_kf_mean, 
                     self.delta_kr_mean, self.delta_k_mean, self.strategy_logits]:
            nn.init.orthogonal_(head.weight, gain=0.01)
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through network.
        
        Parameters
        ----------
        state : torch.Tensor
            Batch of FLA features, shape (batch, 25)
            
        Returns
        -------
        Dict containing distributions and value estimate
        """
        h = self.shared(state)
        
        # Continuous action distributions
        p_junior_mean = torch.sigmoid(self.p_junior_mean(h))
        p_junior_std = F.softplus(self.p_junior_logstd).expand_as(p_junior_mean)
        p_junior_std = torch.clamp(p_junior_std, min=0.01, max=0.5)
        
        delta_kf_mean = 0.2 * torch.tanh(self.delta_kf_mean(h))
        delta_kf_std = F.softplus(self.delta_kf_logstd).expand_as(delta_kf_mean)
        delta_kf_std = torch.clamp(delta_kf_std, min=0.01, max=0.2)
        
        delta_kr_mean = 0.2 * torch.tanh(self.delta_kr_mean(h))
        delta_kr_std = F.softplus(self.delta_kr_logstd).expand_as(delta_kr_mean)
        delta_kr_std = torch.clamp(delta_kr_std, min=0.01, max=0.2)
        
        delta_k_mean = 5.0 * torch.tanh(self.delta_k_mean(h))
        delta_k_std = F.softplus(self.delta_k_logstd).expand_as(delta_k_mean)
        delta_k_std = torch.clamp(delta_k_std, min=0.1, max=3.0)
        
        # Strategy probabilities
        strategy_logits = self.strategy_logits(h)
        strategy_probs = F.softmax(strategy_logits, dim=-1)
        
        # Value estimate
        value = self.value_head(h)
        
        return {
            'p_junior_mean': p_junior_mean,
            'p_junior_std': p_junior_std,
            'delta_kf_mean': delta_kf_mean,
            'delta_kf_std': delta_kf_std,
            'delta_kr_mean': delta_kr_mean,
            'delta_kr_std': delta_kr_std,
            'delta_k_mean': delta_k_mean,
            'delta_k_std': delta_k_std,
            'strategy_logits': strategy_logits,
            'strategy_probs': strategy_probs,
            'value': value,
        }
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[Action, Dict[str, torch.Tensor]]:
        """
        Sample action for execution.
        
        Parameters
        ----------
        state : torch.Tensor
            Single state, shape (1, 25) or (25,)
        deterministic : bool
            If True, use mean actions instead of sampling
            
        Returns
        -------
        action : Action
            Sampled or deterministic action
        info : dict
            Contains log_prob and value for training
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        with torch.no_grad():
            out = self.forward(state)
            
            if deterministic:
                p_junior = out['p_junior_mean'].squeeze()
                delta_kf = out['delta_kf_mean'].squeeze()
                delta_kr = out['delta_kr_mean'].squeeze()
                delta_k = out['delta_k_mean'].squeeze()
                strategy = torch.argmax(out['strategy_probs'], dim=-1).squeeze()
            else:
                # Sample from distributions
                p_junior_dist = Normal(out['p_junior_mean'], out['p_junior_std'])
                delta_kf_dist = Normal(out['delta_kf_mean'], out['delta_kf_std'])
                delta_kr_dist = Normal(out['delta_kr_mean'], out['delta_kr_std'])
                delta_k_dist = Normal(out['delta_k_mean'], out['delta_k_std'])
                strategy_dist = Categorical(out['strategy_probs'])
                
                p_junior = p_junior_dist.sample().squeeze()
                delta_kf = delta_kf_dist.sample().squeeze()
                delta_kr = delta_kr_dist.sample().squeeze()
                delta_k = delta_k_dist.sample().squeeze()
                strategy = strategy_dist.sample().squeeze()
            
            # Clamp actions to valid ranges
            p_junior = torch.clamp(p_junior, 0.0, 1.0)
            delta_kf = torch.clamp(delta_kf, -0.2, 0.2)
            delta_kr = torch.clamp(delta_kr, -0.2, 0.2)
            delta_k = torch.clamp(delta_k, -5.0, 5.0)
            
        action = Action(
            p_junior=float(p_junior.item()),
            delta_kf=float(delta_kf.item()),
            delta_kr=float(delta_kr.item()),
            delta_k=float(delta_k.item()),
            strategy=int(strategy.item()),
        )
        
        info = {
            'value': out['value'].squeeze().item(),
            'p_junior_mean': out['p_junior_mean'].squeeze().item(),
            'delta_kf_mean': out['delta_kf_mean'].squeeze().item(),
        }
        
        return action, info
    
    def evaluate_actions(
        self, 
        states: torch.Tensor, 
        actions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given state-action pairs.
        
        Used during PPO update.
        
        Parameters
        ----------
        states : torch.Tensor
            Batch of states, shape (batch, 25)
        actions : dict
            Dictionary containing batched actions
            
        Returns
        -------
        log_probs : torch.Tensor
            Sum of log probabilities for all action components
        values : torch.Tensor
            State value estimates
        entropy : torch.Tensor
            Sum of entropies for exploration bonus
        """
        out = self.forward(states)
        
        # Create distributions
        p_junior_dist = Normal(out['p_junior_mean'], out['p_junior_std'])
        delta_kf_dist = Normal(out['delta_kf_mean'], out['delta_kf_std'])
        delta_kr_dist = Normal(out['delta_kr_mean'], out['delta_kr_std'])
        delta_k_dist = Normal(out['delta_k_mean'], out['delta_k_std'])
        strategy_dist = Categorical(out['strategy_probs'])
        
        # Compute log probabilities
        log_prob_p_junior = p_junior_dist.log_prob(actions['p_junior'].unsqueeze(-1)).squeeze(-1)
        log_prob_delta_kf = delta_kf_dist.log_prob(actions['delta_kf'].unsqueeze(-1)).squeeze(-1)
        log_prob_delta_kr = delta_kr_dist.log_prob(actions['delta_kr'].unsqueeze(-1)).squeeze(-1)
        log_prob_delta_k = delta_k_dist.log_prob(actions['delta_k'].unsqueeze(-1)).squeeze(-1)
        log_prob_strategy = strategy_dist.log_prob(actions['strategy'])
        
        total_log_prob = (
            log_prob_p_junior + log_prob_delta_kf + 
            log_prob_delta_kr + log_prob_delta_k + log_prob_strategy
        )
        
        # Compute entropy
        entropy = (
            p_junior_dist.entropy().mean() +
            delta_kf_dist.entropy().mean() +
            delta_kr_dist.entropy().mean() +
            delta_k_dist.entropy().mean() +
            strategy_dist.entropy().mean()
        )
        
        return total_log_prob, out['value'].squeeze(-1), entropy
    
    def save(self, path: str) -> None:
        """Save model weights."""
        import platform
        from pathlib import Path
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'platform': platform.system(),
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'GSKPolicyNetwork':
        """
        Load model from checkpoint.
        
        Parameters
        ----------
        path : str
            Path to checkpoint file
        device : str
            Device to load model to ('cpu', 'mps', or 'cuda')
            
        Returns
        -------
        GSKPolicyNetwork
            Loaded model
        """
        # Load checkpoint with proper device mapping
        try:
            # Try with weights_only=True first (safer, PyTorch 2.0+)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            # Older PyTorch version
            checkpoint = torch.load(path, map_location=device)
        
        model = cls(config=checkpoint.get('config', PolicyConfig()))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        return model


# =============================================================================
# Experience Storage
# =============================================================================

@dataclass
class Transition:
    """Single transition for replay buffer."""
    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class ReplayBuffer:
    """
    Experience replay buffer for PPO training.
    
    Stores complete trajectories for on-policy learning.
    """
    
    def __init__(self, max_size: int = 10000):
        self.buffer: deque = deque(maxlen=max_size)
        self.trajectories: List[List[Transition]] = []
        self.current_trajectory: List[Transition] = []
        
    def add(self, transition: Transition) -> None:
        """Add transition to current trajectory."""
        self.current_trajectory.append(transition)
        self.buffer.append(transition)
        
        if transition.done:
            self.trajectories.append(self.current_trajectory)
            self.current_trajectory = []
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored transitions as batched tensors.
        
        Returns
        -------
        dict with keys: states, actions, rewards, dones, log_probs, values
        """
        if len(self.buffer) == 0:
            return {}
            
        transitions = list(self.buffer)
        
        states = torch.tensor(np.array([t.state for t in transitions]), dtype=torch.float32)
        
        actions = {
            'p_junior': torch.tensor([t.action.p_junior for t in transitions], dtype=torch.float32),
            'delta_kf': torch.tensor([t.action.delta_kf for t in transitions], dtype=torch.float32),
            'delta_kr': torch.tensor([t.action.delta_kr for t in transitions], dtype=torch.float32),
            'delta_k': torch.tensor([t.action.delta_k for t in transitions], dtype=torch.float32),
            'strategy': torch.tensor([t.action.strategy for t in transitions], dtype=torch.long),
        }
        
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        old_log_probs = torch.tensor([t.log_prob for t in transitions], dtype=torch.float32)
        old_values = torch.tensor([t.value for t in transitions], dtype=torch.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'old_log_probs': old_log_probs,
            'old_values': old_values,
        }
    
    def clear(self) -> None:
        """Clear buffer after PPO update."""
        self.buffer.clear()
        self.trajectories.clear()
        self.current_trajectory.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# PPO Training Algorithm
# =============================================================================

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for GSK controller.
    
    Implements clipped surrogate objective with GAE advantage estimation.
    
    Parameters
    ----------
    policy : GSKPolicyNetwork
        Policy network to train
    config : PolicyConfig, optional
        Training configuration
    device : str
        Device to use ('cpu' or 'cuda')
    """
    
    def __init__(
        self, 
        policy: GSKPolicyNetwork,
        config: Optional[PolicyConfig] = None,
        device: str = 'cpu'
    ):
        self.policy = policy.to(device)
        self.config = config or policy.config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            policy.parameters(), 
            lr=self.config.lr,
            eps=1e-5
        )
        
        self.buffer = ReplayBuffer()
        
        # Training statistics
        self.n_updates = 0
        self.training_stats: List[Dict] = []
        
    def compute_gae(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor,
        next_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Parameters
        ----------
        rewards : torch.Tensor
            Rewards for each timestep
        values : torch.Tensor
            Value estimates for each timestep
        dones : torch.Tensor
            Episode termination flags
        next_value : float
            Bootstrap value for last state
            
        Returns
        -------
        advantages : torch.Tensor
        returns : torch.Tensor
        """
        n = len(rewards)
        advantages = torch.zeros(n, dtype=torch.float32)
        
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform PPO update on collected experience.
        
        Parameters
        ----------
        n_epochs : int
            Number of passes through the data
        batch_size : int
            Mini-batch size for updates
            
        Returns
        -------
        dict with training statistics
        """
        if len(self.buffer) < batch_size:
            return {}
        
        data = self.buffer.get_batch()
        
        states = data['states'].to(self.device)
        actions = {k: v.to(self.device) for k, v in data['actions'].items()}
        rewards = data['rewards']
        dones = data['dones']
        old_log_probs = data['old_log_probs'].to(self.device)
        old_values = data['old_values']
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, old_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # PPO update epochs
        n_samples = len(states)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0
        
        for epoch in range(n_epochs):
            indices = torch.randperm(n_samples)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = {k: v[batch_idx] for k, v in actions.items()}
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                
                # Evaluate actions
                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.clip_eps, 
                    1 + self.config.clip_eps
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.config.value_coef * value_loss - 
                    self.config.entropy_coef * entropy
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_batches += 1
        
        self.n_updates += 1
        
        stats = {
            'policy_loss': total_policy_loss / n_batches,
            'value_loss': total_value_loss / n_batches,
            'entropy': total_entropy / n_batches,
            'n_samples': n_samples,
        }
        self.training_stats.append(stats)
        
        # Clear buffer after update
        self.buffer.clear()
        
        return stats
    
    def add_experience(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float
    ) -> None:
        """Add experience to replay buffer."""
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
        )
        self.buffer.add(transition)
    
    def get_training_stats(self) -> List[Dict]:
        """Return training statistics history."""
        return self.training_stats


# =============================================================================
# Reward Function
# =============================================================================

class RewardCalculator:
    """
    Multi-objective reward function for GSK training.
    
    Components:
    1. Fitness improvement (primary objective)
    2. Diversity maintenance (prevent premature convergence)
    3. Escape bonus (reward for escaping local optima)
    4. Stagnation penalty (discourage no progress)
    
    Parameters
    ----------
    alpha : float
        Weight for fitness improvement
    beta : float
        Weight for diversity maintenance
    gamma : float
        Weight for escape bonus
    delta : float
        Weight for stagnation penalty
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.3,
        delta: float = 0.2,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        # State tracking
        self.prev_best = None
        self.prev_diversity = None
        self.best_ever = float('inf')
        self.stagnation_counter = 0
        
    def compute(
        self,
        current_best: float,
        current_diversity: float,
        improved_ratio: float = 0.0,
    ) -> float:
        """
        Compute reward for current generation.
        
        Parameters
        ----------
        current_best : float
            Best fitness in current generation
        current_diversity : float
            Population diversity measure [0, 1]
        improved_ratio : float
            Fraction of individuals that improved
            
        Returns
        -------
        float : Reward value
        """
        reward = 0.0
        
        # 1. Fitness improvement reward
        if self.prev_best is not None:
            # Relative improvement
            improvement = (self.prev_best - current_best) / (abs(self.prev_best) + 1e-10)
            reward += self.alpha * np.clip(improvement * 100, -1, 1)  # Scale up small improvements
        
        # 2. Diversity reward (encourage exploration when stagnating)
        if self.prev_diversity is not None and self.stagnation_counter > 3:
            diversity_change = current_diversity - self.prev_diversity
            reward += self.beta * np.clip(diversity_change * 10, 0, 0.5)
        
        # 3. Escape local optima bonus
        if current_best < self.best_ever * 0.999:  # Meaningful improvement
            if self.stagnation_counter > 5:
                reward += self.gamma * min(1.0, self.stagnation_counter / 10)
            self.best_ever = current_best
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        # 4. Stagnation penalty
        if self.stagnation_counter > 10:
            penalty = self.delta * (self.stagnation_counter - 10) / 50
            reward -= np.clip(penalty, 0, 0.3)
        
        # 5. Population improvement bonus
        reward += 0.05 * improved_ratio
        
        # Update state
        self.prev_best = current_best
        self.prev_diversity = current_diversity
        
        return float(reward)
    
    def reset(self) -> None:
        """Reset for new optimization run."""
        self.prev_best = None
        self.prev_diversity = None
        self.best_ever = float('inf')
        self.stagnation_counter = 0
