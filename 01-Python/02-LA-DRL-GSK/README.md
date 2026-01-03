# LA-DRL-GSK: Landscape-Aware Deep RL Controller for GSK

**Q1 Implementation** for continuous black-box optimization.

LA-DRL-GSK combines the Gaining-Sharing Knowledge (GSK) algorithm with reinforcement learning-based parameter control, using online landscape analysis features to adapt algorithm behavior.

## Features

- **25 Zero-Cost FLA Features**: Population and fitness-based landscape analysis with no extra evaluations
- **Windowed Control**: RL decision every W generations for stable training
- **Action Space**: Controls K (knowledge rate), kf (step size), kr (crossover rate), p (stratification)
- **Multiple Controllers**: Fixed (baseline), Heuristic (no deps), SB3 PPO (trained policy)
- **NumPy-First Design**: Fast vectorized operations, optional PyTorch for neural policies

## Installation

```bash
# Minimal (NumPy only)
pip install -r requirements-minimal.txt

# Full (with RL training)
pip install -r requirements.txt
pip install gymnasium stable-baselines3
```

## Quick Start

### Baseline GSK
```python
from la_drl_gsk import create_baseline_gsk

optimizer = create_baseline_gsk(dim=30, max_nfes=300000, seed=42)
result = optimizer.optimize(objective_function)
print(f"Best fitness: {result.best_f}")
```

### LA-DRL-GSK with Heuristic Controller
```python
from la_drl_gsk import create_ladrl_gsk

optimizer = create_ladrl_gsk(
    dim=30, 
    max_nfes=300000, 
    controller_backend="heuristic"
)
result = optimizer.optimize(objective_function)
```

### LA-DRL-GSK with Trained SB3 Policy
```python
from la_drl_gsk import create_ladrl_gsk

optimizer = create_ladrl_gsk(
    dim=30, 
    controller_backend="sb3",
    policy_path="models/ppo_gsk_D30.zip"
)
result = optimizer.optimize(objective_function)
```

## CLI Commands

```bash
# Demo
python run.py demo
python run.py demo --detailed

# Train (requires gymnasium + stable-baselines3)
python run.py train --dims 10 --timesteps 200000

# Evaluate
python run.py evaluate --dims 10 30 --runs 30
```

## Q1 Action Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| K | [1, 20] | Knowledge rate (junior→senior) |
| kf | [0.05, 1.0] | Step size |
| kr | [0.05, 0.99] | Crossover rate |
| p | [0.05, 0.20] | Stratification fraction |

## License

MIT License
