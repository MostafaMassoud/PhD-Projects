"""Baseline GSK optimizer package.

This package intentionally contains **only** the baseline Gaining–Sharing
Knowledge (GSK) algorithm and the minimal utilities required to:

- run reproducible experiments on the CEC2017 benchmark suite, and
- validate outputs against provided reference CSV summaries.

No reinforcement learning (RL), no model-based (MB) components, and no hybrid
extensions are included.
"""

from .gsk import GSKConfig, GSKResult, gsk_optimize
from .experiment import run_cec2017_experiments
from .validation import validate_against_reference

__all__ = [
    "GSKConfig",
    "GSKResult",
    "gsk_optimize",
    "run_cec2017_experiments",
    "validate_against_reference",
]
