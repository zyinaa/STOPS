"""STOPS: Sequential Tests for Online stopping and Plateau-based Selection.

When to stop a self-evolving LLM loop, and which artifact to return.

Reference implementation of "When Is Enough Enough in Self-Evolving LLM Systems?"
(Yin, Liu, Qi).
"""

from .betting import AGRAPA, FixedMixture
from .evaluate import load_outcomes, paired_comparison
from .monitor import Monitor, RoundRecord, paired_counts
from .trace import ReplayResult, Trace

__all__ = [
    "Monitor",
    "RoundRecord",
    "paired_counts",
    "AGRAPA",
    "FixedMixture",
    "Trace",
    "ReplayResult",
    "load_outcomes",
    "paired_comparison",
]

__version__ = "0.1.0"
