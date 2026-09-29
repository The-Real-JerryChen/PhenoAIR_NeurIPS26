"""Optional, strategy-neutral building blocks for self-evolution."""

from .models import EvolutionProposal, FailureCluster
from .pipeline import EvolutionEngine
from .trajectories import compress_failure, compress_success

__all__ = [
    "EvolutionEngine",
    "EvolutionProposal",
    "FailureCluster",
    "compress_failure",
    "compress_success",
]

