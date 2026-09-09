"""R16a long-context exact extension of the R13a fast path."""

from .long_repair import load_long_context_repair
from .plugin import DeepGemmFusedCandidateR16a, create_variant

__all__ = [
    "DeepGemmFusedCandidateR16a",
    "create_variant",
    "load_long_context_repair",
]
