"""Variable-N evidence adapter for the R13a producer."""

from .plugin import DeepGemmFusedCandidateR13aNSweep, create_variant

__all__ = ["DeepGemmFusedCandidateR13aNSweep", "create_variant"]
