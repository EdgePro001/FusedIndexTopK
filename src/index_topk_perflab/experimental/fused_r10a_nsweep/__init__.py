"""Host-only variable-N probe for the frozen R10a device kernels."""

from .plugin import DeepGemmFusedCandidateR10aNSweep, create_variant

__all__ = ["DeepGemmFusedCandidateR10aNSweep", "create_variant"]
