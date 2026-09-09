"""Host-only variable-N probe for the fused R11d device kernels."""

from .plugin import DeepGemmFusedCandidateR11dNSweep, create_variant

__all__ = ["DeepGemmFusedCandidateR11dNSweep", "create_variant"]
