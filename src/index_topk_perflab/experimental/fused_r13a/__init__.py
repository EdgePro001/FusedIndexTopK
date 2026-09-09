"""R13a producer occupancy experiment."""

from .plugin import DeepGemmFusedCandidateR13a, create_variant
from .producer import load_candidate_producer

__all__ = [
    "DeepGemmFusedCandidateR13a",
    "create_variant",
    "load_candidate_producer",
]
