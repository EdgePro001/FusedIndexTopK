"""Experimental fused R11d exact 9+7-bit prefix-radix reducer."""

from index_topk_perflab.experimental.fused_r10a import (
    common_random_sample_ids,
    guarded_sample_rank,
    sample_elements_for_context,
)

from .candidate_reducer import load_segmented_candidate_reducer
from .plugin import DeepGemmFusedCandidateR11d, create_variant

__all__ = [
    "DeepGemmFusedCandidateR11d",
    "common_random_sample_ids",
    "create_variant",
    "guarded_sample_rank",
    "load_segmented_candidate_reducer",
    "sample_elements_for_context",
]
