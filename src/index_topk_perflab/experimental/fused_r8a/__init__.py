"""R6f with a 128-thread sampled-threshold kernel."""

from index_topk_perflab.experimental.fused_r5i import load_candidate_producer
from index_topk_perflab.experimental.fused_r6f import load_segmented_candidate_reducer

from .sampling import (
    common_random_sample_ids,
    guarded_sample_rank,
    load_sampling_extension,
    sample_elements_for_context,
)

__all__ = [
    "common_random_sample_ids",
    "guarded_sample_rank",
    "load_candidate_producer",
    "load_sampling_extension",
    "load_segmented_candidate_reducer",
    "sample_elements_for_context",
]
