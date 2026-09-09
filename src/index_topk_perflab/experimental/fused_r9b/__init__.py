"""R9a with a 2816-candidate sampling target."""

from index_topk_perflab.experimental.fused_r9a import (
    common_random_sample_ids,
    guarded_sample_rank,
    load_candidate_producer,
    load_sampling_extension,
    load_segmented_candidate_reducer,
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
