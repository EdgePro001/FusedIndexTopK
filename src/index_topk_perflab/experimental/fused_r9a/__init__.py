"""R8a with 256 independent token samples."""

from index_topk_perflab.experimental.fused_r8a import (
    common_random_sample_ids,
    guarded_sample_rank,
    load_candidate_producer,
    load_sampling_extension,
    load_segmented_candidate_reducer,
)

from .plugin import sample_elements_for_context

__all__ = [
    "common_random_sample_ids",
    "guarded_sample_rank",
    "load_candidate_producer",
    "load_sampling_extension",
    "load_segmented_candidate_reducer",
    "sample_elements_for_context",
]
