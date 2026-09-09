"""Experimental producer-side DeepGEMM/R5i fusion components."""

from .candidate_reducer import (
    load_candidate_reducer,
    load_segmented_candidate_reducer,
)
from .producer import load_candidate_producer
from .sampling import (
    common_random_sample_ids,
    guarded_sample_rank,
    load_sampling_extension,
    sample_elements_for_context,
)

__all__ = [
    "load_candidate_producer",
    "load_candidate_reducer",
    "load_segmented_candidate_reducer",
    "load_sampling_extension",
    "sample_elements_for_context",
    "guarded_sample_rank",
    "common_random_sample_ids",
]
