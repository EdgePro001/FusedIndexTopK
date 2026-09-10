from __future__ import annotations

import pytest

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_index_topk.plugin import (
    create_variant,
    long_context_sample_elements,
)
from index_topk_perflab.experimental.fused_index_topk.sampling import (
    guarded_sample_rank,
)
from index_topk_perflab.registry import available_variants, load_variant


def _case(context_tokens: int) -> PrefillCase:
    return PrefillCase(
        case_id=f"fused-index-topk-n{context_tokens}",
        query_tokens=4096,
        context_tokens=context_tokens,
        top_k=2048,
        seed=1,
    )


def test_public_registry_exposes_one_fused_operator() -> None:
    fused = [name for name in available_variants() if name == "fused_index_topk"]
    assert fused == ["fused_index_topk"]
    plugin = load_variant("fused_index_topk")
    assert plugin.descriptor.plugin_id == "fused_index_topk"
    assert plugin.descriptor.display_name == "FusedIndexTopK"
    assert plugin.descriptor.exact_topk is True


def test_final_operator_supports_short_and_long_contexts() -> None:
    plugin = create_variant()
    for context in (6144, 8192, 12288, 16384, 32768, 65536, 131072, 262144):
        assert plugin.supports(_case(context))
    assert not plugin.supports(_case(6000))


def test_long_context_sample_schedule() -> None:
    assert long_context_sample_elements(32768) == 512
    assert long_context_sample_elements(40000) == 1024
    assert long_context_sample_elements(65536) == 1024
    assert long_context_sample_elements(131072) == 1536
    assert long_context_sample_elements(262144) == 3072


def test_guarded_sample_rank_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError):
        guarded_sample_rank(0, 16384)
    with pytest.raises(ValueError):
        guarded_sample_rank(256, 16384, target_candidates=0)
