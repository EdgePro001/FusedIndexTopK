from __future__ import annotations

import pytest

from fused_index_topk.api import PrefillCase
from fused_index_topk.kernel.onchip_producer import _IMPLEMENTATIONS
from fused_index_topk.kernel.plugin import (
    create_variant,
    long_context_sample_elements,
    production_sample_elements,
)
from fused_index_topk.kernel.sampling import (
    guarded_sample_rank,
)
from fused_index_topk.registry import available_variants, load_variant


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


def test_final_operator_supports_qualified_contexts_with_one_kernel() -> None:
    plugin = create_variant()
    for context in (8192, 12288, 16384, 32768, 65536, 131072, 163840):
        assert plugin.supports(_case(context))
    for context in (6144, 163968, 262144):
        assert not plugin.supports(_case(context))


def test_long_context_sample_schedule() -> None:
    assert long_context_sample_elements(32768) == 512
    assert long_context_sample_elements(40000) == 1024
    assert long_context_sample_elements(65536) == 1024
    assert long_context_sample_elements(131072) == 1536
    assert long_context_sample_elements(163840) == 2048


def test_unified_kernel_preserves_the_qualified_sampling_schedule() -> None:
    assert tuple(_IMPLEMENTATIONS) == ("long",)
    assert production_sample_elements(8192) == 512
    assert production_sample_elements(16384) == 256
    assert production_sample_elements(32768) == 512


def test_guarded_sample_rank_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError):
        guarded_sample_rank(0, 16384)
    with pytest.raises(ValueError):
        guarded_sample_rank(256, 16384, target_candidates=0)
