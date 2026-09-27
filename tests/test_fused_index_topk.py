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


def test_frozen_q1_release_identity_and_scope() -> None:
    import fused_index_topk

    plugin = create_variant()
    assert fused_index_topk.__version__ == "0.3.0"
    assert plugin.descriptor.implementation_version == "2.3.0"
    assert plugin.descriptor.source_revision == "fused-index-topk-v2.3-q1"
    assert plugin.block_q == 1
    assert plugin.sample_guard_sigmas == 2.0
    for option in ("conditional_repair", "sample_elements_override"):
        with pytest.raises(ValueError, match="unknown"):
            create_variant({option: True})


def test_final_operator_supports_qualified_contexts_with_one_kernel() -> None:
    plugin = create_variant()
    assert plugin.block_q == 1
    for context in (8192, 12288, 16384, 32768, 65536, 131072, 163840):
        assert plugin.supports(_case(context))
    for context in (6144, 163968, 262144):
        assert not plugin.supports(_case(context))


def test_block_q_two_remains_an_explicit_control() -> None:
    plugin = create_variant({"block_q": 2})
    assert plugin.block_q == 2
    assert plugin.supports(_case(16384))


def test_q1_default_does_not_expand_the_repair_contract_to_odd_queries() -> None:
    odd = PrefillCase(
        case_id="odd-query-control", query_tokens=63,
        context_tokens=8192, top_k=2048, seed=1,
    )
    assert not create_variant().supports(odd)


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
