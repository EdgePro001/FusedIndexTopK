from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r11d_nsweep.plugin import create_variant


def _case(context_tokens: int) -> PrefillCase:
    return PrefillCase(
        case_id=f"r11d-n{context_tokens}",
        query_tokens=4096,
        context_tokens=context_tokens,
        top_k=2048,
        seed=1,
    )


def test_r11d_nsweep_supports_frozen_context_matrix() -> None:
    plugin = create_variant()
    assert all(plugin.supports(_case(context)) for context in (6144, 8192, 12288, 16384))
    assert not plugin.supports(_case(5888))
    assert not plugin.supports(_case(16640))
    metadata = plugin.fingerprint_metadata()
    assert metadata["device_kernel_delta_from_r11d"] == "none"
    assert metadata["prefix_radix_bits"] == [9, 7, 8, 8]
