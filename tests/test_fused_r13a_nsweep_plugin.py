from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r13a_nsweep.plugin import create_variant


def test_r13a_nsweep_supports_frozen_context_matrix() -> None:
    plugin = create_variant()
    metadata = plugin.fingerprint_metadata()
    assert plugin.descriptor.exact_topk is True
    for context in (6144, 8192, 12288, 16384):
        case = PrefillCase(
            case_id=f"r13a-nsweep-{context}",
            query_tokens=4096,
            context_tokens=context,
            top_k=2048,
            seed=1,
        )
        assert plugin.supports(case)
    assert metadata["producer_block_kv"] == 128
    assert metadata["producer_math_threads"] == 256
    assert metadata["fast_producer_device_delta_from_r13a"] == "none"
    assert metadata["reducer_device_delta_from_r11d"] == "none"
