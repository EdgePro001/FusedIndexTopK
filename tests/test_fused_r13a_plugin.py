from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r13a.plugin import create_variant


def test_r13a_declares_half_width_two_cta_producer() -> None:
    plugin = create_variant()
    case = PrefillCase(
        case_id="r13a-test",
        query_tokens=4096,
        context_tokens=16384,
        top_k=2048,
        seed=1,
    )
    metadata = plugin.fingerprint_metadata()
    assert plugin.descriptor.exact_topk is True
    assert plugin.supports(case)
    assert metadata["producer_block_kv"] == 128
    assert metadata["producer_math_threads"] == 256
    assert metadata["producer_grid_sms_multiplier"] == 2
    assert metadata["producer_logical_candidate_segments"] == 16
    assert metadata["producer_segments_per_math_warp"] == 2
    assert metadata["upstream_deepgemm_modified"] is False
