from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r11d.plugin import create_variant


def test_r11d_declares_exact_nine_plus_seven_prefix_radix() -> None:
    plugin = create_variant()
    case = PrefillCase(
        case_id="r11d-test",
        query_tokens=4096,
        context_tokens=16384,
        top_k=2048,
        seed=1,
    )
    metadata = plugin.fingerprint_metadata()
    assert plugin.descriptor.exact_topk is True
    assert plugin.supports(case)
    assert metadata["fast_working_capacity"] == 6656
    assert metadata["prefix_radix_bits"] == [9, 7, 8, 8]
