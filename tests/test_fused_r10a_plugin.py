from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r10a.plugin import create_variant


def test_r10a_declares_bounded_exact_fast_working_set() -> None:
    plugin = create_variant()
    case = PrefillCase(
        case_id="r10a-test",
        query_tokens=4096,
        context_tokens=16384,
        top_k=2048,
        seed=1,
    )
    metadata = plugin.fingerprint_metadata()
    assert plugin.descriptor.exact_topk is True
    assert plugin.supports(case)
    assert metadata["fast_input_capacity"] == 14080
    assert metadata["fast_working_capacity"] == 6656
    assert metadata["fast_block_threads"] == 512
    assert "exact repair flag" in metadata["single_control_delta"]
