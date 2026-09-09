from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r16a.plugin import (
    create_variant,
    long_context_sample_elements,
)


def _case(context_tokens: int) -> PrefillCase:
    return PrefillCase(
        case_id=f"r16a-n{context_tokens}",
        query_tokens=4096,
        context_tokens=context_tokens,
        top_k=2048,
        seed=1,
    )


def test_r16a_preserves_16k_and_supports_long_contexts() -> None:
    plugin = create_variant()
    metadata = plugin.fingerprint_metadata()
    for context in (16384, 32768, 40000, 65536, 131072, 262144):
        assert plugin.supports(_case(context))
    assert plugin.descriptor.exact_topk is True
    assert metadata["fast_path_delta_from_r13a"] == "none"
    assert metadata["long_repair_chunk_elements"] == 16384
    assert metadata["long_repair_host_flag_read"] is False


def test_r16a_long_context_sample_schedule() -> None:
    assert long_context_sample_elements(32768) == 512
    assert long_context_sample_elements(40000) == 1024
    assert long_context_sample_elements(65536) == 1024
    assert long_context_sample_elements(131072) == 1536
    assert long_context_sample_elements(262144) == 3072
