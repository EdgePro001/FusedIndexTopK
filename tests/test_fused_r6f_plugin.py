from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r6f.plugin import create_variant


def _case(*, context: int = 16384, queries: int = 4096) -> PrefillCase:
    return PrefillCase(
        case_id="r6f-test",
        query_tokens=queries,
        context_tokens=context,
        top_k=2048,
        seed=1,
    )


def test_fused_r6f_is_exact_and_uses_sparse_direct_atomics() -> None:
    plugin = create_variant()
    assert plugin.descriptor.plugin_id == "deepgemm_fused_candidate_topk_r6f"
    assert plugin.descriptor.exact_topk is True
    assert "partition-third-histogram-fusion" in plugin.descriptor.tags
    assert "sparse-direct-shared-atomic" in plugin.descriptor.tags
    assert plugin.supports(_case())
    assert not plugin.supports(_case(context=32768))
    assert not plugin.supports(_case(queries=4095))


def test_fused_r6f_fingerprint_declares_single_delta() -> None:
    metadata = create_variant().fingerprint_metadata()
    assert metadata["upstream_deepgemm_modified"] is False
    assert metadata["timed_repair"] is True
    assert "direct shared atomic" in metadata["single_control_delta"]
