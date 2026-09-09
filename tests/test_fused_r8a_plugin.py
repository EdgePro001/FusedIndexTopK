from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r8a.plugin import create_variant


def _case(*, context: int = 16384, queries: int = 4096) -> PrefillCase:
    return PrefillCase(
        case_id="r8a-test",
        query_tokens=queries,
        context_tokens=context,
        top_k=2048,
        seed=1,
    )


def test_fused_r8a_is_exact_and_changes_only_threshold_launch_shape() -> None:
    plugin = create_variant()
    assert plugin.descriptor.plugin_id == "deepgemm_fused_candidate_topk_r8a"
    assert plugin.descriptor.exact_topk is True
    assert "sample-threshold-128-threads" in plugin.descriptor.tags
    assert plugin.supports(_case())
    assert not plugin.supports(_case(context=32768))
    assert not plugin.supports(_case(queries=4095))


def test_fused_r8a_fingerprint_declares_single_delta() -> None:
    metadata = create_variant().fingerprint_metadata()
    assert metadata["threshold_threads"] == 128
    assert metadata["threshold_sample_count"] == 128
    assert "each thread clears two histogram bins" in metadata["single_control_delta"]
