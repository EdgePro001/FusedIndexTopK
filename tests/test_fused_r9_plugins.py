from __future__ import annotations

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r9a.plugin import create_variant as r9a
from index_topk_perflab.experimental.fused_r9b.plugin import create_variant as r9b


def _case() -> PrefillCase:
    return PrefillCase(
        case_id="r9-test",
        query_tokens=4096,
        context_tokens=16384,
        top_k=2048,
        seed=1,
    )


def test_r9a_fills_one_deepgemm_sample_tile() -> None:
    plugin = r9a()
    metadata = plugin.fingerprint_metadata()
    assert plugin.descriptor.exact_topk is True
    assert plugin.supports(_case())
    assert metadata["sample_elements"] == 256
    assert metadata["deepgemm_sample_block_kv"] == 256
    assert metadata["sample_target_candidates"] == 3072


def test_r9b_lowers_only_the_r9a_candidate_target() -> None:
    plugin = r9b()
    metadata = plugin.fingerprint_metadata()
    assert plugin.descriptor.exact_topk is True
    assert plugin.supports(_case())
    assert metadata["sample_elements"] == 256
    assert metadata["target_candidates"] == 2816
    assert "preserve all exact guard" in metadata["single_control_delta"]
