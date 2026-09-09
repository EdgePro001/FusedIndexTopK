from __future__ import annotations

import sys

import pytest

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.experimental.fused_r5i.plugin import create_variant
from index_topk_perflab.experimental.fused_r5i.sampling import (
    guarded_sample_rank,
    sample_elements_for_context,
)


def _case(**overrides: int | bool | str) -> PrefillCase:
    values: dict[str, int | bool | str] = {
        "case_id": "q4096-n16384-k2048",
        "query_tokens": 4096,
        "context_tokens": 16384,
        "top_k": 2048,
        "seed": 1234,
    }
    values.update(overrides)
    return PrefillCase(**values)  # type: ignore[arg-type]


def test_fused_r5i_descriptor_is_honest_and_cuda_lazy() -> None:
    torch_before = sys.modules.get("torch")
    deep_gemm_before = sys.modules.get("deep_gemm")

    plugin = create_variant()
    metadata = plugin.fingerprint_metadata()

    assert plugin.descriptor.plugin_id == "deepgemm_fused_candidate_topk_r5i"
    assert plugin.descriptor.mode == "fused"
    assert plugin.descriptor.exact_topk is True
    assert plugin.descriptor.implementation_version == (
        "producer-fused-sampled-r5i-masked-exact-repair-v3"
    )
    assert metadata["algorithm"] == (
        "producer-fused-sampled-r5i-masked-exact-repair-v3"
    )
    assert metadata["upstream_deepgemm_modified"] is False
    assert metadata["timed_repair"] is True
    assert metadata["repair_dispatch"] == "device-mask-no-host-sync"
    assert sys.modules.get("torch") is torch_before
    assert sys.modules.get("deep_gemm") is deep_gemm_before


def test_fused_r5i_supports_only_the_specialized_shape_contract() -> None:
    plugin = create_variant()

    assert plugin.supports(_case())
    assert not plugin.supports(_case(query_tokens=4095))
    assert not plugin.supports(_case(context_tokens=32768))
    assert not plugin.supports(_case(top_k=1024))
    assert not plugin.supports(_case(head_dim=64))
    assert not plugin.supports(_case(causal=False))


def test_fused_r5i_sampling_schedule_matches_measured_h20_case() -> None:
    sample_elements = sample_elements_for_context(16384)

    assert sample_elements == 128
    assert guarded_sample_rank(
        sample_elements,
        16384,
        target_candidates=3072,
    ) == 33


@pytest.mark.parametrize(
    ("sample_elements", "context_tokens", "target_candidates"),
    ((0, 16384, 3072), (129, 128, 64), (128, 16384, 0), (128, 16384, 16385)),
)
def test_fused_r5i_sample_rank_rejects_invalid_contracts(
    sample_elements: int,
    context_tokens: int,
    target_candidates: int,
) -> None:
    with pytest.raises(ValueError):
        guarded_sample_rank(
            sample_elements,
            context_tokens,
            target_candidates=target_candidates,
        )


def test_fused_r5i_sample_schedule_rejects_empty_context() -> None:
    with pytest.raises(ValueError):
        sample_elements_for_context(0)
