"""R9a: fill DeepGEMM's existing 256-wide sampled-KV tile."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PreparedGraph, RunMode, VariantDescriptor
from index_topk_perflab.experimental.fused_r5i import plugin as r5i_plugin
from index_topk_perflab.experimental.fused_r8a.plugin import DeepGemmFusedCandidateR8a
from index_topk_perflab.provenance import path_fingerprint

_ROOT = Path(__file__).resolve().parent
_SAMPLE_ELEMENTS = 256


def sample_elements_for_context(context_tokens: int) -> int:
    """Use 256 samples, which already fit one DeepGEMM BLOCK_KV tile."""

    if context_tokens < _SAMPLE_ELEMENTS:
        raise ValueError("R9a requires at least 256 context tokens")
    return _SAMPLE_ELEMENTS


class DeepGemmFusedCandidateR9a(DeepGemmFusedCandidateR8a):
    """R8a with twice as many samples and the unchanged 3072 target."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r9a",
        display_name="DeepGEMM fused candidate TopK R9a",
        api_version="1.0",
        implementation_version="r8a-plus-sample256-r9a-v1",
        mode="fused",
        description=(
            "Exact R8a pipeline using all 256 columns of DeepGEMM's sampled "
            "BLOCK_KV tile while preserving the 3072 candidate target"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="r8a:local;sample256:r9a-v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "producer-fusion",
            "no-dense-logits",
            "random-token-sampling-256",
            "sample-threshold-128-threads",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r8a-plus-sample256-r9a-v1",
            "r9a_sources": path_fingerprint(_ROOT),
            "sample_elements": _SAMPLE_ELEMENTS,
            "sample_target_candidates": 3072,
            "deepgemm_sample_block_kv": 256,
            "single_control_delta": (
                "increase independent token samples from 128 to 256 while "
                "preserving the target and exact repair path"
            ),
        }

    def prepare(
        self,
        case: Any,
        inputs: Any,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        frozen_schedule = r5i_plugin.sample_elements_for_context
        r5i_plugin.sample_elements_for_context = sample_elements_for_context
        try:
            graph = super().prepare(case, inputs, options=options, mode=mode)
        finally:
            r5i_plugin.sample_elements_for_context = frozen_schedule

        graph.metadata = {
            **graph.metadata,
            "parent_candidate": "fused_r8a-threshold128threads-v1",
            "sample_schedule_revision": "r9a-fixed-256-independent-tokens-v1",
            "sample_elements": _SAMPLE_ELEMENTS,
            "sample_deepgemm_tile_utilization": "256-of-256",
            "promotion_status": "exact-experimental-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR9a:
    return DeepGemmFusedCandidateR9a(options)
