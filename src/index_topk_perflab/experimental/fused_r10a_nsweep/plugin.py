"""Variable-N evidence probe around the frozen R10a CUDA implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PrefillCase, PreparedGraph, RunMode, VariantDescriptor
from index_topk_perflab.experimental.fused_r5i import plugin as r5i_plugin
from index_topk_perflab.experimental.fused_r10a.plugin import (
    DeepGemmFusedCandidateR10a,
)
from index_topk_perflab.provenance import path_fingerprint
from index_topk_perflab.variants.common import supports_frozen_deepgemm_case

from .producer import load_candidate_producer

_ROOT = Path(__file__).resolve().parent
_MIN_CONTEXT = 6144
_MAX_CONTEXT = 16384


class DeepGemmFusedCandidateR10aNSweep(DeepGemmFusedCandidateR10a):
    """R10a device code with only its repair host guard generalized."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r10a_nsweep",
        display_name="DeepGEMM fused candidate TopK R10a variable-N probe",
        api_version="1.0",
        implementation_version="r10a-device-plus-host-nle16384-probe-v1",
        mode="fused",
        description=(
            "Frozen R10a device kernels with a host-only repair guard allowing "
            "256-aligned contexts from 6144 through 16384"
        ),
        implementation="project-local-r10a-device-byte-identical+host-probe",
        exact_topk=True,
        source_revision="r10a:f9d140a;host-nle16384-probe:v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "r10a-device-byte-identical",
            "variable-n-probe",
            "host-guard-only-delta",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def supports(self, case: PrefillCase) -> bool:
        first_row_valid = case.context_tokens - case.query_tokens + 1
        return (
            supports_frozen_deepgemm_case(case)
            and _MIN_CONTEXT <= case.context_tokens <= _MAX_CONTEXT
            and case.context_tokens % 256 == 0
            and case.top_k == 2048
            and case.query_tokens % 2 == 0
            and first_row_valid >= case.top_k
        )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r10a-device-plus-host-nle16384-probe-v1",
            "nsweep_adapter_sources": path_fingerprint(_ROOT),
            "device_kernel_delta_from_r10a": "none",
            "host_delta_from_r10a": (
                "repair seq_len_kv assertion relaxed from ==16384 to "
                "0<N<=16384"
            ),
            "supported_context_range": [_MIN_CONTEXT, _MAX_CONTEXT],
            "context_alignment": 256,
            "repair_completeness_argument": (
                "16 warp-owned segments x 1024 entries cover every runtime "
                "N<=16384"
            ),
        }

    def prepare(
        self,
        case: PrefillCase,
        inputs: Any,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        frozen_loader = r5i_plugin.load_candidate_producer
        r5i_plugin.load_candidate_producer = load_candidate_producer
        try:
            graph = super().prepare(case, inputs, options=options, mode=mode)
        finally:
            r5i_plugin.load_candidate_producer = frozen_loader
        graph.metadata = {
            **graph.metadata,
            "probe_role": "variable-N sensitivity only",
            "device_kernel_delta_from_r10a": "none",
            "host_repair_context_guard": "0<N<=16384",
            "repair_runtime_context_tokens": case.context_tokens,
            "promotion_status": "exact-experimental-n-sweep-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR10aNSweep:
    return DeepGemmFusedCandidateR10aNSweep(options)
