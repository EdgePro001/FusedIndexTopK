"""Variable-N evidence adapter around the R11d CUDA implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PrefillCase, PreparedGraph, RunMode, VariantDescriptor
from index_topk_perflab.experimental.fused_r5i import plugin as r5i_plugin
from index_topk_perflab.experimental.fused_r10a_nsweep.producer import (
    load_candidate_producer,
)
from index_topk_perflab.experimental.fused_r11d.plugin import (
    DeepGemmFusedCandidateR11d,
)
from index_topk_perflab.provenance import path_fingerprint
from index_topk_perflab.variants.common import supports_frozen_deepgemm_case

_ROOT = Path(__file__).resolve().parent
_MIN_CONTEXT = 6144
_MAX_CONTEXT = 16384


class DeepGemmFusedCandidateR11dNSweep(DeepGemmFusedCandidateR11d):
    """R11d device code with only its repair host guard generalized."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r11d_nsweep",
        display_name="DeepGEMM fused candidate TopK R11d variable-N probe",
        api_version="1.0",
        implementation_version="r11d-device-plus-host-nle16384-probe-v1",
        mode="fused",
        description=(
            "Frozen R11d device kernels with the R10a host-only repair adapter "
            "allowing 256-aligned contexts from 6144 through 16384"
        ),
        implementation="project-local-r11d-device-byte-identical+host-probe",
        exact_topk=True,
        source_revision="r11d:local;host-nle16384-probe:v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "r11d-device-byte-identical",
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
            "algorithm": "r11d-device-plus-host-nle16384-probe-v1",
            "nsweep_adapter_sources": path_fingerprint(_ROOT),
            "device_kernel_delta_from_r11d": "none",
            "host_delta_from_r11d": (
                "repair seq_len_kv assertion relaxed from ==16384 to 0<N<=16384"
            ),
            "supported_context_range": [_MIN_CONTEXT, _MAX_CONTEXT],
            "context_alignment": 256,
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
            "device_kernel_delta_from_r11d": "none",
            "host_repair_context_guard": "0<N<=16384",
            "repair_runtime_context_tokens": case.context_tokens,
            "promotion_status": "exact-experimental-n-sweep-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR11dNSweep:
    return DeepGemmFusedCandidateR11dNSweep(options)
