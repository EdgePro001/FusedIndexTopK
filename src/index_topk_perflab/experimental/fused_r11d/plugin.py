"""R11d: replace the first 8+8 prefix radix with an exact 9+7 split."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PreparedGraph, RunMode, StageNode, VariantDescriptor
from index_topk_perflab.experimental.fused_r10a import plugin as r10a_plugin
from index_topk_perflab.experimental.fused_r10a.plugin import (
    DeepGemmFusedCandidateR10a,
)
from index_topk_perflab.provenance import path_fingerprint

from .candidate_reducer import load_segmented_candidate_reducer

_ROOT = Path(__file__).resolve().parent


class DeepGemmFusedCandidateR11d(DeepGemmFusedCandidateR10a):
    """R10a with an exact 9+7-bit first-16-bit radix split."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r11d",
        display_name="DeepGEMM fused candidate TopK R11d",
        api_version="1.0",
        implementation_version="r10a-plus-prefix-radix-9-7-r11d-v1",
        mode="fused",
        description=(
            "Exact R10a pipeline whose reducer resolves the first 16 score "
            "bits with 9-bit and 7-bit radix passes"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="r10a:local;prefix-radix-9-7:r11d-v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "producer-fusion",
            "random-token-sampling-256",
            "candidate-radix",
            "prefix-radix-9-7",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r10a-plus-prefix-radix-9-7-r11d-v1",
            "r11d_sources": path_fingerprint(_ROOT),
            "prefix_radix_bits": [9, 7, 8, 8],
            "single_control_delta": (
                "replace R10a's first 8+8 radix split with an exact 9+7 "
                "split; preserve its 16-bit prefix and all exact guards"
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
        frozen_loader = r10a_plugin.load_segmented_candidate_reducer
        r10a_plugin.load_segmented_candidate_reducer = load_segmented_candidate_reducer
        try:
            graph = super().prepare(case, inputs, options=options, mode=mode)
        finally:
            r10a_plugin.load_segmented_candidate_reducer = frozen_loader

        nodes: list[StageNode] = []
        for node in graph.nodes:
            if node.spec.stage_id in {"candidate_reducer", "repair_reducer"}:
                spec = replace(
                    node.spec,
                    description=(node.spec.description + "; R11d exact 9+7 prefix radix"),
                    kernel_regexes=("itk_fused_r11d_segmented_candidate_radix",),
                )
                nodes.append(StageNode(spec, node.run))
            else:
                nodes.append(node)
        graph.nodes = tuple(nodes)
        graph.metadata = {
            **graph.metadata,
            "parent_candidate": "fused_r10a-working6656-threads512-v1",
            "candidate_reducer_revision": "r11d-prefix-radix-9-7-v1",
            "prefix_radix_bits": [9, 7, 8, 8],
            "promotion_status": "exact-experimental-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR11d:
    return DeepGemmFusedCandidateR11d(options)
