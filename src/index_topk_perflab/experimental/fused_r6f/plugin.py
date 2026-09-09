"""R6f: use sparse direct atomics for the fused third histogram."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PreparedGraph, RunMode, StageNode, VariantDescriptor
from index_topk_perflab.experimental.fused_r5i import plugin as r5i_plugin
from index_topk_perflab.experimental.fused_r5i.plugin import DeepGemmFusedCandidateR5i
from index_topk_perflab.provenance import path_fingerprint

from .candidate_reducer import load_segmented_candidate_reducer

_ROOT = Path(__file__).resolve().parent
_R5I_ROOT = _ROOT.parent / "fused_r5i"


class DeepGemmFusedCandidateR6f(DeepGemmFusedCandidateR5i):
    """R6e pipeline without full-warp aggregation for sparse updates."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r6f",
        display_name="DeepGEMM fused candidate TopK R6f",
        api_version="1.0",
        implementation_version="r6e-plus-sparse-direct-atomic-r6f-v1",
        mode="fused",
        description=(
            "R6e exact reducer using direct shared atomics only for selected "
            "values instead of full-warp histogram aggregation"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="r5i:c478d53;r6e:local;sparse-direct-atomic:r6f-v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "producer-fusion",
            "no-dense-logits",
            "candidate-radix",
            "prefix16-partition",
            "partition-third-histogram-fusion",
            "sparse-direct-shared-atomic",
            "eight-warp-tail-selection",
            "named-subgroup-barrier",
            "tail-capacity-256",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            "algorithm": "r6e-plus-sparse-direct-atomic-r6f-v1",
            "r6f_sources": path_fingerprint(_ROOT),
            "frozen_r5i_shared_sources": path_fingerprint(_R5I_ROOT),
            "upstream_deepgemm_modified": False,
            "timed_repair": True,
            "repair_dispatch": "device-mask-no-host-sync",
            "single_control_delta": (
                "replace R6e's full-warp histogram aggregation with one "
                "direct shared atomic per selected-prefix value"
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
        frozen_loader = r5i_plugin.load_segmented_candidate_reducer
        r5i_plugin.load_segmented_candidate_reducer = load_segmented_candidate_reducer
        try:
            graph = super().prepare(case, inputs, options=options, mode=mode)
        finally:
            r5i_plugin.load_segmented_candidate_reducer = frozen_loader

        nodes: list[StageNode] = []
        for node in graph.nodes:
            if node.spec.stage_id in {"candidate_reducer", "repair_reducer"}:
                spec = replace(
                    node.spec,
                    description=(
                        node.spec.description
                        + "; R6f sparse direct-atomic third histogram"
                    ),
                    kernel_regexes=(
                        "itk_fused_r6f_segmented_candidate_radix",
                    ),
                )
                nodes.append(StageNode(spec, node.run))
            else:
                nodes.append(node)
        graph.nodes = tuple(nodes)
        graph.metadata = {
            **graph.metadata,
            "parent_candidate": "fused_r6e-fused-third-histogram-v1",
            "candidate_reducer_revision": "r6f-sparse-direct-atomic-v1",
            "warp_tail_capacity": 256,
            "warp_tail_threads": 256,
            "tail_radix_bytes": 1,
            "partition_fused_radix_bytes": 1,
            "partition_histogram_update": "direct-atomic-selected-only",
            "warp_tail_overflow_repair": True,
            "promotion_status": "exact-experimental-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR6f:
    return DeepGemmFusedCandidateR6f(options)
