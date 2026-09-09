"""R10a: exploit R9b's tighter candidate tail to run four reducers per SM."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PreparedGraph, RunMode, StageNode, VariantDescriptor
from index_topk_perflab.experimental.fused_r6f import plugin as r6f_plugin
from index_topk_perflab.experimental.fused_r9b.plugin import DeepGemmFusedCandidateR9b
from index_topk_perflab.provenance import path_fingerprint

from .candidate_reducer import load_segmented_candidate_reducer

_ROOT = Path(__file__).resolve().parent
_R9B_ROOT = _ROOT.parent / "fused_r9b"


class DeepGemmFusedCandidateR10a(DeepGemmFusedCandidateR9b):
    """R9b with a bounded 6656-entry fast-reducer working set."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r10a",
        display_name="DeepGEMM fused candidate TopK R10a",
        api_version="1.0",
        implementation_version="r9b-plus-working6656-threads512-r10a-v1",
        mode="fused",
        description=(
            "Exact R9b pipeline with a 6656-entry fast-reducer SMEM working "
            "set, 512 threads, and device overflow repair"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="r9b:local;working6656-threads512:r10a-v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "producer-fusion",
            "random-token-sampling-256",
            "candidate-radix",
            "fast-working-capacity-6656",
            "reducer-512-threads",
            "target-four-cta-per-sm",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r9b-plus-working6656-threads512-r10a-v1",
            "r10a_sources": path_fingerprint(_ROOT),
            "frozen_r9b_shared_sources": path_fingerprint(_R9B_ROOT),
            "fast_input_capacity": 14080,
            "fast_working_capacity": 6656,
            "fast_block_threads": 512,
            "single_control_delta": (
                "bound the fast reducer's on-chip working set at 6656 and "
                "use 512 threads; overflow sets the existing exact repair flag"
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
        frozen_loader = r6f_plugin.load_segmented_candidate_reducer
        r6f_plugin.load_segmented_candidate_reducer = load_segmented_candidate_reducer
        try:
            graph = super().prepare(case, inputs, options=options, mode=mode)
        finally:
            r6f_plugin.load_segmented_candidate_reducer = frozen_loader

        nodes: list[StageNode] = []
        for node in graph.nodes:
            if node.spec.stage_id in {"candidate_reducer", "repair_reducer"}:
                spec = replace(
                    node.spec,
                    description=(
                        node.spec.description
                        + "; R10a 6656-entry/512-thread fast working set"
                    ),
                    kernel_regexes=("itk_fused_r10a_segmented_candidate_radix",),
                )
                nodes.append(StageNode(spec, node.run))
            else:
                nodes.append(node)
        graph.nodes = tuple(nodes)
        graph.metadata = {
            **graph.metadata,
            "parent_candidate": "fused_r9b-sample256-target2816-v1",
            "candidate_reducer_revision": "r10a-working6656-threads512-v1",
            "fast_reducer_input_capacity": 14080,
            "fast_reducer_working_capacity": 6656,
            "fast_reducer_threads": 512,
            "fast_working_overflow_repair": True,
            "promotion_status": "exact-experimental-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR10a:
    return DeepGemmFusedCandidateR10a(options)
