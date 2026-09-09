"""R8a: match the threshold block size to the 128 sampled scores."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PreparedGraph, RunMode, StageNode, VariantDescriptor
from index_topk_perflab.experimental.fused_r5i import plugin as r5i_plugin
from index_topk_perflab.experimental.fused_r6f.plugin import DeepGemmFusedCandidateR6f
from index_topk_perflab.provenance import path_fingerprint

from .sampling import load_sampling_extension

_ROOT = Path(__file__).resolve().parent
_R6F_ROOT = _ROOT.parent / "fused_r6f"


class DeepGemmFusedCandidateR8a(DeepGemmFusedCandidateR6f):
    """R6f with four threshold warps instead of eight."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r8a",
        display_name="DeepGEMM fused candidate TopK R8a",
        api_version="1.0",
        implementation_version="r6f-plus-threshold128threads-r8a-v1",
        mode="fused",
        description=(
            "Exact R6f pipeline whose 128-score sampled-threshold radix "
            "kernel launches 128 threads instead of 256"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="r6f:6f5a420;threshold128threads:r8a-v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "producer-fusion",
            "no-dense-logits",
            "candidate-radix",
            "sample-threshold-128-threads",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r6f-plus-threshold128threads-r8a-v1",
            "r8a_sources": path_fingerprint(_ROOT),
            "frozen_r6f_shared_sources": path_fingerprint(_R6F_ROOT),
            "threshold_threads": 128,
            "threshold_sample_count": 128,
            "single_control_delta": (
                "launch four rather than eight warps for the 128 sampled "
                "scores; each thread clears two histogram bins"
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
        frozen_loader = r5i_plugin.load_sampling_extension
        r5i_plugin.load_sampling_extension = load_sampling_extension
        try:
            graph = super().prepare(case, inputs, options=options, mode=mode)
        finally:
            r5i_plugin.load_sampling_extension = frozen_loader

        nodes: list[StageNode] = []
        for node in graph.nodes:
            if node.spec.stage_id == "sample_threshold":
                spec = replace(
                    node.spec,
                    description=node.spec.description + "; R8a 128-thread block",
                    kernel_regexes=("sampled_threshold_radix",),
                )
                nodes.append(StageNode(spec, node.run))
            else:
                nodes.append(node)
        graph.nodes = tuple(nodes)
        graph.metadata = {
            **graph.metadata,
            "parent_candidate": "fused_r6f-sparse-direct-atomic-v1",
            "sample_threshold_revision": "r8a-128-thread-block-v1",
            "sample_threshold_threads": 128,
            "sample_threshold_histogram_bins_per_thread": 2,
            "promotion_status": "exact-experimental-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR8a:
    return DeepGemmFusedCandidateR8a(options)
