"""R9b: use the lower-variance 256-sample estimate to target 2816."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PreparedGraph, RunMode, StageNode, VariantDescriptor
from index_topk_perflab.experimental.fused_r5i import plugin as r5i_plugin
from index_topk_perflab.experimental.fused_r9a.plugin import DeepGemmFusedCandidateR9a
from index_topk_perflab.provenance import path_fingerprint

_ROOT = Path(__file__).resolve().parent
_TARGET_CANDIDATES = 2816


class DeepGemmFusedCandidateR9b(DeepGemmFusedCandidateR9a):
    """R9a with a guarded 2816-candidate target and unchanged exact repair."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r9b",
        display_name="DeepGEMM fused candidate TopK R9b",
        api_version="1.0",
        implementation_version="r9a-plus-target2816-r9b-v1",
        mode="fused",
        description=(
            "Exact 256-sample R9a pipeline with its guarded fast-path target "
            "lowered from 3072 to 2816"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="r9a:local;target2816:r9b-v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "producer-fusion",
            "no-dense-logits",
            "random-token-sampling-256",
            "target-candidates-2816",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r9a-plus-target2816-r9b-v1",
            "r9b_sources": path_fingerprint(_ROOT),
            "target_candidates": _TARGET_CANDIDATES,
            "single_control_delta": (
                "lower R9a's sampled fast-path candidate target from 3072 "
                "to 2816; preserve all exact guard and repair code"
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
        frozen_target = r5i_plugin._TARGET_CANDIDATES
        r5i_plugin._TARGET_CANDIDATES = _TARGET_CANDIDATES
        try:
            graph = super().prepare(case, inputs, options=options, mode=mode)
        finally:
            r5i_plugin._TARGET_CANDIDATES = frozen_target

        nodes: list[StageNode] = []
        for node in graph.nodes:
            if node.spec.stage_id == "sample_threshold":
                original_run = node.run

                def run_target_threshold(
                    context: Any,
                    artifacts: dict[str, Any],
                    run: Any = original_run,
                ) -> None:
                    previous = r5i_plugin._TARGET_CANDIDATES
                    r5i_plugin._TARGET_CANDIDATES = _TARGET_CANDIDATES
                    try:
                        run(context, artifacts)
                    finally:
                        r5i_plugin._TARGET_CANDIDATES = previous

                spec = replace(
                    node.spec,
                    description=node.spec.description + "; R9b target 2816",
                )
                nodes.append(StageNode(spec, run_target_threshold))
            else:
                nodes.append(node)
        graph.nodes = tuple(nodes)
        graph.metadata = {
            **graph.metadata,
            "parent_candidate": "fused_r9a-sample256-target3072-v1",
            "sample_target_candidates": _TARGET_CANDIDATES,
            "candidate_target_revision": "r9b-target2816-v1",
            "promotion_status": "exact-experimental-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR9b:
    return DeepGemmFusedCandidateR9b(options)
