"""R13a: halve the producer KV tile and target two resident CTAs per SM."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PreparedGraph, RunMode, StageNode, VariantDescriptor
from index_topk_perflab.experimental.fused_r11d.plugin import (
    DeepGemmFusedCandidateR11d,
)
from index_topk_perflab.provenance import path_fingerprint

from .producer import load_candidate_producer

_ROOT = Path(__file__).resolve().parent
_CANDIDATE_CAPACITY = 14080
_CANDIDATE_SEGMENTS = 16


class DeepGemmFusedCandidateR13a(DeepGemmFusedCandidateR11d):
    """R11d with a 128-wide, 256-math-thread candidate producer."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r13a",
        display_name="DeepGEMM fused candidate TopK R13a",
        api_version="1.0",
        implementation_version="r11d-plus-producer-blockkv128-math256-r13a-v1",
        mode="fused",
        description=(
            "Exact R11d pipeline with a 128-wide candidate producer tile, "
            "256 math threads, and a two-CTA-per-SM launch target"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="r11d:81036a4;producer-blockkv128-math256:r13a-v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "producer-block-kv-128",
            "producer-math-threads-256",
            "producer-two-cta-target",
            "logical-segments-16",
            "prefix-radix-9-7",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r11d-plus-producer-blockkv128-math256-r13a-v1",
            "r13a_sources": path_fingerprint(_ROOT),
            "producer_block_kv": 128,
            "producer_math_threads": 256,
            "producer_specialized_threads": 128,
            "producer_grid_sms_multiplier": 2,
            "producer_physical_math_warps": 8,
            "producer_logical_candidate_segments": 16,
            "producer_segments_per_math_warp": 2,
            "upstream_deepgemm_modified": False,
            "single_control_delta": (
                "replace R11d producer BLOCK_KV=256/math=512/grid=1xSM "
                "with BLOCK_KV=128/math=256/grid=2xSM; preserve 16 reducer "
                "segments by splitting each warp on KV-block parity"
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
        graph = super().prepare(case, inputs, options=options, mode=mode)

        import deep_gemm
        import torch

        producer = load_candidate_producer(
            deep_gemm,
            verbose=bool(self.options.get("verbose_build", False)),
        )
        candidate_pairs = torch.empty(
            (case.query_tokens, _CANDIDATE_CAPACITY),
            device=inputs.q.device,
            dtype=torch.int64,
        )
        segment_counts = torch.empty(
            (case.query_tokens, _CANDIDATE_SEGMENTS),
            device=inputs.q.device,
            dtype=torch.int32,
        )

        def run_candidate_producer(context: Any, artifacts: dict[str, Any]) -> None:
            standard = artifacts["inputs"]
            producer.fp8_mqa_candidate_r13a_out(
                standard.q,
                standard.kv,
                standard.kv_scales,
                standard.weights,
                standard.k_start,
                standard.k_end,
                artifacts["thresholds"],
                candidate_pairs,
                segment_counts,
            )
            artifacts["candidate_pairs"] = candidate_pairs
            artifacts["segment_counts"] = segment_counts

        nodes: list[StageNode] = []
        found_producer = False
        for node in graph.nodes:
            if node.spec.stage_id == "candidate_producer":
                found_producer = True
                nodes.append(
                    StageNode(
                        replace(
                            node.spec,
                            description=(
                                "R13a DeepGEMM candidate producer with "
                                "BLOCK_KV=128, 256 math threads, and two "
                                "logical segments per math warp"
                            ),
                            kernel_regexes=(
                                "itk_sm90_fp8_mqa_candidate_producer_r13a",
                            ),
                        ),
                        run_candidate_producer,
                    )
                )
            else:
                nodes.append(node)
        if not found_producer:
            raise RuntimeError("R13a parent graph is missing candidate_producer")

        graph.descriptor = self.descriptor
        graph.nodes = tuple(nodes)
        graph.metadata = {
            **graph.metadata,
            "parent_candidate": "fused_r11d-prefix-radix-9-7-v1",
            "candidate_producer_revision": "r13a-blockkv128-math256-v1",
            "producer_block_kv": 128,
            "producer_math_threads": 256,
            "producer_specialized_threads": 128,
            "producer_grid_sms_multiplier": 2,
            "producer_dynamic_shared_bytes": 101508,
            "producer_physical_math_warps": 8,
            "producer_logical_candidate_segments": 16,
            "producer_segment_mapping": "warp-plus-kv-block-parity",
            "repair_producer_delta_from_r11d": "none",
            "candidate_reducer_delta_from_r11d": "none",
            "promotion_status": "exact-experimental-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR13a:
    return DeepGemmFusedCandidateR13a(options)
