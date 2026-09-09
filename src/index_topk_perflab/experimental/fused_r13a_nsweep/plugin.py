"""Run the exact R13a producer with R11d's variable-N repair adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import PreparedGraph, RunMode, StageNode, VariantDescriptor
from index_topk_perflab.experimental.fused_r11d_nsweep.plugin import (
    DeepGemmFusedCandidateR11dNSweep,
)
from index_topk_perflab.experimental.fused_r13a.producer import (
    load_candidate_producer,
)
from index_topk_perflab.provenance import path_fingerprint

_ROOT = Path(__file__).resolve().parent
_CANDIDATE_CAPACITY = 14080
_CANDIDATE_SEGMENTS = 16


class DeepGemmFusedCandidateR13aNSweep(DeepGemmFusedCandidateR11dNSweep):
    """R11d variable-N graph with only its fast producer replaced by R13a."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r13a_nsweep",
        display_name="DeepGEMM fused candidate TopK R13a variable-N probe",
        api_version="1.0",
        implementation_version="r13a-producer-plus-r11d-host-nle16384-probe-v1",
        mode="fused",
        description=(
            "Exact R13a 128-wide producer with R11d's host-only variable-N "
            "repair adapter"
        ),
        implementation="project-local-r13a+host-variable-n-probe",
        exact_topk=True,
        source_revision="r13a:c97a710;r11d-nsweep:local",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "r13a-producer",
            "variable-n-probe",
            "host-repair-adapter",
            "prefix-radix-9-7",
            "device-exactness-guard",
            "masked-exact-repair",
        ),
    )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r13a-producer-plus-r11d-host-nle16384-probe-v1",
            "r13a_nsweep_sources": path_fingerprint(_ROOT),
            "producer_block_kv": 128,
            "producer_math_threads": 256,
            "producer_grid_sms_multiplier": 2,
            "fast_producer_device_delta_from_r13a": "none",
            "reducer_device_delta_from_r11d": "none",
            "repair_host_delta": "accept runtime 0<N<=16384",
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
                                "R13a BLOCK_KV=128/math=256 fast producer "
                                "running at the variable replay context"
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
            raise RuntimeError("R13a N-sweep graph is missing candidate_producer")

        graph.descriptor = self.descriptor
        graph.nodes = tuple(nodes)
        graph.metadata = {
            **graph.metadata,
            "parent_candidate": "fused_r11d_nsweep",
            "candidate_producer_revision": "r13a-blockkv128-math256-v1",
            "producer_runtime_context_tokens": case.context_tokens,
            "producer_block_kv": 128,
            "producer_math_threads": 256,
            "producer_grid_sms_multiplier": 2,
            "fast_producer_device_delta_from_r13a": "none",
            "repair_runtime_context_tokens": case.context_tokens,
            "promotion_status": "exact-experimental-n-sweep-control",
        }
        return graph


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR13aNSweep:
    return DeepGemmFusedCandidateR13aNSweep(options)
