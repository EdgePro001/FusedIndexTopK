"""R16a: extend R13a exact repair beyond 16K with fixed-size chunks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from index_topk_perflab.api import (
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    StageNode,
    StageSpec,
    VariantDescriptor,
)
from index_topk_perflab.experimental.fused_r5i import (
    common_random_sample_ids,
    guarded_sample_rank,
    load_sampling_extension,
)
from index_topk_perflab.experimental.fused_r10a_nsweep.producer import (
    load_candidate_producer as load_chunk_repair_producer,
)
from index_topk_perflab.experimental.fused_r11d import (
    load_segmented_candidate_reducer,
)
from index_topk_perflab.experimental.fused_r13a.plugin import (
    DeepGemmFusedCandidateR13a,
)
from index_topk_perflab.experimental.fused_r13a.producer import (
    load_candidate_producer as load_r13a_producer,
)
from index_topk_perflab.provenance import path_fingerprint
from index_topk_perflab.variants.common import supports_frozen_deepgemm_case

from .long_repair import load_long_context_repair

_ROOT = Path(__file__).resolve().parent
_SAMPLING_SEED_XOR = 0x4655534544523549
_TOP_K = 2048
_TARGET_CANDIDATES = 2816
_FAST_CANDIDATE_CAPACITY = 14080
_REPAIR_CHUNK_ELEMENTS = 16384
_SEGMENTS = 16
_INT32_MAX = (1 << 31) - 1


def long_context_sample_elements(context_tokens: int) -> int:
    """Return the measured, conservative long-context sampling schedule."""

    if context_tokens <= _REPAIR_CHUNK_ELEMENTS:
        raise ValueError("R16a long-context sampling requires N > 16384")
    if context_tokens <= 32768:
        samples = 512
    elif context_tokens <= 65536:
        samples = 1024
    else:
        # 1536 samples at 128K, rounded to a DeepGEMM-friendly 256 rows.
        proportional = math.ceil(context_tokens * 3 / 256)
        samples = max(1024, math.ceil(proportional / 256) * 256)
    return min(context_tokens, samples)


class DeepGemmFusedCandidateR16a(DeepGemmFusedCandidateR13a):
    """R13a fast path plus device-masked, hierarchical 16K exact repair."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r16a",
        display_name="DeepGEMM fused candidate TopK R16a long exact",
        api_version="1.0",
        implementation_version="r13a-plus-hierarchical-16k-repair-r16a-v1",
        mode="fused",
        description=(
            "Unchanged R13a fast producer and R11d reducer with a fixed-memory "
            "device-masked 16K chunk repair and online exact TopK merge"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="r13a:c97a710;hierarchical-long-repair:r16a-v1",
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "r13a-fast-path-byte-identical",
            "long-context",
            "device-exactness-guard",
            "masked-repair",
            "repair-chunk-16k",
            "online-topk-pair-merge",
            "no-host-sync",
            "no-dense-logits",
        ),
    )

    def supports(self, case: PrefillCase) -> bool:
        if case.context_tokens == _REPAIR_CHUNK_ELEMENTS:
            return super().supports(case)
        first_row_valid = case.context_tokens - case.query_tokens + 1
        return (
            supports_frozen_deepgemm_case(case)
            and _REPAIR_CHUNK_ELEMENTS < case.context_tokens <= _INT32_MAX
            and case.top_k == _TOP_K
            and case.query_tokens % 2 == 0
            and first_row_valid >= case.top_k
        )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **super().fingerprint_metadata(),
            "algorithm": "r13a-plus-hierarchical-16k-repair-r16a-v1",
            "r16a_sources": path_fingerprint(_ROOT),
            "fast_path_delta_from_r13a": "none",
            "short_context_dispatch": "unchanged-r13a-at-n16384",
            "long_repair_chunk_elements": _REPAIR_CHUNK_ELEMENTS,
            "long_repair_merge": "online-exact-top2048-pair-accumulator",
            "long_repair_workspace_complexity": "O(Q*K)+one-16K-candidate-tile",
            "long_repair_host_flag_read": False,
        }

    def prepare(
        self,
        case: PrefillCase,
        inputs: PrefillInputs,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        if not self.supports(case):
            raise ValueError(f"unsupported fused R16a case: {case}")
        if case.context_tokens == _REPAIR_CHUNK_ELEMENTS:
            # Preserve the released R13a graph exactly at its qualified shape.
            graph = super().prepare(case, inputs, options=options, mode=mode)
            graph.descriptor = self.descriptor
            graph.metadata = {
                **graph.metadata,
                "r16a_dispatch": "unchanged-r13a-16k",
                "fast_path_delta_from_r13a": "none",
                "long_context_repair_launched": False,
            }
            return graph
        if options and dict(options) != self.options:
            raise ValueError("variant options changed after R16a construction")

        import deep_gemm
        import torch

        verbose = bool(self.options.get("verbose_build", False))
        sampling = load_sampling_extension(verbose=verbose)
        fast_producer = load_r13a_producer(deep_gemm, verbose=verbose)
        fast_reducer = load_segmented_candidate_reducer(verbose=verbose)
        chunk_producer = load_chunk_repair_producer(deep_gemm, verbose=verbose)
        long_repair = load_long_context_repair(verbose=verbose)

        rows = case.query_tokens
        sample_elements = long_context_sample_elements(case.context_tokens)
        sample_rank = guarded_sample_rank(
            sample_elements,
            case.context_tokens,
            target_candidates=_TARGET_CANDIDATES,
            guard_sigmas=2.0,
        )
        sampling_seed = (int(case.seed) ^ _SAMPLING_SEED_XOR) & ((1 << 63) - 1)
        sample_ids = common_random_sample_ids(
            torch,
            case.context_tokens,
            sample_elements,
            seed=sampling_seed,
            device=inputs.q.device,
        )
        sampled_kv = torch.empty(
            (sample_elements, case.head_dim),
            device=inputs.q.device,
            dtype=inputs.kv.dtype,
        )
        sampled_scales = torch.empty(sample_elements, device=inputs.q.device, dtype=torch.float32)
        sample_start = torch.zeros_like(inputs.k_start)
        sample_end = torch.full_like(inputs.k_end, sample_elements)
        thresholds = torch.empty(rows, device=inputs.q.device, dtype=torch.float32)

        candidate_pairs = torch.empty(
            (rows, _FAST_CANDIDATE_CAPACITY),
            device=inputs.q.device,
            dtype=torch.int64,
        )
        segment_counts = torch.empty((rows, _SEGMENTS), device=inputs.q.device, dtype=torch.int32)
        output_ids = torch.empty((rows, _TOP_K), device=inputs.q.device, dtype=torch.int32)
        failure_flags = torch.empty(rows, device=inputs.q.device, dtype=torch.uint8)

        repair_candidate_pairs = torch.empty(
            (rows, _REPAIR_CHUNK_ELEMENTS),
            device=inputs.q.device,
            dtype=torch.int64,
        )
        repair_segment_counts = torch.empty(
            (rows, _SEGMENTS), device=inputs.q.device, dtype=torch.int32
        )
        local_pairs = torch.empty((rows, _TOP_K), device=inputs.q.device, dtype=torch.int64)
        local_counts = torch.empty(rows, device=inputs.q.device, dtype=torch.int32)
        accumulator_pairs = torch.empty((rows, _TOP_K), device=inputs.q.device, dtype=torch.int64)
        accumulator_counts = torch.empty(rows, device=inputs.q.device, dtype=torch.int32)
        repair_error_flags = torch.empty(rows, device=inputs.q.device, dtype=torch.uint8)

        chunks: list[tuple[int, int, Any, Any]] = []
        for logical_offset in range(0, case.context_tokens, _REPAIR_CHUNK_ELEMENTS):
            chunk_elements = min(
                _REPAIR_CHUNK_ELEMENTS,
                case.context_tokens - logical_offset,
            )
            chunk_start = torch.clamp(
                inputs.k_start - logical_offset,
                min=0,
                max=chunk_elements,
            ).contiguous()
            chunk_end = torch.clamp(
                inputs.k_end - logical_offset,
                min=0,
                max=chunk_elements,
            ).contiguous()
            chunks.append(
                (
                    logical_offset,
                    chunk_elements,
                    chunk_start,
                    chunk_end,
                )
            )

        def run_sample_gather(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            sampling.gather_out(
                standard.kv,
                standard.kv_scales,
                sample_ids,
                sampled_kv,
                sampled_scales,
            )
            artifacts["sampled_kv"] = sampled_kv
            artifacts["sampled_scales"] = sampled_scales

        def run_sample_indexer(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            artifacts["sampled_scores"] = deep_gemm.fp8_mqa_logits(
                standard.q,
                (sampled_kv, sampled_scales),
                standard.weights,
                sample_start,
                sample_end,
                clean_logits=False,
            )

        def run_sample_threshold(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            sampling.threshold_out(
                artifacts["sampled_scores"],
                sample_ids,
                standard.k_start,
                standard.k_end,
                thresholds,
                sample_rank,
                _TARGET_CANDIDATES,
            )
            artifacts["thresholds"] = thresholds

        def run_candidate_producer(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            fast_producer.fp8_mqa_candidate_r13a_out(
                standard.q,
                standard.kv,
                standard.kv_scales,
                standard.weights,
                standard.k_start,
                standard.k_end,
                thresholds,
                candidate_pairs,
                segment_counts,
            )
            artifacts["candidate_pairs"] = candidate_pairs
            artifacts["segment_counts"] = segment_counts

        def run_candidate_reducer(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            fast_reducer.topk_out(
                candidate_pairs,
                segment_counts,
                standard.k_start,
                standard.k_end,
                output_ids,
                failure_flags,
            )
            artifacts["fast_failure_flags"] = failure_flags
            artifacts["fast_indices"] = output_ids.unsqueeze(1)

        def run_hierarchical_repair(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            for chunk_index, chunk in enumerate(chunks):
                logical_offset, chunk_elements, chunk_start, chunk_end = chunk
                chunk_kv = standard.kv.narrow(0, logical_offset, chunk_elements)
                chunk_scales = standard.kv_scales.narrow(0, logical_offset, chunk_elements)
                if not chunk_kv.is_contiguous() or not chunk_scales.is_contiguous():
                    raise RuntimeError("R16a repair chunk views must remain contiguous")
                chunk_producer.fp8_mqa_candidate_repair_r5i_out(
                    standard.q,
                    chunk_kv,
                    chunk_scales,
                    standard.weights,
                    chunk_start,
                    chunk_end,
                    failure_flags,
                    repair_candidate_pairs,
                    repair_segment_counts,
                )
                long_repair.local_topk_pairs_out(
                    repair_candidate_pairs,
                    repair_segment_counts,
                    chunk_start,
                    chunk_end,
                    failure_flags,
                    local_pairs,
                    local_counts,
                    repair_error_flags,
                    logical_offset,
                    chunk_index == 0,
                )
                long_repair.merge_topk_pairs_out(
                    accumulator_pairs,
                    accumulator_counts,
                    local_pairs,
                    local_counts,
                    failure_flags,
                    repair_error_flags,
                    chunk_index == 0,
                )
            long_repair.finalize_repair_out(
                accumulator_pairs,
                accumulator_counts,
                output_ids,
                failure_flags,
                repair_error_flags,
            )
            artifacts["failure_flags"] = failure_flags
            artifacts["repair_error_flags"] = repair_error_flags
            artifacts["indices"] = output_ids.unsqueeze(1)

        nodes = (
            StageNode(
                StageSpec(
                    stage_id="sample_gather",
                    dependencies=(),
                    consumes=("inputs",),
                    produces=("sampled_kv", "sampled_scales"),
                    semantic_ops=("indexer",),
                    description="Gather common random-token KV sample",
                    kernel_regexes=("gather_sampled_kv",),
                ),
                run_sample_gather,
            ),
            StageNode(
                StageSpec(
                    stage_id="sample_indexer",
                    dependencies=("sample_gather",),
                    consumes=("inputs", "sampled_kv", "sampled_scales"),
                    produces=("sampled_scores",),
                    semantic_ops=("indexer",),
                    description="Unmodified DeepGEMM scorer over sampled KV",
                    kernel_regexes=("sm90_fp8_mqa_logits",),
                ),
                run_sample_indexer,
            ),
            StageNode(
                StageSpec(
                    stage_id="sample_threshold",
                    dependencies=("sample_indexer",),
                    consumes=("inputs", "sampled_scores"),
                    produces=("thresholds",),
                    semantic_ops=("topk",),
                    description="Long-context guarded sample radix threshold",
                    kernel_regexes=("sampled_threshold_radix",),
                ),
                run_sample_threshold,
            ),
            StageNode(
                StageSpec(
                    stage_id="candidate_producer",
                    dependencies=("sample_threshold",),
                    consumes=("inputs", "thresholds"),
                    produces=("candidate_pairs", "segment_counts"),
                    semantic_ops=("indexer", "topk"),
                    description="Unchanged R13a fused candidate producer",
                    kernel_regexes=("itk_sm90_fp8_mqa_candidate_producer_r13a",),
                ),
                run_candidate_producer,
            ),
            StageNode(
                StageSpec(
                    stage_id="candidate_reducer",
                    dependencies=("candidate_producer",),
                    consumes=("inputs", "candidate_pairs", "segment_counts"),
                    produces=("fast_failure_flags", "fast_indices"),
                    semantic_ops=("topk", "output"),
                    description="Unchanged R11d fast reducer and exactness guard",
                    kernel_regexes=("itk_fused_r11d_segmented_candidate_radix",),
                ),
                run_candidate_reducer,
            ),
            StageNode(
                StageSpec(
                    stage_id="hierarchical_repair",
                    dependencies=("candidate_reducer",),
                    consumes=("inputs", "fast_failure_flags", "fast_indices"),
                    produces=("failure_flags", "repair_error_flags", "indices"),
                    semantic_ops=("indexer", "topk", "output"),
                    description=(
                        "Device-masked exact 16K chunk rescans with online Top-2048 pair merge"
                    ),
                    kernel_regexes=(
                        "itk_sm90_fp8_mqa_masked_repair_producer_r5i",
                        "itk_r16a_chunk_local_topk",
                        "itk_r16a_merge_topk_pairs",
                        "itk_r16a_finalize_repair",
                    ),
                ),
                run_hierarchical_repair,
            ),
        )
        return PreparedGraph(
            descriptor=self.descriptor,
            nodes=nodes,
            initial_artifacts={"inputs": inputs},
            terminal_artifact="indices",
            metadata={
                "algorithm": "r13a-plus-hierarchical-16k-repair-r16a-v1",
                "sample_elements": sample_elements,
                "sample_rank_descending": sample_rank,
                "sample_target_candidates": _TARGET_CANDIDATES,
                "sample_guard_sigmas": 2.0,
                "sampling_granularity": "individual-random-token",
                "dense_logits_materialized": False,
                "candidate_capacity": _FAST_CANDIDATE_CAPACITY,
                "candidate_segments": _SEGMENTS,
                "device_failure_flags": True,
                "timed_failure_repair": True,
                "repair_dispatch": "device-mask-no-host-sync",
                "repair_chunk_elements": _REPAIR_CHUNK_ELEMENTS,
                "repair_chunks": len(chunks),
                "repair_merge": "online-exact-top2048-pairs",
                "repair_workspace_bounded_in_n": True,
                "fast_path_delta_from_r13a": "none",
                "fast_reducer_delta_from_r11d": "none",
                "upstream_deepgemm_modified": False,
                "promotion_status": "exact-experimental-long-context",
                "sampling_resources": dict(sampling.resource_report()),
                "fast_reducer_resources": dict(fast_reducer.resource_report()),
                "long_repair_resources": dict(long_repair.resource_report()),
            },
        )


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR16a:
    return DeepGemmFusedCandidateR16a(options)
