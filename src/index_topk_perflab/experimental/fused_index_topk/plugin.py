"""Public FusedIndexTopK operator for short and long contexts."""

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
from index_topk_perflab.provenance import path_fingerprint
from index_topk_perflab.variants.common import supports_frozen_deepgemm_case

from .candidate_reducer import load_segmented_candidate_reducer
from .long_repair import load_long_context_repair
from .producer import load_candidate_producer
from .repair_producer import load_candidate_producer as load_repair_producer
from .sampling import (
    common_random_sample_ids,
    guarded_sample_rank,
    load_sampling_extension,
)

_ROOT = Path(__file__).resolve().parent
_SAMPLING_SEED_XOR = 0x4655534544523549
_TOP_K = 2048
_TARGET_CANDIDATES = 2816
_FAST_CANDIDATE_CAPACITY = 14080
_REPAIR_CHUNK_ELEMENTS = 16384
_SEGMENTS = 16
_INT32_MAX = (1 << 31) - 1
_MIN_CONTEXT = 6144
_SHORT_SAMPLE_ELEMENTS = 256


def long_context_sample_elements(context_tokens: int) -> int:
    """Return the measured, conservative long-context sampling schedule."""

    if context_tokens <= _REPAIR_CHUNK_ELEMENTS:
        raise ValueError("long-context sampling requires N > 16384")
    if context_tokens <= 32768:
        samples = 512
    elif context_tokens <= 65536:
        samples = 1024
    else:
        # 1536 samples at 128K, rounded to a DeepGEMM-friendly 256 rows.
        proportional = math.ceil(context_tokens * 3 / 256)
        samples = max(1024, math.ceil(proportional / 256) * 256)
    return min(context_tokens, samples)


class FusedIndexTopK:
    """DeepGEMM-style fused Indexer and exact Top-K implementation."""

    descriptor = VariantDescriptor(
        plugin_id="fused_index_topk",
        display_name="FusedIndexTopK",
        api_version="1.0",
        implementation_version="1.0.0",
        mode="fused",
        description=(
            "DeepGEMM-style score/candidate fusion, exact radix reduction, "
            "and fixed-memory device-side repair"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="fused-index-topk-v1",
        tags=(
            "prefill",
            "sm90",
            "deepseek-v3.2",
            "deepgemm-style",
            "exact-topk",
            "long-context",
            "device-exactness-guard",
            "masked-repair",
            "repair-chunk-16k",
            "online-topk-pair-merge",
            "no-host-sync",
            "no-dense-logits",
        ),
    )

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options = dict(options or {})
        unknown = set(self.options) - {"verbose_build"}
        if unknown:
            raise ValueError(f"unknown FusedIndexTopK options: {sorted(unknown)}")

    def supports(self, case: PrefillCase) -> bool:
        first_row_valid = case.context_tokens - case.query_tokens + 1
        common = (
            supports_frozen_deepgemm_case(case)
            and _MIN_CONTEXT <= case.context_tokens <= _INT32_MAX
            and case.top_k == _TOP_K
            and case.query_tokens % 2 == 0
            and first_row_valid >= case.top_k
        )
        if not common:
            return False
        return case.context_tokens > _REPAIR_CHUNK_ELEMENTS or case.context_tokens % 256 == 0

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            "algorithm": "sampled-fused-indexer-exact-radix-with-device-repair-v1",
            "operator_sources": path_fingerprint(_ROOT),
            "upstream_deepgemm_modified": False,
            "short_context_range": [_MIN_CONTEXT, _REPAIR_CHUNK_ELEMENTS],
            "long_repair_chunk_elements": _REPAIR_CHUNK_ELEMENTS,
            "long_repair_merge": "online-exact-top2048-pair-accumulator",
            "long_repair_workspace_complexity": "O(Q*K)+one-16K-candidate-tile",
            "long_repair_host_flag_read": False,
        }

    def _prepare_short(
        self,
        case: PrefillCase,
        inputs: PrefillInputs,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        """Build the measured 6K--16K fast path and complete-row repair."""

        del mode
        if options and dict(options) != self.options:
            raise ValueError("variant options changed after FusedIndexTopK construction")

        import deep_gemm
        import torch

        verbose = bool(self.options.get("verbose_build", False))
        sampling = load_sampling_extension(verbose=verbose)
        fast_producer = load_candidate_producer(deep_gemm, verbose=verbose)
        repair_producer = load_repair_producer(deep_gemm, verbose=verbose)
        reducer = load_segmented_candidate_reducer(verbose=verbose)

        rows = case.query_tokens
        sample_elements = _SHORT_SAMPLE_ELEMENTS
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
        sampled_scales = torch.empty(
            sample_elements,
            device=inputs.q.device,
            dtype=torch.float32,
        )
        sample_start = torch.zeros_like(inputs.k_start)
        sample_end = torch.full_like(inputs.k_end, sample_elements)
        thresholds = torch.empty(rows, device=inputs.q.device, dtype=torch.float32)
        candidate_pairs = torch.empty(
            (rows, _FAST_CANDIDATE_CAPACITY),
            device=inputs.q.device,
            dtype=torch.int64,
        )
        segment_counts = torch.empty(
            (rows, _SEGMENTS),
            device=inputs.q.device,
            dtype=torch.int32,
        )
        output_ids = torch.empty((rows, _TOP_K), device=inputs.q.device, dtype=torch.int32)
        failure_flags = torch.empty(rows, device=inputs.q.device, dtype=torch.uint8)
        repair_candidate_pairs = torch.empty(
            (rows, _REPAIR_CHUNK_ELEMENTS),
            device=inputs.q.device,
            dtype=torch.int64,
        )
        repair_segment_counts = torch.empty(
            (rows, _SEGMENTS),
            device=inputs.q.device,
            dtype=torch.int32,
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
            fast_producer.produce_candidates_out(
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
            reducer.topk_out(
                candidate_pairs,
                segment_counts,
                standard.k_start,
                standard.k_end,
                output_ids,
                failure_flags,
            )
            artifacts["fast_failure_flags"] = failure_flags
            artifacts["fast_indices"] = output_ids.unsqueeze(1)

        def run_repair_producer(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            repair_producer.produce_repair_candidates_out(
                standard.q,
                standard.kv,
                standard.kv_scales,
                standard.weights,
                standard.k_start,
                standard.k_end,
                failure_flags,
                repair_candidate_pairs,
                repair_segment_counts,
            )
            artifacts["repair_candidate_pairs"] = repair_candidate_pairs
            artifacts["repair_segment_counts"] = repair_segment_counts

        def run_repair_reducer(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            reducer.repair_topk_out(
                repair_candidate_pairs,
                repair_segment_counts,
                standard.k_start,
                standard.k_end,
                output_ids,
                failure_flags,
            )
            artifacts["failure_flags"] = failure_flags
            artifacts["indices"] = output_ids.unsqueeze(1)

        nodes = (
            StageNode(
                StageSpec(
                    stage_id="sample_gather",
                    dependencies=(),
                    consumes=("inputs",),
                    produces=("sampled_kv", "sampled_scales"),
                    semantic_ops=("indexer",),
                    description="Gather one common random-token KV sample",
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
                    description="Causal-aware sampled radix threshold",
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
                    description="DeepGEMM-style score and candidate fusion",
                    kernel_regexes=("candidate_producer",),
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
                    description="Exact 9+7+8+8 radix reducer and device guard",
                    kernel_regexes=("segmented_candidate_radix",),
                ),
                run_candidate_reducer,
            ),
            StageNode(
                StageSpec(
                    stage_id="repair_producer",
                    dependencies=("candidate_reducer",),
                    consumes=("inputs", "fast_failure_flags"),
                    produces=("repair_candidate_pairs", "repair_segment_counts"),
                    semantic_ops=("indexer", "topk"),
                    description="Device-masked complete-row score rescan",
                    kernel_regexes=("masked_repair_producer",),
                ),
                run_repair_producer,
            ),
            StageNode(
                StageSpec(
                    stage_id="repair_reducer",
                    dependencies=("repair_producer",),
                    consumes=(
                        "inputs",
                        "fast_failure_flags",
                        "fast_indices",
                        "repair_candidate_pairs",
                        "repair_segment_counts",
                    ),
                    produces=("failure_flags", "indices"),
                    semantic_ops=("topk", "output"),
                    description="Device-masked exact repair; safe rows keep fast output",
                    kernel_regexes=("segmented_candidate_radix",),
                ),
                run_repair_reducer,
            ),
        )
        return PreparedGraph(
            descriptor=self.descriptor,
            nodes=nodes,
            initial_artifacts={"inputs": inputs},
            terminal_artifact="indices",
            metadata={
                "algorithm": "sampled-fused-indexer-exact-radix-with-device-repair-v1",
                "sample_elements": sample_elements,
                "sample_rank_descending": sample_rank,
                "sample_target_candidates": _TARGET_CANDIDATES,
                "sample_guard_sigmas": 2.0,
                "sampling_granularity": "individual-random-token",
                "sampling_without_replacement": True,
                "dense_logits_materialized": False,
                "candidate_capacity": _FAST_CANDIDATE_CAPACITY,
                "candidate_segments": _SEGMENTS,
                "fast_reducer_working_capacity": 6656,
                "prefix_radix_bits": [9, 7, 8, 8],
                "producer_block_kv": 128,
                "producer_math_threads": 256,
                "producer_grid_sms_multiplier": 2,
                "device_failure_flags": True,
                "timed_failure_repair": True,
                "repair_dispatch": "device-mask-no-host-sync",
                "repair_candidate_capacity": _REPAIR_CHUNK_ELEMENTS,
                "upstream_deepgemm_modified": False,
                "promotion_status": "public-final",
                "sampling_resources": dict(sampling.resource_report()),
                "reducer_resources": dict(reducer.resource_report()),
            },
        )

    def prepare(
        self,
        case: PrefillCase,
        inputs: PrefillInputs,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        if not self.supports(case):
            raise ValueError(f"unsupported FusedIndexTopK case: {case}")
        if case.context_tokens <= _REPAIR_CHUNK_ELEMENTS:
            return self._prepare_short(case, inputs, options=options, mode=mode)
        if options and dict(options) != self.options:
            raise ValueError("variant options changed after FusedIndexTopK construction")

        import deep_gemm
        import torch

        verbose = bool(self.options.get("verbose_build", False))
        sampling = load_sampling_extension(verbose=verbose)
        fast_producer = load_candidate_producer(deep_gemm, verbose=verbose)
        fast_reducer = load_segmented_candidate_reducer(verbose=verbose)
        chunk_producer = load_repair_producer(deep_gemm, verbose=verbose)
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
            fast_producer.produce_candidates_out(
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
                    raise RuntimeError("repair chunk views must remain contiguous")
                chunk_producer.produce_repair_candidates_out(
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
                    description="DeepGEMM-style fused candidate producer",
                    kernel_regexes=("candidate_producer",),
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
                    description="Exact 9+7+8+8 radix reducer and device guard",
                    kernel_regexes=("segmented_candidate_radix",),
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
                        "masked_repair_producer",
                        "chunk_local_topk",
                        "merge_topk_pairs",
                        "finalize_repair",
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
                "algorithm": "sampled-fused-indexer-exact-radix-with-device-repair-v1",
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
                "upstream_deepgemm_modified": False,
                "promotion_status": "public-final-long-context",
                "sampling_resources": dict(sampling.resource_report()),
                "fast_reducer_resources": dict(fast_reducer.resource_report()),
                "long_repair_resources": dict(long_repair.resource_report()),
            },
        )


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> FusedIndexTopK:
    return FusedIndexTopK(options)
