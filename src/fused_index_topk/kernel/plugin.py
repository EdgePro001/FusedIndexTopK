"""Public FusedIndexTopK operator with same-kernel exact Top-K."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from fused_index_topk.api import (
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    StageNode,
    StageSpec,
    VariantDescriptor,
)
from fused_index_topk.provenance import path_fingerprint
from fused_index_topk.variants.common import supports_frozen_deepgemm_case

from .long_repair import load_long_context_repair
from .onchip_producer import load_long_producer
from .repair_producer import load_repair_producer
from .sampling import (
    common_random_sample_ids,
    guarded_sample_rank,
    load_sampling_extension,
)

_ROOT = Path(__file__).resolve().parent
_SAMPLING_SEED_XOR = 0x4655534544523549
_TOP_K = 2048
_TARGET_CANDIDATES = 2816
_FAST_CANDIDATE_CAPACITY = 5888
_REPAIR_CHUNK_ELEMENTS = 16384
_SEGMENTS = 16
_MAX_CONTEXT = 163840
_SPILL_PER_SEGMENT = 256
_SPILL_SEGMENTS = 8


def long_context_sample_elements(context_tokens: int) -> int:
    """Return the qualified long-context sampling schedule."""

    if context_tokens <= _REPAIR_CHUNK_ELEMENTS:
        raise ValueError("long-context sampling requires N > 16384")
    if context_tokens <= 32768:
        samples = 512
    elif context_tokens <= 65536:
        samples = 1024
    else:
        proportional = math.ceil(context_tokens * 3 / 256)
        samples = max(1024, math.ceil(proportional / 256) * 256)
    return min(context_tokens, samples)


def production_sample_elements(context_tokens: int) -> int:
    """Return the qualified threshold-sampling size for a supported context."""

    if context_tokens == _REPAIR_CHUNK_ELEMENTS:
        return 256
    if context_tokens < _REPAIR_CHUNK_ELEMENTS:
        return 512
    return long_context_sample_elements(context_tokens)


class FusedIndexTopK:
    """Same-kernel GEMM/Top-K with bounded exact device-side repair."""

    descriptor = VariantDescriptor(
        plugin_id="fused_index_topk",
        display_name="FusedIndexTopK",
        api_version="1.0",
        implementation_version="2.2.0",
        mode="fused",
        description=(
            "Same-kernel DeepGEMM-style score production and exact Top-K with "
            "bounded overflow salvage and device-masked exact repair"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision="fused-index-topk-v2.2",
        tags=(
            "prefill",
            "sm90",
            "deepseek-v3.2",
            "same-kernel-exact-topk",
            "bounded-overflow-salvage",
            "device-exactness-guard",
            "no-host-sync",
            "no-dense-logits",
        ),
    )

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options = dict(options or {})
        allowed = {
            "verbose_build",
            "sample_guard_sigmas",
            "debug_ctas",
            "diagnostic",
        }
        unknown = set(self.options) - allowed
        if unknown:
            raise ValueError(f"unknown FusedIndexTopK options: {sorted(unknown)}")
        guard = self.options.get("sample_guard_sigmas", 2.0)
        if (
            isinstance(guard, bool)
            or not isinstance(guard, (int, float))
            or not math.isfinite(guard)
            or guard < 0
        ):
            raise ValueError("sample_guard_sigmas must be finite and non-negative")
        self.sample_guard_sigmas = float(guard)
        self.debug_ctas = self.options.get("debug_ctas", 0)
        if type(self.debug_ctas) is not int or self.debug_ctas < 0:
            raise ValueError("debug_ctas must be a nonnegative integer")
        self.diagnostic = self.options.get("diagnostic", False)
        if type(self.diagnostic) is not bool:
            raise ValueError("diagnostic must be bool")
        self.tuning = 6
        self.math_schedule = 2
        self.math_registers = 184
        self.cache_weights = True

    def supports(self, case: PrefillCase) -> bool:
        return (
            supports_frozen_deepgemm_case(case)
            and 8192 <= case.context_tokens <= _MAX_CONTEXT
            and case.context_tokens % 128 == 0
            and case.top_k == _TOP_K
            and case.query_tokens % 2 == 0
        )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            "algorithm": "same-kernel-exact-topk-with-bounded-overflow-salvage-v2",
            "operator_sources": path_fingerprint(_ROOT),
            "options": self.options,
            "normal_candidate_gmem_bytes": 0,
            "spill_capacity_per_segment": _SPILL_PER_SEGMENT,
            "spill_workspace_bytes_per_row": _SPILL_SEGMENTS * _SPILL_PER_SEGMENT * 8,
            "spill_index_bits": 32,
            "candidate_index_bits": 16,
            "index_encoding": "segment-relative-lossless",
            "candidate_segments": 8,
            "max_context": _MAX_CONTEXT,
            "candidate_capacity": 5888,
            "fused_smem_bytes": 228160,
            "timed_repair": True,
            "repair_host_sync": False,
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
            raise ValueError(f"unsupported FusedIndexTopK case: {case}")
        if options and dict(options) != self.options:
            raise ValueError("variant options changed after FusedIndexTopK construction")
        import deep_gemm
        import torch

        verbose = bool(self.options.get("verbose_build", False))
        sampling = load_sampling_extension(verbose=verbose)
        fast_producer = load_long_producer(deep_gemm, verbose=verbose)
        chunk_producer = load_repair_producer(deep_gemm, verbose=verbose)
        long_repair = load_long_context_repair(verbose=verbose)

        rows = case.query_tokens
        # Preserve the qualified 16K sampling schedule while using the unified
        # bounded-overflow producer for every supported N.
        sample_elements = production_sample_elements(case.context_tokens)
        sample_rank = guarded_sample_rank(
            sample_elements,
            case.context_tokens,
            target_candidates=_TARGET_CANDIDATES,
            guard_sigmas=self.sample_guard_sigmas,
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

        output_ids = torch.empty((rows, _TOP_K), device=inputs.q.device, dtype=torch.int32)
        failure_flags = torch.empty(rows, device=inputs.q.device, dtype=torch.uint8)
        spill_pairs = torch.empty(
            (rows, _SPILL_SEGMENTS, _SPILL_PER_SEGMENT),
            device=inputs.q.device,
            dtype=torch.int64,
        )

        trace = torch.zeros(
            (rows // 2, 64) if self.diagnostic else (0,), device=inputs.q.device, dtype=torch.int64
        )

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

        num_chunks = (case.context_tokens + _REPAIR_CHUNK_ELEMENTS - 1) // _REPAIR_CHUNK_ELEMENTS
        chunk_starts = torch.empty((num_chunks, rows), device=inputs.q.device, dtype=torch.int32)
        chunk_ends = torch.empty_like(chunk_starts)
        chunks: list[tuple[int, int, Any, Any]] = []
        for logical_offset in range(0, case.context_tokens, _REPAIR_CHUNK_ELEMENTS):
            chunk_elements = min(
                _REPAIR_CHUNK_ELEMENTS,
                case.context_tokens - logical_offset,
            )
            chunk_index = logical_offset // _REPAIR_CHUNK_ELEMENTS
            chunk_start = chunk_starts[chunk_index]
            chunk_end = chunk_ends[chunk_index]
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

        def run_fused_topk(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard = artifacts["inputs"]
            arguments = (
                standard.q,
                standard.kv,
                standard.kv_scales,
                standard.weights,
                standard.k_start,
                standard.k_end,
                artifacts["thresholds"],
                output_ids,
                failure_flags,
                4,
                self.debug_ctas,
                self.tuning,
                self.math_schedule,
                self.math_registers,
                self.cache_weights,
                self.diagnostic,
                trace,
            )
            fast_producer.fp8_mqa_topk_out(*arguments, spill_pairs)
            artifacts["fast_failure_flags"] = failure_flags
            artifacts["fast_indices"] = output_ids.unsqueeze(1)
            artifacts["phase_trace"] = trace
            artifacts["spill_pairs"] = spill_pairs

        def run_hierarchical_repair(context: Any, artifacts: dict[str, Any]) -> None:
            del context
            standard: PrefillInputs = artifacts["inputs"]
            long_repair.update_ranges_out(
                standard.k_start, standard.k_end, chunk_starts, chunk_ends, case.context_tokens
            )
            for chunk_index, chunk in enumerate(chunks):
                logical_offset, chunk_elements, chunk_start, chunk_end = chunk
                chunk_kv = standard.kv.narrow(0, logical_offset, chunk_elements)
                chunk_scales = standard.kv_scales.narrow(0, logical_offset, chunk_elements)
                if not chunk_kv.is_contiguous() or not chunk_scales.is_contiguous():
                    raise RuntimeError("FusedIndexTopK repair chunk views must remain contiguous")
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
                    stage_id="fused_topk",
                    dependencies=("sample_threshold",),
                    consumes=("inputs", "thresholds"),
                    produces=("fast_failure_flags", "fast_indices"),
                    semantic_ops=("indexer", "topk", "output"),
                    description="GEMM and exact TopK in CTA SMEM; lossless segment-relative IDs",
                    kernel_regexes=("itk_fused_index_topk_long",),
                ),
                run_fused_topk,
            ),
            StageNode(
                StageSpec(
                    stage_id="hierarchical_repair",
                    dependencies=("fused_topk",),
                    consumes=("inputs", "fast_failure_flags", "fast_indices"),
                    produces=("failure_flags", "repair_error_flags", "indices"),
                    semantic_ops=("indexer", "topk", "output"),
                    description=(
                        "Device-masked exact 16K chunk rescans with online Top-2048 pair merge"
                    ),
                    kernel_regexes=(
                        "masked_repair_producer",
                        "itk_update_chunk_ranges",
                        "itk_chunk_local_topk",
                        "itk_merge_topk_pairs",
                        "itk_finalize_repair",
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
                "algorithm": "same-kernel-bounded-overflow-salvage-v2",
                "sample_elements": sample_elements,
                "sample_rank_descending": sample_rank,
                "sample_target_candidates": _TARGET_CANDIDATES,
                "sample_guard_sigmas": self.sample_guard_sigmas,
                "sampling_granularity": "individual-random-token",
                "dense_logits_materialized": False,
                "candidate_capacity": _FAST_CANDIDATE_CAPACITY,
                "candidate_segments": 8,
                "device_failure_flags": True,
                "timed_failure_repair": True,
                "repair_dispatch": "device-mask-no-host-sync",
                "repair_chunk_elements": _REPAIR_CHUNK_ELEMENTS,
                "repair_chunks": len(chunks),
                "repair_merge": "online-exact-top2048-pairs",
                "repair_workspace_bounded_in_n": True,
                "normal_candidate_gmem_bytes": 0,
                "spill_capacity_per_segment": _SPILL_PER_SEGMENT,
                "spill_workspace_bytes": spill_pairs.numel() * spill_pairs.element_size(),
                "spill_merge": "same-CTA-consumer-with-next-Q-math-overlap",
                "same_kernel_exact_topk": True,
                "upstream_deepgemm_modified": False,
                "promotion_status": "public-qualified",
                "sampling_resources": dict(sampling.resource_report()),
                "long_repair_resources": dict(long_repair.resource_report()),
            },
        )


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> FusedIndexTopK:
    return FusedIndexTopK(options)
