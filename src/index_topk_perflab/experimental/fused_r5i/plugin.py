"""Experimental producer-side DeepGEMM + sampled exact-TopK fast path."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from .producer import load_candidate_producer
from .sampling import (
    common_random_sample_ids,
    guarded_sample_rank,
    load_sampling_extension,
    sample_elements_for_context,
)

_ROOT = Path(__file__).resolve().parent
_SAMPLING_SEED_XOR = 0x4655534544523549
_TARGET_CANDIDATES = 3072
_CANDIDATE_CAPACITY = 14080
_CANDIDATE_SEGMENTS = 16
_REPAIR_CANDIDATE_CAPACITY = 16384


class DeepGemmFusedCandidateR5i:
    """No-dense-logits R5i with device-masked complete-row exact repair."""

    descriptor = VariantDescriptor(
        plugin_id="deepgemm_fused_candidate_topk_r5i",
        display_name="DeepGEMM producer-fused sampled candidate TopK R5i",
        api_version="1.0",
        implementation_version=(
            "producer-fused-sampled-r5i-masked-exact-repair-v3"
        ),
        mode="fused",
        description=(
            "Random-token KV sample threshold prepass, DeepGEMM score epilogue "
            "candidate emission, exact candidate radix, and device-masked "
            "complete-row repair"
        ),
        implementation="project-local-deepgemm-derivative+custom-cuda-radix",
        exact_topk=True,
        source_revision=(
            "deepgemm:7c95b14;producer-fused-r5i-masked-exact-repair-v3"
        ),
        tags=(
            "experimental",
            "prefill",
            "sm90",
            "producer-fusion",
            "no-dense-logits",
            "random-token-sampling",
            "candidate-radix",
            "role-split-one-ballot",
            "device-exactness-guard",
            "masked-repair-producer",
            "complete-row-repair-candidates",
            "exact-repair-radix",
            "no-host-sync",
        ),
    )

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self.options = dict(options or {})
        unknown = set(self.options) - {"verbose_build"}
        if unknown:
            raise ValueError(f"unknown fused R5i options: {sorted(unknown)}")

    def supports(self, case: PrefillCase) -> bool:
        return (
            supports_frozen_deepgemm_case(case)
            and case.context_tokens == _REPAIR_CANDIDATE_CAPACITY
            and case.top_k == 2048
            and case.query_tokens % 2 == 0
        )

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            "algorithm": "producer-fused-sampled-r5i-masked-exact-repair-v3",
            "project_sources": path_fingerprint(_ROOT / "csrc"),
            "plugin_python": path_fingerprint(Path(__file__)),
            "upstream_deepgemm_modified": False,
            "timed_repair": True,
            "repair_dispatch": "device-mask-no-host-sync",
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
            raise ValueError(f"unsupported fused R5i case: {case}")
        if options and dict(options) != self.options:
            raise ValueError("variant options changed after fused R5i construction")

        import deep_gemm
        import torch

        verbose = bool(self.options.get("verbose_build", False))
        sampling = load_sampling_extension(verbose=verbose)
        producer = load_candidate_producer(deep_gemm, verbose=verbose)
        reducer = load_segmented_candidate_reducer(verbose=verbose)

        sample_elements = sample_elements_for_context(case.context_tokens)
        sample_rank = guarded_sample_rank(
            sample_elements,
            case.context_tokens,
            target_candidates=_TARGET_CANDIDATES,
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
            sample_elements, device=inputs.q.device, dtype=torch.float32
        )
        sample_start = torch.zeros_like(inputs.k_start)
        sample_end = torch.full_like(inputs.k_end, sample_elements)
        thresholds = torch.empty(
            case.query_tokens, device=inputs.q.device, dtype=torch.float32
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
        output_ids = torch.empty(
            (case.query_tokens, case.top_k),
            device=inputs.q.device,
            dtype=torch.int32,
        )
        failure_flags = torch.empty(
            case.query_tokens, device=inputs.q.device, dtype=torch.uint8
        )
        repair_candidate_pairs = torch.empty(
            (case.query_tokens, _REPAIR_CANDIDATE_CAPACITY),
            device=inputs.q.device,
            dtype=torch.int64,
        )
        repair_segment_counts = torch.empty(
            (case.query_tokens, _CANDIDATE_SEGMENTS),
            device=inputs.q.device,
            dtype=torch.int32,
        )

        def run_sample_gather(context: Any, artifacts: dict[str, Any]) -> None:
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
            standard: PrefillInputs = artifacts["inputs"]
            artifacts["sampled_scores"] = deep_gemm.fp8_mqa_logits(
                standard.q,
                (artifacts["sampled_kv"], artifacts["sampled_scales"]),
                standard.weights,
                sample_start,
                sample_end,
                clean_logits=False,
            )

        def run_sample_threshold(context: Any, artifacts: dict[str, Any]) -> None:
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
            standard: PrefillInputs = artifacts["inputs"]
            producer.fp8_mqa_candidate_r5i_out(
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

        def run_candidate_reducer(context: Any, artifacts: dict[str, Any]) -> None:
            standard: PrefillInputs = artifacts["inputs"]
            reducer.topk_out(
                artifacts["candidate_pairs"],
                artifacts["segment_counts"],
                standard.k_start,
                standard.k_end,
                output_ids,
                failure_flags,
            )
            artifacts["fast_failure_flags"] = failure_flags
            artifacts["fast_indices"] = output_ids.unsqueeze(1)

        def run_repair_producer(context: Any, artifacts: dict[str, Any]) -> None:
            standard: PrefillInputs = artifacts["inputs"]
            producer.fp8_mqa_candidate_repair_r5i_out(
                standard.q,
                standard.kv,
                standard.kv_scales,
                standard.weights,
                standard.k_start,
                standard.k_end,
                artifacts["fast_failure_flags"],
                repair_candidate_pairs,
                repair_segment_counts,
            )
            artifacts["repair_candidate_pairs"] = repair_candidate_pairs
            artifacts["repair_segment_counts"] = repair_segment_counts

        def run_repair_reducer(context: Any, artifacts: dict[str, Any]) -> None:
            standard: PrefillInputs = artifacts["inputs"]
            reducer.repair_topk_out(
                artifacts["repair_candidate_pairs"],
                artifacts["repair_segment_counts"],
                standard.k_start,
                standard.k_end,
                output_ids,
                artifacts["fast_failure_flags"],
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
                    description="Gather one common random token subset from KV",
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
                    description="Causal-aware four-pass radix sample quantile",
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
                    description=(
                        "DeepGEMM FP8 MQA with threshold/candidate epilogue; "
                        "never materializes dense logits"
                    ),
                    kernel_regexes=("itk_sm90_fp8_mqa_candidate_producer_r5i",),
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
                    description=(
                        "Exact radix TopK over bounded fast candidates plus "
                        "device failure flags"
                    ),
                    kernel_regexes=("itk_fused_r5i_segmented_candidate_radix",),
                ),
                run_candidate_reducer,
            ),
            StageNode(
                StageSpec(
                    stage_id="repair_producer",
                    dependencies=("candidate_reducer",),
                    consumes=("inputs", "fast_failure_flags"),
                    produces=(
                        "repair_candidate_pairs",
                        "repair_segment_counts",
                    ),
                    semantic_ops=("indexer", "topk"),
                    description=(
                        "Device-masked DeepGEMM rescan emitting every score "
                        "only for failed rows"
                    ),
                    kernel_regexes=(
                        "itk_sm90_fp8_mqa_masked_repair_producer_r5i",
                    ),
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
                    description=(
                        "Device-masked exact radix repair over complete-row "
                        "candidates; safe rows preserve fast output"
                    ),
                    kernel_regexes=("itk_fused_r5i_segmented_candidate_radix",),
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
                "sample_elements": sample_elements,
                "sample_rank_descending": sample_rank,
                "sample_target_candidates": _TARGET_CANDIDATES,
                "sample_guard_sigmas": 2.0,
                "sampling_granularity": "individual-random-token",
                "sampling_without_replacement": True,
                "sampling_ids_sorted_after_selection": True,
                "sampled_kv_gather_timed": True,
                "sampled_indexer": "unmodified-deepgemm-fp8-mqa",
                "dense_logits_materialized": False,
                "candidate_capacity": _CANDIDATE_CAPACITY,
                "candidate_segments": _CANDIDATE_SEGMENTS,
                "candidate_segment_layout": "math-warp",
                "candidate_segment_capacity": (
                    _CANDIDATE_CAPACITY // _CANDIDATE_SEGMENTS
                ),
                "device_failure_flags": True,
                "timed_failure_repair": True,
                "repair_dispatch": "device-mask-no-host-sync",
                "repair_context_specialization": _REPAIR_CANDIDATE_CAPACITY,
                "repair_candidate_capacity": _REPAIR_CANDIDATE_CAPACITY,
                "repair_candidate_segments": _CANDIDATE_SEGMENTS,
                "repair_candidate_segment_capacity": (
                    _REPAIR_CANDIDATE_CAPACITY // _CANDIDATE_SEGMENTS
                ),
                "repair_candidate_completeness": (
                    "all scores in [k_start, k_end) for every flagged row"
                ),
                "promotion_status": "exact-candidate",
                "sampling_resources": dict(sampling.resource_report()),
                "reducer_resources": dict(reducer.resource_report()),
            },
        )


def create_variant(
    options: Mapping[str, Any] | None = None,
) -> DeepGemmFusedCandidateR5i:
    return DeepGemmFusedCandidateR5i(options)
