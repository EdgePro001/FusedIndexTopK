"""Resolved, machine-readable definition of the fusion-v1 research problem."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

PROBLEM_CONTRACT_VERSION = "fusion-v1"

FORMAL_DECISION_POLICY = {
    "minimum_campaign_blocks": 3,
    "minimum_trials_per_run": 20,
    "maximum_normalized_mad": 0.05,
    "maximum_p90_to_median_ratio": 1.10,
    "improved_minimum_speedup": 1.03,
    "regressed_maximum_speedup": 0.97,
    "maximum_candidate_to_baseline_p95_ratio": 1.05,
    "confidence_level": 0.95,
    "campaign_bootstrap_repetitions": 10_000,
    "uncertainty_unit": "independent_campaign_block",
}


def protocol_identity(config: Any) -> dict[str, Any]:
    """Hash only input/output semantics, independent of the run matrix."""

    workload = config.workload
    payload = {
        "version": PROBLEM_CONTRACT_VERSION,
        "batch_size": workload.batch_size,
        "top_k": workload.top_k,
        "indexer_heads": workload.indexer_heads,
        "head_dim": workload.head_dim,
        "causal": workload.causal,
        "q_dtype": workload.q_dtype,
        "kv_dtype": workload.kv_dtype,
        "kv_scale_dtype": workload.kv_scale_dtype,
        "weight_dtype": workload.weight_dtype,
        "range_dtype": workload.range_dtype,
        "score_dtype": workload.score_dtype,
        "output_dtype": workload.output_dtype,
        "selection": workload.selection,
        "output_order": workload.output_order,
        "padding_index": workload.padding_index,
        "tie_policy": workload.tie_policy,
        "causal_range": workload.causal_range,
        "input_recipe_version": "contiguous-prefill-fp8-v2",
    }
    from .artifacts import canonical_hash

    return {"sha256": canonical_hash(payload), "payload": payload}


def plan_identity(config: Any) -> dict[str, Any]:
    """Hash only the formal correctness/benchmark plan.

    Nsight targets and metric lists are diagnostic controls.  They deliberately
    do not invalidate formal correctness or benchmark identities.
    """

    payload = {
        "schema_version": config.schema_version,
        "name": config.name,
        "seed": config.seed,
        "target": asdict(config.target),
        "benchmark_cases": [asdict(case) for case in config.workload.benchmark_cases],
        "correctness_cases": [asdict(case) for case in config.workload.correctness_cases],
        "exact_reference_variant": config.exact_reference_variant,
        "timing": asdict(config.timing),
        "formal_decision_policy": FORMAL_DECISION_POLICY,
    }
    from .artifacts import canonical_hash

    return {"sha256": canonical_hash(payload), "payload": payload}


def profiling_plan_identity(
    config: Any,
    *,
    mode: str,
    target_length: int,
    stage: str,
) -> dict[str, Any]:
    """Hash one diagnostic capture plan independently of formal timing."""

    if mode not in {"nsys", "ncu"}:
        raise ValueError("profile mode must be nsys or ncu")
    matching_cases = [
        case
        for case in config.profile_shapes(mode)
        if case.context_tokens == target_length
    ]
    if not matching_cases:
        raise ValueError(f"target length {target_length} is not configured for {mode}")
    if len(matching_cases) != 1:
        raise ValueError(
            f"target length {target_length} is ambiguous for {mode}; select a Q/N case"
        )
    target_case = matching_cases[0]
    payload = {
        "schema_version": config.schema_version,
        "formal_plan_hash": plan_identity(config)["sha256"],
        "mode": mode,
        "target_case": asdict(target_case),
        "stage": stage,
        "captured_iterations": config.profiling.captured_iterations,
        "ncu_metrics": list(config.profiling.ncu_metrics) if mode == "ncu" else [],
        "ncu_sections": list(config.profiling.ncu_sections) if mode == "ncu" else [],
    }
    from .artifacts import canonical_hash

    return {"sha256": canonical_hash(payload), "payload": payload}


def resolved_problem_contract(config: Any) -> dict[str, Any]:
    """Resolve config fields and fixed input-generation semantics in one place.

    The physical implementation is intentionally absent: a legal plugin may
    materialize logits, stream candidates, or use one fused kernel.  Only the
    input-to-indices semantics and the formal measurement boundary are frozen.
    """

    workload = config.workload
    benchmark_cases = [asdict(case) for case in workload.benchmark_cases]
    correctness_cases = [asdict(case) for case in workload.correctness_cases]
    query_token_values = sorted({case.query_tokens for case in workload.benchmark_cases})
    context_token_values = sorted({case.context_tokens for case in workload.benchmark_cases})
    irregular_cases = [
        asdict(case)
        for case in workload.benchmark_cases
        if case.context_tokens & (case.context_tokens - 1)
    ]
    return {
        "version": PROBLEM_CONTRACT_VERSION,
        "scope": "contiguous causal prefill forward Indexer plus exact TopK",
        "target": asdict(config.target),
        "dimensions": {
            "batch_size": workload.batch_size,
            "query_token_values": query_token_values,
            "indexer_heads": workload.indexer_heads,
            "head_dim": workload.head_dim,
            "top_k": workload.top_k,
            "benchmark_cases": benchmark_cases,
            "context_token_values": context_token_values,
            "irregular_benchmark_cases": irregular_cases,
            "correctness_cases": correctness_cases,
            "correctness_kv_tile_remainders": {
                f"q{case.query_tokens}-kv{case.context_tokens}": case.context_tokens % 256
                for case in workload.correctness_cases
            },
        },
        "inputs": {
            "q": {
                "shape": "[Q,H,D]",
                "dtype": workload.q_dtype,
                "layout": "contiguous",
            },
            "kv": {
                "shape": "[N,D]",
                "dtype": workload.kv_dtype,
                "layout": "contiguous",
            },
            "kv_scales": {
                "shape": "[N]",
                "dtype": workload.kv_scale_dtype,
                "layout": "contiguous",
                "granularity": "one scale per KV row",
            },
            "weights": {
                "shape": "[Q,H]",
                "dtype": workload.weight_dtype,
                "layout": "contiguous",
            },
            "k_start": {
                "shape": "[Q]",
                "dtype": workload.range_dtype,
                "value": "0",
            },
            "k_end": {
                "shape": "[Q]",
                "dtype": workload.range_dtype,
                "value": "N-Q+q+1",
                "interval_semantics": "end-exclusive",
            },
        },
        "input_recipe": {
            "seed": config.seed,
            "source_dtype": "bfloat16",
            "version": "contiguous-prefill-fp8-v2",
            "fixtures": ["A", "B"],
            "generation_order": "independent_rng_streams",
            "q": "normal / sqrt(D), then direct cast to float8_e4m3fn",
            "kv": "normal / sqrt(D), rowwise amax clamp 1e-4 / 448 quantization",
            "weights": "normal / sqrt(H), retained as float32",
            "storage": "compact per case; no larger backing tensor views",
            "content_integrity": "SHA-256 before and after candidate lifecycle",
        },
        "score": {
            "dtype": workload.score_dtype,
            "valid_range": workload.causal_range,
            "formula": (
                "kv_scale[k] * sum_h(weights[q,h] * "
                "relu(sum_d(q_fp8[q,h,d] * kv_fp8[k,d])))"
            ),
            "invalid_value": "-inf",
            "oracle": "realized released-DeepGEMM FP32 scores",
            "exact_reference_variant": config.exact_reference_variant,
        },
        "selection": {
            "mode": workload.selection,
            "top_k": workload.top_k,
            "order": workload.output_order,
            "tie_policy": workload.tie_policy,
            "tie_precondition": "score[K-1] != score[K] whenever valid_count > K",
        },
        "output": {
            "name": "indices",
            "shape": "[Q,1,K]",
            "dtype": workload.output_dtype,
            "layout": "contiguous",
            "padding_index": workload.padding_index,
            "scores_returned": False,
        },
        "supported_range_policy": {
            "k_start": "zero only",
            "context_parallel_nonzero_start": False,
            "includes_current_query_position": True,
        },
        "formal_measurement": {
            "scope": "standard inputs to final INT32 indices",
            "primary_timing_source": "DeepGEMM-style Kineto/CUPTI kernel sum",
            "supplemental_timing_source": "whole-pipeline CUDA Event",
            "method": config.timing.method,
            "warmup_iterations": config.timing.warmup_iterations,
            "kineto_trials": config.timing.kineto_trials,
            "event_trials": config.timing.event_trials,
            "l2_flush_bytes": config.timing.l2_flush_bytes,
            "kineto_schedule": {"wait": 1, "warmup": 0, "active": 1, "repeat": 1},
            "cooldown_seconds": config.timing.cooldown_seconds,
            "campaign_orders": list(config.timing.campaign_orders),
            "decision_policy": FORMAL_DECISION_POLICY,
            "performance_baseline_variant": config.baseline_variant,
            "excluded": [
                "input generation and quantization",
                "JIT compilation and autotuning",
                "correctness oracle",
                "cache scrub itself",
                "Nsight instrumentation",
                "downstream sparse attention",
            ],
        },
    }
