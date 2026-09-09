#!/usr/bin/env python3
"""Run the unchanged formal CUPTI/Event benchmark on frozen replay inputs."""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from index_topk_perflab.api import RunMode
from index_topk_perflab.artifacts import canonical_hash, write_json_atomic
from index_topk_perflab.benchmark import BenchmarkProtocol, benchmark_prepared_graph
from index_topk_perflab.config import load_config
from index_topk_perflab.lifecycle import CaseLifecycle
from index_topk_perflab.provenance import framework_fingerprint
from index_topk_perflab.registry import load_variant
from index_topk_perflab.replay import ReplayInputFactory, sha256_file
from index_topk_perflab.runtime import (
    collect_gpu_state,
    collect_runtime,
    runtime_identity,
    validate_gpu_exclusivity,
    validate_runtime,
)

VARIANTS = {
    "torch": (
        "index_topk_perflab.variants.deepgemm_torch:create_variant",
        {"sorted": False},
    ),
    "flashinfer": (
        "index_topk_perflab.variants.deepgemm_flashinfer.plugin:create_auto_variant",
        {"verbose_build": False},
    ),
    "r13a": (
        "index_topk_perflab.experimental.fused_r13a_nsweep.plugin:create_variant",
        {"verbose_build": False},
    ),
    "r16a": (
        "index_topk_perflab.experimental.fused_r16a.plugin:create_variant",
        {"verbose_build": False},
    ),
}


def _source_identity(
    plugin: Any,
    *,
    factory: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Bind one result to the exact framework and operator source bytes."""

    payload = {
        "schema_version": 1,
        "descriptor": asdict(plugin.descriptor),
        "factory": factory,
        "options": dict(options),
        "fingerprint_metadata": dict(plugin.fingerprint_metadata()),
    }
    return {
        "sha256": canonical_hash(payload),
        "payload": payload,
    }


def _metric(result: dict[str, Any], pass_name: str, stage_id: str) -> dict[str, Any] | None:
    rows = [
        row
        for row in result["measurements"]
        if row["pass"] == pass_name and row["stage_id"] == stage_id
    ]
    values = [float(row["latency_ms"]) for row in rows]
    if not values:
        return None
    by_fixture = {
        fixture: [float(row["latency_ms"]) for row in rows if row["fixture_id"] == fixture]
        for fixture in ("A", "B")
    }
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "fixture_median_ms": {
            fixture: statistics.median(samples)
            for fixture, samples in by_fixture.items()
            if samples
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["test_normal", "test_hard"])
    parser.add_argument("--variants", nargs="+", default=["flashinfer", "r13a"])
    parser.add_argument("--contexts", type=int, nargs="+", default=[])
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--event-trials", type=int, default=20)
    parser.add_argument("--kineto-trials", type=int, default=30)
    parser.add_argument("--l2-flush-bytes", type=int, default=8_000_000_000)
    parser.add_argument("--extra-site-packages")
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/r13a_h20_release.json"),
    )
    args = parser.parse_args()

    import torch

    if args.extra_site_packages:
        sys.path.append(args.extra_site_packages)

    unknown = set(args.variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = BenchmarkProtocol(
        warmup_iterations=args.warmup_iterations,
        event_trials=args.event_trials,
        kineto_trials=args.kineto_trials,
        l2_flush_bytes=args.l2_flush_bytes,
    )
    workload = SimpleNamespace(
        q_dtype="float8_e4m3fn",
        kv_dtype="float8_e4m3fn",
        kv_scale_dtype="float32",
        weight_dtype="float32",
        range_dtype="int32",
    )
    runtime_config = load_config(args.runtime_config)
    runtime = collect_runtime()
    validate_runtime(runtime, runtime_config)
    runtime_fingerprint = runtime_identity(runtime)
    initial_exclusivity = validate_gpu_exclusivity(
        collect_gpu_state(), runtime, phase="replay_benchmark_initial"
    )
    reference_options = {"sorted": False}
    summary_rows: list[dict[str, Any]] = []
    for split in args.splits:
        factory = ReplayInputFactory(manifest, split=split, verify_sha256=True)
        cases = factory.cases(seed=20260825)
        if args.contexts:
            cases = tuple(case for case in cases if case.context_tokens in args.contexts)
        for case in cases:
            for variant_name in args.variants:
                reference = load_variant(
                    "index_topk_perflab.variants.deepgemm_torch:create_variant",
                    options=reference_options,
                )
                reference_factory = (
                    "index_topk_perflab.variants.deepgemm_torch:create_variant"
                )
                factory_reference, options = VARIANTS[variant_name]
                candidate = load_variant(factory_reference, options=options)
                run_id = f"real-replay-{split}-n{case.context_tokens}-{variant_name}"
                print(
                    f"benchmark split={split} N={case.context_tokens} variant={variant_name}",
                    flush=True,
                )
                exclusivity_before = validate_gpu_exclusivity(
                    collect_gpu_state(), runtime, phase=f"{run_id}:before"
                )
                lifecycle = CaseLifecycle(
                    case=case,
                    workload=workload,
                    reference=reference,
                    reference_options=reference_options,
                    mode=RunMode.BENCHMARK,
                    input_factory=factory,
                )
                oracle = lifecycle.build_oracles()
                graph = lifecycle.prepare_candidate(candidate, options)
                result = benchmark_prepared_graph(
                    graph,
                    case,
                    protocol=protocol,
                    run_id=run_id,
                    reference_metadata={
                        "replay_manifest": str(manifest),
                        "replay_manifest_sha256": sha256_file(manifest),
                        "replay_split": split,
                        "oracle": oracle,
                    },
                    lifecycle=lifecycle,
                )
                exclusivity_after = validate_gpu_exclusivity(
                    collect_gpu_state(), runtime, phase=f"{run_id}:after"
                )
                result["runtime"] = runtime
                result["runtime_identity"] = runtime_fingerprint
                result["source_identity"] = {
                    "framework_sha256": framework_fingerprint(),
                    "reference": _source_identity(
                        reference,
                        factory=reference_factory,
                        options=reference_options,
                    ),
                    "candidate": _source_identity(
                        candidate,
                        factory=factory_reference,
                        options=options,
                    ),
                }
                result["gpu_exclusivity"] = {
                    "before": exclusivity_before,
                    "after": exclusivity_after,
                }
                result_path = output / split / f"n{case.context_tokens}_{variant_name}.json"
                write_json_atomic(result_path, result)
                stage_ids = sorted(
                    {
                        str(item["stage_id"])
                        for item in result["measurements"]
                        if item["pass"] == "kineto_stage_kernel_sum"
                    }
                )
                row = {
                    "split": split,
                    "context_tokens": case.context_tokens,
                    "query_tokens": case.query_tokens,
                    "variant": variant_name,
                    "plugin_id": candidate.descriptor.plugin_id,
                    "source_identity_sha256": result["source_identity"]["candidate"][
                        "sha256"
                    ],
                    "framework_sha256": result["source_identity"][
                        "framework_sha256"
                    ],
                    "promotion_eligible": bool(candidate.descriptor.exact_topk),
                    "result": result_path.relative_to(output).as_posix(),
                    "formal_operator": _metric(result, "formal_kernel_sum", "operator_total"),
                    "cupti_indexer": _metric(result, "kineto_stage_kernel_sum", "indexer"),
                    "cupti_topk": _metric(result, "kineto_stage_kernel_sum", "topk"),
                    "cupti_output": _metric(result, "kineto_stage_kernel_sum", "output"),
                    "cupti_stages": {
                        stage_id: _metric(result, "kineto_stage_kernel_sum", stage_id)
                        for stage_id in stage_ids
                    },
                    "cuda_event_operator": _metric(result, "cuda_event_total", "operator_total"),
                    "exactness": result["postflight_exact_reference_check"]["status"],
                    "kineto_topology": result["kineto"]["topology_gate"]["status"],
                    "gpu_exclusivity": "passed",
                }
                summary_rows.append(row)
                write_json_atomic(
                    output / "summary.json",
                    {
                        "schema_version": 1,
                        "manifest": str(manifest),
                        "manifest_sha256": sha256_file(manifest),
                        "runtime_identity": runtime_fingerprint,
                        "initial_gpu_exclusivity": initial_exclusivity,
                        "protocol": {
                            "method": protocol.method,
                            "warmup_iterations": protocol.warmup_iterations,
                            "event_trials": protocol.event_trials,
                            "kineto_trials": protocol.kineto_trials,
                            "l2_flush_bytes": protocol.l2_flush_bytes,
                        },
                        "rows": summary_rows,
                    },
                )
                print(
                    f"done {variant_name}: CUPTI operator="
                    f"{row['formal_operator']['median_ms']:.6f} ms",
                    flush=True,
                )
                del result, graph, lifecycle, candidate, reference
                gc.collect()
                torch.cuda.empty_cache()
    print(output / "summary.json")


if __name__ == "__main__":
    main()
