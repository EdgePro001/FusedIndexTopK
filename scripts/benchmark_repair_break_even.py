#!/usr/bin/env python3
"""Measure exact-repair latency as a controlled function of failed rows.

This is an out-of-band diagnostic. It leaves the public FusedIndexTopK device
implementation unchanged and inserts one constant-work device OR between the
fast reducer and repair.  Every forced-row level, including zero, pays for the
same injection kernel, so repair deltas are measured against the instrumented
zero-row control rather than against an uninstrumented operator.
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from index_topk_perflab.api import (
    PreparedGraph,
    RunMode,
    StageNode,
    StageSpec,
)
from index_topk_perflab.artifacts import write_json_atomic
from index_topk_perflab.benchmark import BenchmarkProtocol, benchmark_prepared_graph
from index_topk_perflab.config import load_config
from index_topk_perflab.graph import topological_nodes
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

BASE_VARIANTS = {
    "fused": "index_topk_perflab.experimental.fused_index_topk.plugin:create_variant",
}
REFERENCE_FACTORY = "index_topk_perflab.variants.deepgemm_torch:create_variant"
FLASHINFER_FACTORY = "index_topk_perflab.variants.deepgemm_flashinfer.plugin:create_auto_variant"


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min_ms": min(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values),
        "mean_ms": statistics.fmean(values),
    }


def _measurement_summary(
    result: Mapping[str, Any],
    *,
    pass_name: str,
    stage_id: str,
) -> dict[str, Any] | None:
    rows = [
        row
        for row in result["measurements"]
        if row["pass"] == pass_name and row["stage_id"] == stage_id
    ]
    if not rows:
        return None
    values = [float(row["latency_ms"]) for row in rows]
    fixtures: dict[str, Any] = {}
    for fixture_id in ("A", "B"):
        fixture_values = [
            float(row["latency_ms"]) for row in rows if row["fixture_id"] == fixture_id
        ]
        if fixture_values:
            fixtures[fixture_id] = _statistics(fixture_values)
    return {"all": _statistics(values), "by_fixture": fixtures}


class ForcedRepairVariant:
    """Diagnostic wrapper that ORs a prebuilt row mask into device flags."""

    def __init__(self, base_factory: str, forced_rows: int) -> None:
        self.base_factory = base_factory
        self.forced_rows = int(forced_rows)
        self.options: dict[str, Any] = {"verbose_build": False}
        self.base = load_variant(base_factory, options=self.options)
        self.descriptor = replace(
            self.base.descriptor,
            plugin_id=f"{self.base.descriptor.plugin_id}_forced_repair_{forced_rows}",
            display_name=(f"{self.base.descriptor.display_name} forced repair {forced_rows} rows"),
            implementation_version=(
                f"{self.base.descriptor.implementation_version}+diagnostic-force-{forced_rows}"
            ),
            description=(
                self.base.descriptor.description
                + f"; diagnostic device flag injection for {forced_rows} rows"
            ),
            tags=tuple(self.base.descriptor.tags)
            + ("diagnostic", "forced-repair", "constant-work-flag-injection"),
        )

    def supports(self, case: Any) -> bool:
        return self.base.supports(case) and 0 <= self.forced_rows <= case.query_tokens

    def fingerprint_metadata(self) -> Mapping[str, Any]:
        return {
            **self.base.fingerprint_metadata(),
            "diagnostic": "forced-repair-break-even-v1",
            "forced_rows": self.forced_rows,
            "injection": "device uint8 bitwise-or over all Q flags",
            "base_operator_modified": False,
        }

    def prepare(
        self,
        case: Any,
        inputs: Any,
        *,
        options: Mapping[str, Any],
        mode: RunMode,
    ) -> PreparedGraph:
        if dict(options) != self.options:
            raise ValueError("diagnostic options changed after construction")
        graph = self.base.prepare(case, inputs, options=self.options, mode=mode)

        import torch

        force_mask = torch.zeros(
            case.query_tokens,
            device=inputs.q.device,
            dtype=torch.uint8,
        )
        if self.forced_rows:
            force_mask[: self.forced_rows] = 1
        torch.cuda.synchronize()

        patched: list[StageNode] = []
        found_reducer = False
        found_repair = False
        for node in graph.nodes:
            if node.spec.stage_id == "candidate_reducer":
                found_reducer = True
                original_run = node.run

                def run_candidate_reducer(
                    context: Any,
                    artifacts: dict[str, Any],
                    run: Any = original_run,
                ) -> None:
                    run(context, artifacts)
                    artifacts["pre_injection_flags"] = artifacts.pop("fast_failure_flags")

                patched.append(
                    StageNode(
                        replace(
                            node.spec,
                            produces=("pre_injection_flags", "fast_indices"),
                            description=(
                                node.spec.description + "; expose flags before diagnostic injection"
                            ),
                        ),
                        run_candidate_reducer,
                    )
                )

                def run_repair_injection(
                    context: Any,
                    artifacts: dict[str, Any],
                ) -> None:
                    del context
                    flags = artifacts["pre_injection_flags"]
                    torch.bitwise_or(flags, force_mask, out=flags)
                    artifacts["fast_failure_flags"] = flags

                patched.append(
                    StageNode(
                        StageSpec(
                            stage_id="repair_injection",
                            dependencies=("candidate_reducer",),
                            consumes=("pre_injection_flags",),
                            produces=("fast_failure_flags",),
                            semantic_ops=("topk",),
                            description=(
                                "Diagnostic constant-work device OR of the forced-row mask"
                            ),
                            kernel_regexes=("bitwise_or",),
                        ),
                        run_repair_injection,
                    )
                )
                continue

            if node.spec.stage_id in {"repair_producer", "hierarchical_repair"}:
                found_repair = True
                dependencies = tuple(
                    "repair_injection" if item == "candidate_reducer" else item
                    for item in node.spec.dependencies
                )
                patched.append(StageNode(replace(node.spec, dependencies=dependencies), node.run))
                continue
            patched.append(node)

        if not found_reducer or not found_repair:
            raise RuntimeError("base graph lacks the expected reducer/repair boundary")
        graph.descriptor = self.descriptor
        graph.nodes = tuple(patched)
        graph.metadata = {
            **graph.metadata,
            "diagnostic": "forced-repair-break-even-v1",
            "forced_repair_rows": self.forced_rows,
            "constant_work_injection": True,
            "base_operator_modified": False,
        }
        return graph


def _probe_flags(
    lifecycle: CaseLifecycle,
    graph: PreparedGraph,
) -> dict[str, dict[str, int]]:
    """Read natural and post-injection counts outside the timing windows."""

    import torch

    ordered = topological_nodes(graph)
    result: dict[str, dict[str, int]] = {}
    for fixture_id in ("A", "B"):
        artifacts = lifecycle.bound_artifacts(graph, fixture_id)
        natural = None
        injected = None
        with torch.inference_mode():
            for node in ordered:
                node.run(None, artifacts)
                if node.spec.stage_id == "candidate_reducer":
                    torch.cuda.synchronize()
                    natural = int(torch.count_nonzero(artifacts["pre_injection_flags"]).item())
                if node.spec.stage_id == "repair_injection":
                    torch.cuda.synchronize()
                    injected = int(torch.count_nonzero(artifacts["fast_failure_flags"]).item())
                    break
        if natural is None or injected is None:
            raise RuntimeError("failed to observe diagnostic repair flags")
        result[fixture_id] = {
            "natural_failure_rows": natural,
            "post_injection_failure_rows": injected,
        }
    return result


def _summarize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    stage_ids = sorted(
        {
            str(row["stage_id"])
            for row in result["measurements"]
            if row["pass"] == "kineto_stage_kernel_sum"
        }
    )
    return {
        "cupti_operator": _measurement_summary(
            result, pass_name="formal_kernel_sum", stage_id="operator_total"
        ),
        "event_operator": _measurement_summary(
            result, pass_name="cuda_event_total", stage_id="operator_total"
        ),
        "cupti_stages": {
            stage_id: _measurement_summary(
                result,
                pass_name="kineto_stage_kernel_sum",
                stage_id=stage_id,
            )
            for stage_id in stage_ids
        },
        "exactness": result["postflight_exact_reference_check"]["status"],
        "topology": result["kineto"]["topology_gate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(BASE_VARIANTS), required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--split", default="test_hard")
    parser.add_argument(
        "--forced-rows",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--event-trials", type=int, default=100)
    parser.add_argument("--kineto-trials", type=int, default=50)
    parser.add_argument("--l2-flush-bytes", type=int, default=8_000_000_000)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/fused_index_topk_h20.json"),
    )
    args = parser.parse_args()

    import torch

    forced_rows = sorted(set(args.forced_rows))
    if not forced_rows or forced_rows[0] < 0 or forced_rows[-1] > 4096:
        raise ValueError("--forced-rows must lie in [0, 4096]")

    manifest = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    factory = ReplayInputFactory(manifest, split=args.split, verify_sha256=True)
    case = next(
        item for item in factory.cases(seed=20260825) if item.context_tokens == args.context
    )
    if case.query_tokens != 4096 or case.top_k != 2048:
        raise ValueError("the diagnostic is qualified only for Q=4096, K=2048")

    runtime_config = load_config(args.runtime_config)
    runtime = collect_runtime()
    validate_runtime(runtime, runtime_config)
    initial_exclusivity = validate_gpu_exclusivity(
        collect_gpu_state(), runtime, phase="repair_break_even_initial"
    )
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
    reference_options = {"sorted": False}
    reference = load_variant(REFERENCE_FACTORY, options=reference_options)
    lifecycle = CaseLifecycle(
        case=case,
        workload=workload,
        reference=reference,
        reference_options=reference_options,
        mode=RunMode.BENCHMARK,
        input_factory=factory,
    )
    oracle = lifecycle.build_oracles()

    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "diagnostic": "controlled-device-repair-break-even-v1",
        "variant": args.variant,
        "base_factory": BASE_VARIANTS[args.variant],
        "case": asdict(case),
        "split": args.split,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "protocol": asdict(protocol),
        "runtime_identity": runtime_identity(runtime),
        "initial_gpu_exclusivity": initial_exclusivity,
        "framework_sha256": framework_fingerprint(),
        "methodology": {
            "injection": ("one uint8 device bitwise-OR over all Q flags, paid by every level"),
            "repair_delta_reference": "instrumented forced_rows=0 control",
            "operator_source_modified": False,
            "fixtures": ["A", "B"],
            "baseline_bracketing": ["flashinfer_pre", "flashinfer_post"],
            "oracle": oracle,
        },
        "runs": [],
    }
    write_json_atomic(output / "summary.json", summary)

    def run_candidate(label: str, candidate: Any, flag_probe: Any = None) -> None:
        print(f"benchmark {label}", flush=True)
        exclusivity_before = validate_gpu_exclusivity(
            collect_gpu_state(), runtime, phase=f"{label}:before"
        )
        graph = lifecycle.prepare_candidate(candidate, {"verbose_build": False})
        observed = flag_probe(lifecycle, graph) if flag_probe is not None else None
        result = benchmark_prepared_graph(
            graph,
            case,
            protocol=protocol,
            run_id=f"repair-break-even-{args.variant}-n{args.context}-{label}",
            reference_metadata={
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "split": args.split,
                "oracle": oracle,
            },
            lifecycle=lifecycle,
        )
        exclusivity_after = validate_gpu_exclusivity(
            collect_gpu_state(), runtime, phase=f"{label}:after"
        )
        result["gpu_exclusivity"] = {
            "before": exclusivity_before,
            "after": exclusivity_after,
        }
        raw_path = output / f"{label}.json"
        write_json_atomic(raw_path, result)
        row = {
            "label": label,
            "plugin_id": candidate.descriptor.plugin_id,
            "forced_rows": getattr(candidate, "forced_rows", None),
            "observed_flags": observed,
            "source_metadata": dict(candidate.fingerprint_metadata()),
            "result": raw_path.name,
            **_summarize_result(result),
        }
        summary["runs"].append(row)
        write_json_atomic(output / "summary.json", summary)
        p50 = row["cupti_operator"]["all"]["p50_ms"]
        print(f"done {label}: CUPTI p50={p50:.6f} ms", flush=True)
        lifecycle.candidate_graph = None
        lifecycle.candidate_nodes = ()
        del result, graph
        gc.collect()
        torch.cuda.empty_cache()

    flashinfer_options = {"verbose_build": False}
    flashinfer_pre = load_variant(FLASHINFER_FACTORY, options=flashinfer_options)
    run_candidate("flashinfer_pre", flashinfer_pre)
    for count in forced_rows:
        candidate = ForcedRepairVariant(BASE_VARIANTS[args.variant], count)
        run_candidate(f"forced_{count:04d}", candidate, _probe_flags)
        del candidate
    flashinfer_post = load_variant(FLASHINFER_FACTORY, options=flashinfer_options)
    run_candidate("flashinfer_post", flashinfer_post)

    summary["status"] = "complete"
    summary["final_gpu_exclusivity"] = validate_gpu_exclusivity(
        collect_gpu_state(), runtime, phase="repair_break_even_final"
    )
    write_json_atomic(output / "summary.json", summary)
    print(output / "summary.json")


if __name__ == "__main__":
    main()
