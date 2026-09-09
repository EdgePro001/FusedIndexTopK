"""Correctness-gated Kineto/CUPTI and CUDA-event operator benchmark.

The primary pass follows the pinned DeepGEMM ``bench_kineto`` schedule: one
initialization call, one wait step, one active step, 30 invocations per step,
and an actual 8 GB memset before every invocation.  CUPTI kernel sums are the
formal optimization metric.  A separate, unprofiled CUDA-event pass reports
realized whole-pipeline device latency.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .api import (
    ArtifactStore,
    ExecutionContext,
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    StageNode,
    VariantPlugin,
)
from .artifacts import canonical_hash
from .correctness import compare_topk_indices, validate_topk_indices
from .graph import (
    execute_graph,
    execute_prevalidated,
    graph_fingerprint,
    graph_mapping,
    validate_graph,
)
from .inputs import input_manifest
from .kineto import (
    extract_active_trials,
    operator_range_name,
    stage_range_name,
    validate_trial_topology,
)
from .lifecycle import FIXTURE_IDS, CaseLifecycle
from .summary import summarize_measurements


@dataclass(frozen=True)
class BenchmarkProtocol:
    method: str = "deepgemm_kineto_cupti_v1"
    warmup_iterations: int = 10
    event_trials: int = 20
    kineto_trials: int = 30
    l2_flush_bytes: int = 8_000_000_000
    cooldown_seconds: float = 0.0

    def validate(self) -> None:
        if self.method != "deepgemm_kineto_cupti_v1":
            raise ValueError("unsupported benchmark timing method")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.event_trials <= 0:
            raise ValueError("event_trials must be positive")
        if self.kineto_trials <= 0:
            raise ValueError("kineto_trials must be positive")
        if self.l2_flush_bytes <= 0 or self.l2_flush_bytes % 4:
            raise ValueError("l2_flush_bytes must be positive and divisible by four")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | Any) -> "BenchmarkProtocol":
        timing = config.get("timing", config) if isinstance(config, Mapping) else config.timing

        def field(name: str, default: Any) -> Any:
            return timing.get(name, default) if isinstance(timing, Mapping) else getattr(
                timing, name, default
            )

        result = cls(
            method=str(field("method", "deepgemm_kineto_cupti_v1")),
            warmup_iterations=int(field("warmup_iterations", 10)),
            event_trials=int(field("event_trials", 20)),
            kineto_trials=int(field("kineto_trials", 30)),
            l2_flush_bytes=int(field("l2_flush_bytes", 8_000_000_000)),
            cooldown_seconds=float(field("cooldown_seconds", 0.0)),
        )
        result.validate()
        return result


def prepare_variant(
    plugin: VariantPlugin,
    case: PrefillCase,
    inputs: PrefillInputs,
    *,
    options: Mapping[str, Any] | None = None,
    mode: RunMode = RunMode.BENCHMARK,
) -> PreparedGraph:
    """Prepare and statically validate a plugin graph outside measurements."""

    if not plugin.supports(case):
        raise ValueError(f"variant {plugin.descriptor.plugin_id!r} does not support {case.case_id}")
    graph = plugin.prepare(case, inputs, options=dict(options or {}), mode=mode)
    if graph.descriptor.plugin_id != plugin.descriptor.plugin_id:
        raise ValueError("prepared graph descriptor does not match the loaded plugin")
    validate_graph(graph)
    return graph


def warmup_prepared_graph(
    graph: PreparedGraph,
    case: PrefillCase,
    *,
    iterations: int,
    stream: Any = None,
) -> None:
    """Run JIT/autotune/allocation-triggering iterations and synchronize."""

    if iterations < 0:
        raise ValueError("warmup iterations must be non-negative")

    import torch

    selected_stream = stream if stream is not None else torch.cuda.current_stream()
    with torch.inference_mode(), torch.cuda.stream(selected_stream):
        for iteration in range(iterations):
            artifacts = execute_graph(
                graph,
                ExecutionContext(
                    case=case,
                    mode=RunMode.BENCHMARK,
                    iteration=iteration,
                    stream=selected_stream,
                ),
            )
            if graph.terminal_artifact not in artifacts:
                raise RuntimeError("warmup graph did not produce its terminal artifact")
            del artifacts
    selected_stream.synchronize()


def _run_nodes(
    nodes: Sequence[StageNode],
    context: ExecutionContext,
    artifacts: ArtifactStore,
) -> None:
    """Low-overhead execution after graph validation and warmup."""

    execute_prevalidated(nodes, context, artifacts)


def _measurement(
    *,
    graph: PreparedGraph,
    case: PrefillCase,
    pass_name: str,
    trial_id: int,
    scope_id: str,
    stage_id: str,
    semantic_ops: Sequence[str],
    latency_ms: float,
    timing_source: str,
    run_id: str | None,
    graph_hash: str,
    fixture_id: str = "A",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "variant_id": graph.descriptor.plugin_id,
        "variant_version": graph.descriptor.implementation_version,
        "variant_mode": graph.descriptor.mode,
        "graph_fingerprint": graph_hash,
        "case_id": case.case_id,
        "batch_size": case.batch_size,
        "query_tokens": case.query_tokens,
        "query_start": case.query_start,
        "context_tokens": case.context_tokens,
        "top_k": case.top_k,
        "pass": pass_name,
        "trial_id": trial_id,
        "fixture_id": fixture_id,
        "scope_id": scope_id,
        "stage_id": stage_id,
        "semantic_ops": ",".join(semantic_ops),
        "timing_source": timing_source,
        "latency_ms": float(latency_ms),
        "derived": False,
    }


def _terminal_description(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "contiguous": bool(value.is_contiguous()),
    }


def _allocator_state(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _memory_result(torch: Any, before: Mapping[str, int], *, interpretation: str) -> dict[str, Any]:
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "allocated_before_bytes": int(before["allocated_bytes"]),
        "reserved_before_bytes": int(before["reserved_bytes"]),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_allocated_delta_bytes": max(
            0, peak_allocated - int(before["allocated_bytes"])
        ),
        "peak_reserved_delta_bytes": max(
            0, peak_reserved - int(before["reserved_bytes"])
        ),
        "allocated_end_bytes": int(torch.cuda.memory_allocated()),
        "reserved_end_bytes": int(torch.cuda.memory_reserved()),
        "interpretation": interpretation,
    }


def _deepgemm_l2_flush(torch: Any, *, bytes_count: int, device: Any) -> None:
    """Enqueue the exact 8 GB-style memset expression used by DeepGEMM."""

    torch.empty(bytes_count // 4, dtype=torch.int, device=device).zero_()


def benchmark_prepared_graph(
    graph: PreparedGraph,
    case: PrefillCase,
    *,
    protocol: BenchmarkProtocol,
    stream: Any = None,
    run_id: str | None = None,
    reference_indices: Any | None = None,
    reference_metadata: Mapping[str, Any] | None = None,
    lifecycle: CaseLifecycle | None = None,
) -> dict[str, Any]:
    """Warm and benchmark one already-prepared graph.

    The returned ``measurements`` member is the authoritative long table.  Rows
    with ``pass=formal_kernel_sum`` are the primary CUPTI result.  Independent
    ``cuda_event_total`` rows report realized whole-pipeline device latency;
    Kineto device-span and per-stage rows are diagnostic evidence.
    """

    protocol.validate()
    ordered = validate_graph(graph)
    graph_hash = graph_fingerprint(graph)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal benchmark protocol")
    selected_stream = stream if stream is not None else torch.cuda.current_stream()
    if lifecycle is not None:
        if lifecycle.case != case:
            raise ValueError("benchmark lifecycle belongs to a different case")
        if lifecycle.candidate_graph is not graph:
            raise ValueError("benchmark graph is not the lifecycle candidate graph")
        if stream is not None and stream is not torch.cuda.current_stream():
            raise ValueError("v2 lifecycle benchmark requires the current CUDA stream")

    # Finish any asynchronous plugin preparation before warmup.  Warmup then
    # absorbs lazy CUDA module loading, allocator setup, and autotuning.
    selected_stream.synchronize()
    if lifecycle is None:
        warmup_prepared_graph(
            graph,
            case,
            iterations=protocol.warmup_iterations,
            stream=selected_stream,
        )
    else:
        lifecycle.warmup_candidate(protocol.warmup_iterations)

    # Correctness cases exercise exactness against the oracle.  Every larger
    # timed shape additionally passes the structural/causal output contract
    # once outside the measurement window, catching size-dependent fallbacks.
    if lifecycle is None:
        with torch.inference_mode(), torch.cuda.stream(selected_stream):
            validation_artifacts = execute_graph(
                graph,
                ExecutionContext(
                    case=case,
                    mode=RunMode.BENCHMARK,
                    iteration=-1,
                    stream=selected_stream,
                ),
            )
        selected_stream.synchronize()
        large_case_contract_check = validate_topk_indices(
            validation_artifacts[graph.terminal_artifact],
            case,
            name="benchmark output",
        )
        exact_reference_check = (
            compare_topk_indices(
                validation_artifacts[graph.terminal_artifact],
                reference_indices,
                case,
            )
            if reference_indices is not None
            else {"status": "not_evaluated"}
        )
        del validation_artifacts
        preflight_gate = None
    else:
        preflight_gate = lifecycle.check_candidate(
            phase="benchmark_preflight",
            order=FIXTURE_IDS,
            start_iteration=-2,
        )
        large_case_contract_check = {
            "status": "passed",
            "fixtures": {
                key: value["contract_check"]
                for key, value in preflight_gate["checks"].items()
            },
        }
        exact_reference_check = {
            "status": "passed",
            "fixtures": {
                key: value["exact_reference_check"]
                for key, value in preflight_gate["checks"].items()
            },
        }

    initial_inputs = graph.initial_artifacts.get("inputs")
    device = getattr(getattr(initial_inputs, "q", None), "device", None) or "cuda"
    measurements: list[dict[str, Any]] = []
    terminal_description: dict[str, Any] | None = None
    graph_semantics = tuple(
        sorted({semantic for node in ordered for semantic in node.spec.semantic_ops})
    )
    profiled_nodes = tuple(node for node in ordered if node.spec.profile)

    def bound_artifacts(fixture_id: str) -> ArtifactStore:
        return (
            lifecycle.bound_artifacts(graph, fixture_id)
            if lifecycle is not None
            else dict(graph.initial_artifacts)
        )

    # Measure operator transient memory in an unprofiled, unscrubbed execution.
    # The 8 GB benchmark harness is deliberately reported separately.
    selected_stream.synchronize()
    operator_memory_before = _allocator_state(torch)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.cuda.stream(selected_stream):
        memory_artifacts = bound_artifacts("A")
        _run_nodes(
            ordered,
            ExecutionContext(
                case=case,
                mode=RunMode.BENCHMARK,
                iteration=-3,
                stream=selected_stream,
            ),
            memory_artifacts,
        )
    selected_stream.synchronize()
    if graph.terminal_artifact not in memory_artifacts:
        raise RuntimeError("memory pass did not produce its terminal artifact")
    terminal_description = _terminal_description(memory_artifacts[graph.terminal_artifact])
    del memory_artifacts
    operator_memory = _memory_result(
        torch,
        operator_memory_before,
        interpretation="operator-only pass; excludes L2 flush and profiler",
    )

    # Secondary result: an unprofiled whole-pipeline CUDA Event pass.  The
    # DeepGEMM 8 GB memset is enqueued before the start event, so it both flushes
    # L2 and gives the host time to submit the graph without entering elapsed.
    selected_stream.synchronize()
    event_memory_before = _allocator_state(torch)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.cuda.stream(selected_stream):
        for trial_id in range(protocol.event_trials):
            fixture_id = FIXTURE_IDS[trial_id % len(FIXTURE_IDS)] if lifecycle else "A"
            artifacts = bound_artifacts(fixture_id)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            context = ExecutionContext(
                case=case,
                mode=RunMode.BENCHMARK,
                iteration=trial_id,
                stream=selected_stream,
            )
            _deepgemm_l2_flush(
                torch,
                bytes_count=protocol.l2_flush_bytes,
                device=device,
            )
            start.record(selected_stream)
            _run_nodes(ordered, context, artifacts)
            end.record(selected_stream)
            end.synchronize()
            if graph.terminal_artifact not in artifacts:
                raise RuntimeError("timed graph did not produce its terminal artifact")
            terminal_description = _terminal_description(artifacts[graph.terminal_artifact])
            measurements.append(
                _measurement(
                    graph=graph,
                    case=case,
                    pass_name="cuda_event_total",
                    trial_id=trial_id,
                    scope_id="operator_total",
                    stage_id="operator_total",
                    semantic_ops=graph_semantics,
                    latency_ms=start.elapsed_time(end),
                    timing_source="direct_cuda_event",
                    run_id=run_id,
                    graph_hash=graph_hash,
                    fixture_id=fixture_id,
                )
            )
            del artifacts, start, end
            if protocol.cooldown_seconds:
                time.sleep(protocol.cooldown_seconds)
    selected_stream.synchronize()
    event_harness_memory = _memory_result(
        torch,
        event_memory_before,
        interpretation="whole-pipeline Event pass; includes 8 GB flush allocation in peak",
    )

    # Primary result: the exact DeepGEMM profiler schedule plus CPU ranges used
    # only to correlate every CUDA activity belonging to a multi-kernel graph.
    from torch.profiler import ProfilerActivity, profile, record_function, schedule

    with torch.inference_mode(), torch.cuda.stream(selected_stream):
        init_artifacts = bound_artifacts("A")
        _run_nodes(
            ordered,
            ExecutionContext(
                case=case,
                mode=RunMode.BENCHMARK,
                iteration=protocol.event_trials,
                stream=selected_stream,
            ),
            init_artifacts,
        )
        del init_artifacts
    selected_stream.synchronize()

    kineto_memory_before = _allocator_state(torch)
    torch.cuda.reset_peak_memory_stats()
    profiler_schedule = schedule(wait=1, warmup=0, active=1, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=profiler_schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with torch.inference_mode(), torch.cuda.stream(selected_stream):
            for phase in range(2):
                for trial_id in range(protocol.kineto_trials):
                    fixture_id = (
                        FIXTURE_IDS[trial_id % len(FIXTURE_IDS)] if lifecycle else "A"
                    )
                    artifacts = bound_artifacts(fixture_id)
                    _deepgemm_l2_flush(
                        torch,
                        bytes_count=protocol.l2_flush_bytes,
                        device=device,
                    )
                    context = ExecutionContext(
                        case=case,
                        mode=RunMode.BENCHMARK,
                        iteration=(
                            protocol.event_trials
                            + 1
                            + phase * protocol.kineto_trials
                            + trial_id
                        ),
                        stream=selected_stream,
                    )
                    with record_function(operator_range_name(phase, trial_id)):
                        for node in ordered:
                            if node.spec.profile:
                                with record_function(
                                    stage_range_name(
                                        phase,
                                        trial_id,
                                        node.spec.stage_id,
                                    )
                                ):
                                    node.run(context, artifacts)
                            else:
                                node.run(context, artifacts)
                    if graph.terminal_artifact not in artifacts:
                        raise RuntimeError(
                            "Kineto graph did not produce its terminal artifact"
                        )
                    del artifacts
                profiler.step()
    selected_stream.synchronize()
    kineto_harness_memory = _memory_result(
        torch,
        kineto_memory_before,
        interpretation="Kineto pass; includes 8 GB flush and profiler allocations in peak",
    )
    kineto_trials = extract_active_trials(
        profiler.events(),
        trials=protocol.kineto_trials,
        stage_ids=tuple(node.spec.stage_id for node in profiled_nodes),
    )
    topology_gate = validate_trial_topology(
        kineto_trials,
        stage_ids=tuple(node.spec.stage_id for node in profiled_nodes),
    )
    for item in kineto_trials:
        trial_id = int(item["trial_id"])
        fixture_id = FIXTURE_IDS[trial_id % len(FIXTURE_IDS)] if lifecycle else "A"
        operator = item["operator"]
        for pass_name, field in (
            ("formal_kernel_sum", "kernel_sum_us"),
            ("kineto_activity_sum", "activity_sum_us"),
            ("kineto_device_span", "device_span_us"),
        ):
            row = _measurement(
                graph=graph,
                case=case,
                pass_name=pass_name,
                trial_id=trial_id,
                scope_id="operator_total",
                stage_id="operator_total",
                semantic_ops=graph_semantics,
                latency_ms=float(operator[field]) / 1_000.0,
                timing_source="kineto_cupti",
                run_id=run_id,
                graph_hash=graph_hash,
                fixture_id=fixture_id,
            )
            row.update(
                {
                    "kernel_count": int(operator["kernel_count"]),
                    "activity_count": int(operator["activity_count"]),
                    "inter_activity_gap_us": float(operator["gap_us"]),
                }
            )
            measurements.append(row)
        for node in profiled_nodes:
            stage = item["stages"][node.spec.stage_id]
            row = _measurement(
                graph=graph,
                case=case,
                pass_name="kineto_stage_kernel_sum",
                trial_id=trial_id,
                scope_id="stage",
                stage_id=node.spec.stage_id,
                semantic_ops=node.spec.semantic_ops,
                latency_ms=float(stage["kernel_sum_us"]) / 1_000.0,
                timing_source="kineto_cupti",
                run_id=run_id,
                graph_hash=graph_hash,
                fixture_id=fixture_id,
            )
            row.update(
                {
                    "kernel_count": int(stage["kernel_count"]),
                    "activity_count": int(stage["activity_count"]),
                    "inter_activity_gap_us": float(stage["gap_us"]),
                }
            )
            measurements.append(row)

    selected_stream.synchronize()
    if lifecycle is None:
        with torch.inference_mode(), torch.cuda.stream(selected_stream):
            postflight_artifacts = execute_graph(
                graph,
                ExecutionContext(
                    case=case,
                    mode=RunMode.BENCHMARK,
                    iteration=protocol.event_trials + 1 + 2 * protocol.kineto_trials,
                    stream=selected_stream,
                ),
            )
        selected_stream.synchronize()
        postflight_contract_check = validate_topk_indices(
            postflight_artifacts[graph.terminal_artifact],
            case,
            name="benchmark postflight output",
        )
        postflight_exact_reference_check = (
            compare_topk_indices(
                postflight_artifacts[graph.terminal_artifact],
                reference_indices,
                case,
            )
            if reference_indices is not None
            else {"status": "not_evaluated"}
        )
        del postflight_artifacts
        postflight_gate = None
    else:
        postflight_gate = lifecycle.check_candidate(
            phase="benchmark_postflight",
            order=tuple(reversed(FIXTURE_IDS)),
            start_iteration=protocol.event_trials + 1 + 2 * protocol.kineto_trials,
        )
        postflight_contract_check = {
            "status": "passed",
            "fixtures": {
                key: value["contract_check"]
                for key, value in postflight_gate["checks"].items()
            },
        }
        postflight_exact_reference_check = {
            "status": "passed",
            "fixtures": {
                key: value["exact_reference_check"]
                for key, value in postflight_gate["checks"].items()
            },
        }
    memory = {
        "operator": operator_memory,
        "cuda_event_harness": event_harness_memory,
        "kineto_harness": kineto_harness_memory,
        "l2_flush_bytes": protocol.l2_flush_bytes,
        "reference_indices_bytes": (
            sum(
                int(item.indices.numel()) * int(item.indices.element_size())
                for item in lifecycle.oracles.values()
            )
            if lifecycle is not None
            else (
                int(reference_indices.numel()) * int(reference_indices.element_size())
                if reference_indices is not None
                else 0
            )
        ),
        "interpretation": (
            "use memory.operator for operator workspace; timing harness peaks include "
            "the DeepGEMM-compatible 8 GB flush"
        ),
    }
    manifest = input_manifest(initial_inputs) if isinstance(initial_inputs, PrefillInputs) else None
    input_fingerprint = (
        lifecycle.input_content_hash
        if lifecycle is not None
        else (canonical_hash(manifest) if manifest is not None else None)
    )
    return {
        "schema_version": 2 if lifecycle is not None else 1,
        "run_id": run_id,
        "variant": asdict(graph.descriptor),
        "case": asdict(case),
        "graph": graph_mapping(graph),
        "graph_fingerprint": graph_hash,
        "protocol": asdict(protocol),
        "input_manifest": manifest,
        "input_fixtures": lifecycle.fixture_metadata() if lifecycle is not None else None,
        "input_fingerprint": input_fingerprint,
        "input_content_hash": input_fingerprint if lifecycle is not None else None,
        "lifecycle": lifecycle.metadata() if lifecycle is not None else None,
        "preflight_gate": preflight_gate,
        "postflight_gate": postflight_gate,
        "terminal_artifact": terminal_description,
        "large_case_contract_check": large_case_contract_check,
        "exact_reference_check": exact_reference_check,
        "postflight_contract_check": postflight_contract_check,
        "postflight_exact_reference_check": postflight_exact_reference_check,
        "reference_metadata": dict(reference_metadata or {}),
        "kineto": {
            "collector": "torch.profiler/Kineto/CUPTI",
            "deepgemm_source": "deep_gemm/testing/bench.py::bench_kineto",
            "schedule": {"wait": 1, "warmup": 0, "active": 1, "repeat": 1},
            "tests_per_step": protocol.kineto_trials,
            "l2_flush_bytes": protocol.l2_flush_bytes,
            "activities": ["CPU", "CUDA"],
            "cpu_activity_reason": "correlate all kernels in a multi-kernel operator",
            "topology_gate": topology_gate,
            "raw_trials": kineto_trials,
        },
        "memory": memory,
        "measurements": measurements,
        "summary": summarize_measurements(measurements),
    }


def benchmark_variant(
    plugin: VariantPlugin,
    case: PrefillCase,
    inputs: PrefillInputs,
    *,
    options: Mapping[str, Any] | None = None,
    protocol: BenchmarkProtocol | None = None,
    stream: Any = None,
    run_id: str | None = None,
    reference_indices: Any | None = None,
    reference_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prepare, warm, and benchmark one plugin/case combination."""

    selected_protocol = protocol or BenchmarkProtocol()
    graph = prepare_variant(
        plugin,
        case,
        inputs,
        options=options,
        mode=RunMode.BENCHMARK,
    )
    return benchmark_prepared_graph(
        graph,
        case,
        protocol=selected_protocol,
        stream=stream,
        run_id=run_id,
        reference_indices=reference_indices,
        reference_metadata=reference_metadata,
    )
