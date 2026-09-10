"""Profiler target and report-sidecar finalization for IndexTopK-PerfLab."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .api import RunMode
from .graph import graph_fingerprint, graph_mapping
from .nvtx import graph_labels, ncu_push_pop_filter, nvtx_range, stage_range_factory


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing JSON artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _tensor_description(value: Any) -> dict[str, Any]:
    return {
        "shape": [int(item) for item in value.shape],
        "stride": [int(item) for item in value.stride()],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "contiguous": bool(value.is_contiguous()),
        "logical_bytes": int(value.numel()) * int(value.element_size()),
    }


def _output_manifest(artifacts: Mapping[str, Any], terminal_artifact: str) -> dict[str, Any]:
    value = artifacts.get(terminal_artifact)
    if value is None:
        raise RuntimeError(f"profiled graph did not produce {terminal_artifact!r}")
    if not all(hasattr(value, name) for name in ("shape", "stride", "numel")):
        return {terminal_artifact: {"python_type": type(value).__qualname__}}
    return {terminal_artifact: _tensor_description(value)}


def _profile_command_from_environment() -> list[str] | None:
    encoded = os.environ.get("ITK_PROFILER_COMMAND_JSON")
    if not encoded:
        return None
    try:
        command = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError("ITK_PROFILER_COMMAND_JSON is not valid JSON") from error
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("ITK_PROFILER_COMMAND_JSON must encode a list of strings")
    return command


def _capture(args: argparse.Namespace) -> None:
    # Lazy imports keep ``python -m ...profile --help`` CPU-only.
    import torch

    from .config import load_config
    from .contract import (
        plan_identity,
        profiling_plan_identity,
        protocol_identity,
        resolved_problem_contract,
    )
    from .lifecycle import FIXTURE_IDS, CaseLifecycle
    from .provenance import experiment_identity, variant_identity
    from .registry import load_variant
    from .runner import verify_correctness_artifact
    from .runtime import collect_runtime, runtime_identity, validate_runtime

    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.target_length not in config.profile_lengths(args.mode):
        raise ValueError(
            f"target length {args.target_length} is not configured for {args.mode}; "
            f"allowed={list(config.profile_lengths(args.mode))}"
        )
    case = config.make_case(args.target_length, purpose="profile")
    input_factory = None
    input_source: dict[str, Any] = {"kind": "synthetic_config_recipe"}
    if args.replay_manifest is not None:
        if args.replay_split is None:
            raise ValueError("--replay-split is required with --replay-manifest")
        from .replay import ReplayInputFactory, sha256_file

        replay_manifest = args.replay_manifest.resolve()
        replay_factory = ReplayInputFactory(
            replay_manifest,
            split=args.replay_split,
            verify_sha256=True,
        )
        replay_cases = [
            replay_case
            for replay_case in replay_factory.cases(seed=args.replay_seed)
            if (
                replay_case.query_tokens == case.query_tokens
                and replay_case.context_tokens == case.context_tokens
                and replay_case.top_k == case.top_k
                and replay_case.batch_size == case.batch_size
                and replay_case.indexer_heads == case.indexer_heads
                and replay_case.head_dim == case.head_dim
                and replay_case.causal == case.causal
            )
        ]
        if len(replay_cases) != 1:
            raise ValueError(
                "replay profile requires exactly one shape-compatible case; "
                f"found={len(replay_cases)} split={args.replay_split!r} "
                f"Q={case.query_tokens} N={case.context_tokens}"
            )
        replay_case = replay_cases[0]
        # Keep the config-derived case ID so the existing stable NVTX label and
        # wrapper-side filter remain unchanged.  The replay benchmark seed is
        # retained because it is part of the deterministic sampling path.
        case = replace(case, seed=replay_case.seed)

        def load_replay_inputs(requested_case: Any, device: Any, fixture_id: str) -> Any:
            if requested_case != case:
                raise ValueError("replay profiler requested an unexpected case")
            loaded = replay_factory(replay_case, device, fixture_id)
            return replace(loaded, case=requested_case)

        input_factory = load_replay_inputs
        input_source = {
            "kind": "frozen_replay",
            "manifest": str(replay_manifest),
            "manifest_sha256": sha256_file(replay_manifest),
            "split": args.replay_split,
            "source_case": asdict(replay_case),
            "profile_case_id": case.case_id,
            "profile_seed": case.seed,
        }
    elif args.replay_split is not None:
        raise ValueError("--replay-split requires --replay-manifest")
    variant_options = config.variant_options(args.variant)
    plugin = load_variant(
        config.variant_factory(args.variant),
        options=variant_options,
    )
    if plugin.descriptor.plugin_id != args.variant:
        raise ValueError(
            f"configured variant ID {args.variant!r} does not match plugin ID "
            f"{plugin.descriptor.plugin_id!r}"
        )
    if "sm90" not in plugin.descriptor.supported_arches:
        raise ValueError(f"variant {args.variant!r} does not declare SM90 support")
    identity = variant_identity(
        config,
        args.variant,
        plugin,
        options=variant_options,
    )
    if not plugin.supports(case):
        raise ValueError(f"variant {plugin.descriptor.plugin_id!r} does not support {case.case_id}")
    if args.mode == "nsys" and args.stage != "pipeline":
        raise ValueError("Nsight Systems captures the complete pipeline; use --stage pipeline")
    if args.mode == "ncu" and args.stage == "pipeline":
        raise ValueError("Nsight Compute requires one physical --stage")

    config_sha256 = _sha256(config_path)
    correctness = verify_correctness_artifact(
        args.correctness,
        config_path=config_path,
        plugin_id=plugin.descriptor.plugin_id,
        variant_fingerprint=identity["fingerprint"],
        config=config,
    )
    runtime = collect_runtime()
    validate_runtime(runtime, config)
    current_runtime_identity = runtime_identity(runtime)
    if correctness["runtime_identity_sha256"] != current_runtime_identity["sha256"]:
        raise ValueError(
            "correctness artifact was produced under a different runtime identity"
        )
    mode = RunMode(args.mode)
    reference_options = config.variant_options(config.exact_reference_variant)
    reference = load_variant(
        config.variant_factory(config.exact_reference_variant),
        options=reference_options,
    )
    if reference.descriptor.plugin_id != config.exact_reference_variant:
        raise ValueError(
            "exact reference configured ID does not match its descriptor plugin_id"
        )
    reference_identity = variant_identity(
        config,
        config.exact_reference_variant,
        reference,
        options=reference_options,
    )
    lifecycle = CaseLifecycle(
        case=case,
        workload=config.workload,
        reference=reference,
        reference_options=reference_options,
        mode=mode,
        input_factory=input_factory,
    )
    oracle = lifecycle.build_oracles()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocator_after_reference_cleanup = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }

    graph = lifecycle.prepare_candidate(plugin, variant_options)
    ordered = lifecycle.candidate_nodes

    labels = graph_labels(graph, case)
    nodes = {node.spec.stage_id: node for node in ordered}
    if args.mode == "ncu":
        node = nodes.get(args.stage)
        if node is None:
            raise ValueError(
                f"stage {args.stage!r} is absent; available={sorted(nodes)}"
            )
        if not node.spec.profile:
            raise ValueError(f"stage {args.stage!r} is not marked profileable")
        selected_label = labels.stages[args.stage]
    else:
        selected_label = labels.pipeline

    expected_label = os.environ.get("ITK_EXPECTED_NVTX_LABEL")
    if expected_label is not None and expected_label != selected_label:
        raise ValueError(
            "wrapper/target NVTX label mismatch: "
            f"expected {expected_label!r}, target generated {selected_label!r}"
        )

    warmup_iterations = int(config.timing.warmup_iterations)
    captured_iterations = int(config.profiling.captured_iterations)
    if captured_iterations != 1:
        raise ValueError("profiling protocol requires captured_iterations=1")
    # prepare() above, and these complete graph iterations, force JIT/autotune
    # and allocator growth to finish before cudaProfilerStart.
    lifecycle.warmup_candidate(warmup_iterations)
    preflight_gate = lifecycle.check_candidate(
        phase="profile_preflight",
        order=FIXTURE_IDS,
        start_iteration=-2,
    )

    l2_flush_bytes = int(config.timing.l2_flush_bytes)
    if l2_flush_bytes > 0:
        torch.empty(
            l2_flush_bytes // 4,
            dtype=torch.int,
            device="cuda",
        ).zero_()
        torch.cuda.synchronize()

    allocator_before_capture = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }
    torch.cuda.reset_peak_memory_stats()
    outputs: Mapping[str, Any] = {}
    profiler_started = False
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    profiler_started = True
    try:
        with torch.inference_mode(), nvtx_range(torch, labels.pipeline):
            outputs = lifecycle.execute(
                graph,
                ordered,
                fixture_id="A",
                iteration=0,
                stage_context=stage_range_factory(torch, labels),
            )
        torch.cuda.synchronize()
    finally:
        if profiler_started:
            torch.cuda.profiler.stop()

    captured_check = lifecycle.validate_outputs(
        outputs,
        fixture_id="A",
        name="captured profile output",
    )
    postflight_gate = lifecycle.check_candidate(
        phase="profile_postflight",
        order=tuple(reversed(FIXTURE_IDS)),
        start_iteration=1,
    )

    memory = {
        "allocated_end_bytes": int(torch.cuda.memory_allocated()),
        "reserved_end_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes_during_capture": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes_during_capture": int(torch.cuda.max_memory_reserved()),
        "peak_allocated_delta_from_pre_capture_bytes": int(
            torch.cuda.max_memory_allocated() - allocator_before_capture["allocated_bytes"]
        ),
    }
    logits_bytes = (
        int(case.batch_size)
        * int(case.query_tokens)
        * int(case.context_tokens)
        * 4
    )
    profile_tool = "nsys" if args.mode == "nsys" else f"ncu:{args.stage}"
    runtime_id = current_runtime_identity
    protocol_id = protocol_identity(config)
    formal_plan_id = plan_identity(config)
    plan_id = profiling_plan_identity(
        config,
        mode=args.mode,
        target_length=args.target_length,
        stage=args.stage,
    )
    experiment_id = experiment_identity(
        protocol_hash=protocol_id["sha256"],
        plan_hash=plan_id["sha256"],
        operator_hash=identity["payload"]["operator_fingerprint"],
        runtime_hash=runtime_id["sha256"],
        input_content_hash=lifecycle.input_content_hash,
    )
    metadata = {
        "schema_version": 2,
        "artifact_type": "indextopk_nsight_profile",
        "timestamp": _utc_timestamp(),
        "status": "target_capture_complete",
        "formal_timing": False,
        "profile_tool": profile_tool,
        "run_id": args.run_id,
        "command": [sys.executable, *sys.argv],
        "profiler_command": _profile_command_from_environment(),
        "config": {
            "path": str(args.config),
            "sha256": config_sha256,
        },
        "problem_contract": resolved_problem_contract(config),
        "correctness": correctness,
        "variant": {
            **asdict(graph.descriptor),
            "requested_id": args.variant,
            "factory": config.variant_factory(args.variant),
            "options": dict(variant_options),
            "identity": identity,
            "graph_fingerprint": graph_fingerprint(graph),
        },
        "experiment_identity": experiment_id,
        "protocol_identity": protocol_id,
        "plan_identity": plan_id,
        "formal_plan_identity": formal_plan_id,
        "runtime_identity": runtime_id,
        "graph": graph_mapping(graph),
        "workload": asdict(case),
        "input_source": input_source,
        "input_fixtures": lifecycle.fixture_metadata(),
        "input_fingerprint": lifecycle.input_content_hash,
        "input_content_hash": lifecycle.input_content_hash,
        "lifecycle": lifecycle.metadata(),
        "output_manifest": _output_manifest(outputs, graph.terminal_artifact),
        "large_case_contract_check": preflight_gate,
        "preflight_exact_reference_check": preflight_gate,
        "captured_contract_check": captured_check["contract_check"],
        "captured_exact_reference_check": captured_check["exact_reference_check"],
        "postflight_gate": postflight_gate,
        "input_versions_unchanged": True,
        "exact_reference": {
            "variant": asdict(reference.descriptor),
            "identity": reference_identity,
            "graph_fingerprint": oracle["graph_fingerprint"],
            "fixtures": oracle["fixtures"],
            "allocator_cache_cleared_before_candidate_prepare": True,
        },
        "runtime": runtime,
        "profile": {
            "mode": args.mode,
            "stage": args.stage,
            "target_length": args.target_length,
            "captured_iterations": captured_iterations,
            "warmup_iterations": warmup_iterations,
            "l2_flush_bytes": l2_flush_bytes,
            "selected_nvtx_label": selected_label,
            "ncu_nvtx_filter": (
                ncu_push_pop_filter(selected_label) if args.mode == "ncu" else None
            ),
            "nvtx_ranges": labels.as_mapping(),
        },
        "capture_protocol": {
            "capture_gate": "cudaProfilerApi",
            "complete_graph_inside_capture": True,
            "exactly_one_graph_iteration": True,
            "jit_autotune_and_warmup_outside_capture": True,
            "one_stream": True,
            "stage_synchronizations": 0,
            "synchronize_before_profiler_start": True,
            "synchronize_before_profiler_stop": True,
            "ncu_replay_mode": "application" if args.mode == "ncu" else None,
            "ncu_app_replay_mode": "strict" if args.mode == "ncu" else None,
            "diagnostic_only": True,
            "formal_latency_source": "separate DeepGEMM-style Kineto/CUPTI benchmark",
        },
        "allocator_before_capture": allocator_before_capture,
        "allocator_after_reference_cleanup": allocator_after_reference_cleanup,
        "memory": memory,
        "logical_cost_reference": {
            "materialized_fp32_logits_bytes": logits_bytes,
            "minimum_write_plus_one_read_bytes": 2 * logits_bytes,
        },
        "ncu_requested_metrics": (
            list(config.profiling.ncu_metrics) if args.mode == "ncu" else []
        ),
        "reports": {},
    }
    _atomic_write_json(args.metadata_output, metadata)
    print(
        "profile target complete "
        f"variant={graph.descriptor.plugin_id} case={case.case_id} "
        f"tool={profile_tool} metadata={args.metadata_output}"
    )


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty {role}: {path}")
    return {
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _finalize(args: argparse.Namespace) -> None:
    metadata = _read_json(args.metadata)
    if metadata.get("artifact_type") != "indextopk_nsight_profile":
        raise ValueError(f"not an IndexTopK-PerfLab profile sidecar: {args.metadata}")
    if metadata.get("status") not in {"target_capture_complete", "verified"}:
        raise ValueError(f"profile sidecar has invalid status: {metadata.get('status')!r}")
    profile = metadata.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("profile sidecar has no structured profile contract")
    evidence = _read_json(args.verification_json)
    if evidence.get("status") != "passed" or evidence.get("schema_version") != 1:
        raise ValueError("profile verification evidence is not passed schema version 1")
    expected_tool = profile.get("mode")
    expected_stage = profile.get("stage")
    expected_label = profile.get("selected_nvtx_label")
    if evidence.get("tool") != expected_tool:
        raise ValueError("verification evidence tool does not match captured profile")
    if evidence.get("stage") != expected_stage:
        raise ValueError("verification evidence stage does not match captured profile")
    if evidence.get("expected_nvtx_label") != expected_label:
        raise ValueError("verification evidence NVTX label does not match captured profile")
    source = evidence.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("verification evidence has no source file record")
    evidence_source = Path(str(source.get("path", "")))
    actual_source = _file_record(evidence_source, role="verification_source")
    if (
        source.get("bytes") != actual_source["bytes"]
        or source.get("sha256") != actual_source["sha256"]
    ):
        raise ValueError("verification source changed after it was validated")
    if expected_tool == "ncu":
        requested = set(metadata.get("ncu_requested_metrics", []))
        counts = evidence.get("requested_metric_finite_value_counts")
        if not isinstance(counts, Mapping) or set(counts) != requested:
            raise ValueError("NCU evidence does not cover exactly the requested metric set")
        if any(not isinstance(value, int) or value <= 0 for value in counts.values()):
            raise ValueError("NCU evidence contains a requested metric with no finite values")
        if not str(args.native_report).endswith(".ncu-rep") or not str(args.export).endswith(
            ".csv"
        ):
            raise ValueError("NCU report/export extensions do not match the expected tool")
    elif expected_tool == "nsys":
        if not str(args.native_report).endswith(".nsys-rep") or not str(args.export).endswith(
            ".sqlite"
        ):
            raise ValueError("Nsys report/export extensions do not match the expected tool")
    else:
        raise ValueError(f"unknown captured profile mode: {expected_tool!r}")
    reports = {
        "native": _file_record(args.native_report, role="native_report"),
        "machine_readable": _file_record(args.export, role="machine_readable_export"),
        "verification_evidence": _file_record(
            args.verification_json, role="verification_evidence"
        ),
        "verification_source": actual_source,
    }
    if metadata.get("status") == "verified" and metadata.get("reports") != reports:
        raise ValueError("refusing to reseal profile metadata with different report files")
    metadata["reports"] = reports
    metadata["verification"] = {
        "status": "passed",
        "evidence": evidence,
        "verified_at": _utc_timestamp(),
    }
    metadata["status"] = "verified"
    _atomic_write_json(args.metadata, metadata)
    print(f"profile verified metadata={args.metadata}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="run one profiler target capture")
    capture.add_argument(
        "--config", type=Path, default=Path("configs/fused_index_topk_h20.json")
    )
    capture.add_argument("--variant", required=True)
    capture.add_argument("--target-length", type=int, required=True)
    capture.add_argument("--mode", choices=("nsys", "ncu"), required=True)
    capture.add_argument("--stage", default="pipeline")
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--correctness", type=Path, required=True)
    capture.add_argument("--metadata-output", type=Path, required=True)
    capture.add_argument("--replay-manifest", type=Path)
    capture.add_argument(
        "--replay-split",
        choices=("tuning", "test_normal", "test_hard"),
    )
    capture.add_argument("--replay-seed", type=int, default=20260825)
    capture.set_defaults(handler=_capture)

    finalize = subparsers.add_parser("finalize", help="verify reports and seal metadata")
    finalize.add_argument("--metadata", type=Path, required=True)
    finalize.add_argument("--native-report", type=Path, required=True)
    finalize.add_argument("--export", type=Path, required=True)
    finalize.add_argument("--verification-json", type=Path, required=True)
    finalize.set_defaults(handler=_finalize)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
