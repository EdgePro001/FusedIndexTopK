"""High-level correctness lifecycle shared by all variant implementations."""

from __future__ import annotations

import gc
import hashlib
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .api import RunMode
from .artifacts import canonical_hash, load_json, write_json_atomic
from .benchmark import BenchmarkProtocol, benchmark_prepared_graph
from .contract import plan_identity, protocol_identity, resolved_problem_contract
from .graph import graph_fingerprint, graph_mapping
from .lifecycle import CaseLifecycle
from .provenance import experiment_identity, framework_fingerprint, variant_identity
from .registry import load_variant
from .runtime import (
    collect_gpu_state,
    collect_runtime,
    runtime_identity,
    validate_gpu_exclusivity,
    validate_runtime,
)
from .summary import summarize_measurements, write_measurements


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_run_id(prefix: str = "itk") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_configured_plugin(
    config: Any, variant_id: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    options = config.variant_options(variant_id)
    plugin = load_variant(config.variant_factory(variant_id), options=options)
    if plugin.descriptor.plugin_id != variant_id:
        raise ValueError(
            f"configured variant ID {variant_id!r} does not match descriptor plugin_id "
            f"{plugin.descriptor.plugin_id!r}"
        )
    if "sm90" not in plugin.descriptor.supported_arches:
        raise ValueError(f"variant {variant_id!r} does not declare SM90 support")
    return plugin, options, variant_identity(config, variant_id, plugin, options=options)


def run_correctness(
    config: Any,
    *,
    config_path: str | Path,
    variant_id: str,
    run_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the standard small-case gate against the frozen exact reference."""

    import torch

    config_path = Path(config_path).resolve()
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite correctness artifact: {destination}")
    candidate, candidate_options, candidate_identity = _load_configured_plugin(
        config, variant_id
    )
    reference, reference_options, reference_identity = _load_configured_plugin(
        config, config.exact_reference_variant
    )
    runtime: dict[str, Any] | None = None
    artifact: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "indextopk_correctness",
        "status": "running",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config_hash": canonical_hash(config.as_mapping()),
        "variant": {
            **asdict(candidate.descriptor),
            "requested_id": variant_id,
            "factory": config.variant_factory(variant_id),
            "options": candidate_options,
            "identity": candidate_identity,
        },
        "exact_reference": {
            **asdict(reference.descriptor),
            "requested_id": config.exact_reference_variant,
            "factory": config.variant_factory(config.exact_reference_variant),
            "options": reference_options,
            "identity": reference_identity,
        },
        "contract": {
            "version": "fusion-v1",
            "terminal_artifact": "indices",
            "shape": "[query_tokens,1,top_k]",
            "dtype": config.workload.output_dtype,
            "padding": config.workload.padding_index,
            "order": config.workload.output_order,
            "selection": f"{config.workload.selection} causal TopK",
            "tie_policy": config.workload.tie_policy,
        },
        "problem_contract": resolved_problem_contract(config),
        "cases": [],
    }
    try:
        runtime = collect_runtime()
        validate_runtime(runtime, config)
        artifact["runtime"] = runtime
        runtime_id = runtime_identity(runtime)
        protocol_id = protocol_identity(config)
        plan_id = plan_identity(config)
        artifact["runtime_identity"] = runtime_id
        artifact["protocol_identity"] = protocol_id
        artifact["plan_identity"] = plan_id

        with torch.no_grad():
            for case in config.correctness_cases():
                lifecycle = CaseLifecycle(
                    case=case,
                    workload=config.workload,
                    reference=reference,
                    reference_options=reference_options,
                    mode=RunMode.CORRECTNESS,
                )
                oracle = lifecycle.build_oracles()
                candidate_graph = lifecycle.prepare_candidate(candidate, candidate_options)
                gate = lifecycle.check_candidate(
                    phase="correctness",
                    order=("A", "B", "A"),
                )
                identity_components = experiment_identity(
                    protocol_hash=protocol_id["sha256"],
                    plan_hash=plan_id["sha256"],
                    operator_hash=candidate_identity["payload"]["operator_fingerprint"],
                    runtime_hash=runtime_id["sha256"],
                    input_content_hash=lifecycle.input_content_hash,
                )
                artifact["cases"].append(
                    {
                        "case": asdict(case),
                        "input_fixtures": lifecycle.fixture_metadata(),
                        "input_fingerprint": lifecycle.input_content_hash,
                        "input_content_hash": lifecycle.input_content_hash,
                        "experiment_identity": identity_components,
                        "lifecycle": lifecycle.metadata(),
                        "reference_graph_fingerprint": oracle["graph_fingerprint"],
                        "candidate_graph_fingerprint": graph_fingerprint(candidate_graph),
                        "candidate_graph": graph_mapping(candidate_graph),
                        "reference_check": oracle,
                        "candidate_check": gate,
                        "comparison": gate,
                        "repeated_check": gate,
                        "repeat_comparison": gate,
                        "input_versions_unchanged": True,
                    }
                )
                del lifecycle, candidate_graph
                gc.collect()
                torch.cuda.empty_cache()

        artifact["status"] = "passed"
        artifact["completed_at_utc"] = utc_now()
        write_json_atomic(destination, artifact)
        return artifact
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["completed_at_utc"] = utc_now()
        artifact["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        if runtime is not None:
            artifact["runtime"] = runtime
        write_json_atomic(destination, artifact)
        raise


def verify_correctness_artifact(
    path: str | Path,
    *,
    config_path: str | Path,
    plugin_id: str,
    variant_fingerprint: str,
    config: Any | None = None,
) -> dict[str, Any]:
    """Validate that a v2 correctness gate is complete and reusable.

    The returned runtime hash must still be compared with the runtime collected
    by the consumer.  Keeping that comparison at the call site ensures it is
    made after the current CUDA runtime has actually been initialized.
    """

    if config is None:
        from .config import load_config

        config = load_config(config_path)
    source = Path(path).resolve()
    payload = load_json(source)
    if not isinstance(payload, dict) or payload.get("artifact_type") != "indextopk_correctness":
        raise ValueError(f"not a FusedIndexTopK correctness artifact: {source}")
    if payload.get("status") != "passed":
        raise ValueError(f"correctness gate did not pass: {source}")
    if payload.get("schema_version") != 2:
        raise ValueError("formal v2 runs require a schema-version-2 correctness artifact")
    expected_config = file_sha256(config_path)
    recorded_config = payload.get("config_sha256")
    if not isinstance(recorded_config, str) or len(recorded_config) != 64:
        raise ValueError("correctness artifact has no valid source config hash")
    variant = payload.get("variant")
    if not isinstance(variant, dict) or variant.get("plugin_id") != plugin_id:
        recorded_plugin = variant.get("plugin_id") if isinstance(variant, dict) else None
        raise ValueError(
            f"correctness artifact belongs to {recorded_plugin!r}, not {plugin_id!r}"
        )
    recorded_identity = variant.get("identity") if isinstance(variant, dict) else None
    if (
        not isinstance(recorded_identity, dict)
        or recorded_identity.get("fingerprint") != variant_fingerprint
    ):
        raise ValueError("correctness artifact belongs to a different variant implementation")
    identity_payload = recorded_identity.get("payload")
    if (
        not isinstance(identity_payload, dict)
        or canonical_hash(identity_payload) != recorded_identity.get("fingerprint")
    ):
        raise ValueError("correctness artifact variant identity is internally inconsistent")

    identity_records: dict[str, dict[str, Any]] = {}
    current_identities = {
        "protocol_identity": protocol_identity(config),
        "plan_identity": plan_identity(config),
    }
    for name in ("protocol_identity", "plan_identity", "runtime_identity"):
        record = payload.get(name)
        record_payload = record.get("payload") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or not isinstance(record_payload, dict)
            or canonical_hash(record_payload) != record.get("sha256")
        ):
            raise ValueError(f"correctness artifact has an invalid {name}")
        if (
            name in current_identities
            and record.get("sha256") != current_identities[name]["sha256"]
        ):
            raise ValueError(f"correctness artifact {name} does not match the current config")
        identity_records[name] = record

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("correctness artifact contains no cases")
    expected_cases = {
        (case.query_tokens, case.context_tokens) for case in config.correctness_cases()
    }
    realized_cases: set[tuple[int, int]] = set()
    for item in cases:
        if not isinstance(item, dict) or not isinstance(item.get("case"), dict):
            raise ValueError("correctness artifact contains a malformed case")
        query_tokens = int(item["case"].get("query_tokens", -1))
        context_tokens = int(item["case"].get("context_tokens", -1))
        case_key = (query_tokens, context_tokens)
        case_label = f"Q={query_tokens}, N={context_tokens}"
        if case_key in realized_cases:
            raise ValueError(f"duplicate correctness case for {case_label}")
        realized_cases.add(case_key)

        content_hash = item.get("input_content_hash")
        lifecycle = item.get("lifecycle")
        fixtures = item.get("input_fixtures")
        if not content_hash or content_hash != item.get("input_fingerprint"):
            raise ValueError(f"correctness case {case_label} has invalid input identity")
        if not isinstance(lifecycle, dict) or not isinstance(fixtures, dict):
            raise ValueError("correctness artifact is missing v2 input/lifecycle evidence")
        fixture_hashes: dict[str, str] = {}
        for fixture_id in ("A", "B"):
            fixture = fixtures.get(fixture_id)
            content = fixture.get("content") if isinstance(fixture, dict) else None
            manifest = content.get("manifest") if isinstance(content, dict) else None
            digest = content.get("sha256") if isinstance(content, dict) else None
            if not isinstance(manifest, dict) or canonical_hash(manifest) != digest:
                raise ValueError(
                    f"correctness case {case_label} fixture {fixture_id} "
                    "has invalid content evidence"
                )
            fixture_hashes[fixture_id] = str(digest)
        if fixture_hashes["A"] == fixture_hashes["B"]:
            raise ValueError(f"correctness case {case_label} A/B inputs are identical")
        if canonical_hash(fixture_hashes) != content_hash:
            raise ValueError(f"correctness case {case_label} input hash is inconsistent")
        if lifecycle.get("input_content_hash") != content_hash:
            raise ValueError(f"correctness case {case_label} lifecycle input hash differs")

        gates = lifecycle.get("integrity_checks")
        phases = {
            gate.get("phase")
            for gate in gates or []
            if isinstance(gate, dict) and gate.get("status") == "passed"
        }
        if not {"before_candidate", "candidate_prepare", "correctness"}.issubset(phases):
            raise ValueError("correctness artifact has incomplete input integrity gates")
        anti_precompute = lifecycle.get("anti_precompute_gate")
        if not isinstance(anti_precompute, dict) or anti_precompute.get(
            "candidate_prepare_fixture"
        ) != "A" or anti_precompute.get("required_execution_fixtures") != [
            "A",
            "B",
        ] or anti_precompute.get("same_shape_different_content") is not True:
            raise ValueError("correctness artifact has an invalid anti-precompute contract")

        candidate_gate = item.get("candidate_check")
        checks = candidate_gate.get("checks") if isinstance(candidate_gate, dict) else None
        seen_fixtures: set[str] = set()
        if not isinstance(candidate_gate, dict) or candidate_gate.get("status") != "passed":
            raise ValueError(f"correctness case {case_label} candidate gate is not passed")
        for check in checks.values() if isinstance(checks, dict) else ():
            if not isinstance(check, dict):
                continue
            fixture_id = check.get("fixture_id")
            if fixture_id in {"A", "B"}:
                seen_fixtures.add(str(fixture_id))
            for name in ("contract_check", "exact_reference_check"):
                record = check.get(name)
                if not isinstance(record, dict) or record.get("status") != "passed":
                    raise ValueError(
                        f"correctness case {case_label} has no passed {name}"
                    )
        if seen_fixtures != {"A", "B"}:
            raise ValueError(f"correctness case {case_label} did not execute A and B")

        oracle = item.get("reference_check")
        if not isinstance(oracle, dict) or oracle.get("status") != "passed":
            raise ValueError(f"correctness case {case_label} exact reference oracle failed")
        oracle_fixtures = oracle.get("fixtures")
        if not isinstance(oracle_fixtures, dict) or set(oracle_fixtures) != {"A", "B"}:
            raise ValueError(f"correctness case {case_label} oracle is incomplete")
        for fixture in oracle_fixtures.values():
            if not isinstance(fixture, dict) or any(
                not isinstance(fixture.get(name), dict)
                or fixture[name].get("status") != "passed"
                for name in ("contract_check", "cutoff_tie_check")
            ):
                raise ValueError(f"correctness case {case_label} oracle gate failed")

        experiment = item.get("experiment_identity")
        components = experiment.get("components") if isinstance(experiment, dict) else None
        expected_components = {
            "ProtocolHash": identity_records["protocol_identity"]["sha256"],
            "PlanHash": identity_records["plan_identity"]["sha256"],
            "OperatorHash": identity_payload.get("operator_fingerprint"),
            "RuntimeHash": identity_records["runtime_identity"]["sha256"],
            "InputContentHash": content_hash,
        }
        if (
            components != expected_components
            or not isinstance(experiment, dict)
            or canonical_hash(components) != experiment.get("sha256")
        ):
            raise ValueError(f"correctness case {case_label} identity is invalid")
    if realized_cases != expected_cases:
        raise ValueError("correctness case matrix does not match the current config")
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "status": "passed",
        "run_id": payload.get("run_id"),
        "config_sha256": recorded_config,
        "current_config_sha256": expected_config,
        "config_file_match": recorded_config == expected_config,
        "plugin_id": plugin_id,
        "variant_fingerprint": variant_fingerprint,
        "runtime_identity_sha256": identity_records["runtime_identity"]["sha256"],
    }


def _measurement_contract(config: Any) -> dict[str, Any]:
    return {
        "problem_contract": resolved_problem_contract(config),
        "target": asdict(config.target),
        "sources": asdict(config.sources),
        "workload": asdict(config.workload),
        "seed": config.seed,
        "input_recipe": {
            "version": "contiguous-prefill-fp8-v2",
            "fixture_ids": ["A", "B"],
            "generation_context_tokens": "case.context_tokens",
            "generation_order": "independent_rng_streams",
            "content_hash": "sha256-logical-tensor-bytes",
            "kv_quantization": "row_amax_clamp_1e-4_div_448_fp8_e4m3fn",
        },
        "core_fingerprint": framework_fingerprint(),
        "protocol_identity": protocol_identity(config),
        "plan_identity": plan_identity(config),
        "timing": asdict(config.timing),
        "formal_scope": "operator_total",
        "formal_pass": "formal_kernel_sum",
        "formal_timing_source": "kineto_cupti",
        "supplemental_pass": "cuda_event_total",
        "stream_policy": "single current CUDA stream",
        "compile_excluded": True,
        "allocation_policy": "persistent workspace excluded; per-iteration work included",
        "anti_precompute_gate": "prepare on A; preflight and postflight exactness on A/B",
        "formal_fixture_schedule": "ABAB alternating by trial_id",
        "stage_attribution": "Kineto/CUPTI correlation; diagnostic-only",
    }


def run_benchmark(
    config: Any,
    *,
    config_path: str | Path,
    variant_id: str,
    run_id: str,
    correctness_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the matrix, sealing each Q/N case before writing the run manifest."""

    import torch

    config_path = Path(config_path).resolve()
    destination = Path(output_path)
    csv_path = destination.with_name(f"{destination.stem}.measurements.csv")
    case_directory = destination.with_name(f"{destination.stem}.cases")
    case_paths = {
        (case.query_tokens, case.context_tokens): (
            case_directory / f"Q{case.query_tokens}-N{case.context_tokens}.json"
        )
        for case in config.benchmark_cases()
    }
    for existing in (destination, csv_path, *case_paths.values()):
        if existing.exists():
            raise FileExistsError(f"refusing to overwrite benchmark artifact: {existing}")
    plugin, options, identity = _load_configured_plugin(config, variant_id)
    reference, reference_options, reference_identity = _load_configured_plugin(
        config, config.exact_reference_variant
    )
    correctness = verify_correctness_artifact(
        correctness_path,
        config_path=config_path,
        plugin_id=plugin.descriptor.plugin_id,
        variant_fingerprint=identity["fingerprint"],
        config=config,
    )
    contract = _measurement_contract(config)
    artifact: dict[str, Any] = {
        "schema_version": 3,
        "artifact_type": "indextopk_benchmark",
        "status": "running",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config_hash": canonical_hash(config.as_mapping()),
        "measurement_contract": contract,
        "measurement_contract_hash": canonical_hash(contract),
        "correctness": correctness,
        "variant": {
            **asdict(plugin.descriptor),
            "requested_id": variant_id,
            "factory": config.variant_factory(variant_id),
            "options": options,
            "identity": identity,
        },
        "exact_reference_variant": {
            **asdict(reference.descriptor),
            "options": reference_options,
            "identity": reference_identity,
        },
        "cases": [],
        "case_artifacts": [],
        "measurements": [],
        "summary": [],
    }
    runtime: dict[str, Any] | None = None
    try:
        runtime = collect_runtime()
        validate_runtime(runtime, config)
        artifact["runtime"] = runtime
        runtime_id = runtime_identity(runtime)
        if correctness["runtime_identity_sha256"] != runtime_id["sha256"]:
            raise ValueError(
                "correctness artifact was produced under a different runtime identity"
            )
        protocol_id = protocol_identity(config)
        plan_id = plan_identity(config)
        artifact["runtime_identity"] = runtime_id
        artifact["protocol_identity"] = protocol_id
        artifact["plan_identity"] = plan_id
        protocol = BenchmarkProtocol(
            method=config.timing.method,
            warmup_iterations=config.timing.warmup_iterations,
            event_trials=config.timing.event_trials,
            kineto_trials=config.timing.kineto_trials,
            l2_flush_bytes=config.timing.l2_flush_bytes,
            cooldown_seconds=config.timing.cooldown_seconds,
        )
        for case in config.benchmark_cases():
            lifecycle = CaseLifecycle(
                case=case,
                workload=config.workload,
                reference=reference,
                reference_options=reference_options,
                mode=RunMode.BENCHMARK,
            )
            oracle = lifecycle.build_oracles()
            reference_metadata = {
                "variant_fingerprint": reference_identity["fingerprint"],
                "operator_fingerprint": reference_identity["payload"][
                    "operator_fingerprint"
                ],
                "graph_fingerprint": oracle["graph_fingerprint"],
                "fixtures": oracle["fixtures"],
                "allocator_cache_cleared_before_candidate_prepare": True,
            }
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gpu_state_before_candidate = collect_gpu_state()
            gpu_gate_before = validate_gpu_exclusivity(
                gpu_state_before_candidate,
                runtime,
                phase=f"{case.case_id}:before_candidate",
            )
            candidate_graph = lifecycle.prepare_candidate(plugin, options)
            result = benchmark_prepared_graph(
                candidate_graph,
                case,
                protocol=protocol,
                run_id=run_id,
                reference_metadata=reference_metadata,
                lifecycle=lifecycle,
            )
            result["gpu_state_before_candidate"] = gpu_state_before_candidate
            gpu_state_after_candidate = collect_gpu_state()
            gpu_gate_after = validate_gpu_exclusivity(
                gpu_state_after_candidate,
                runtime,
                phase=f"{case.case_id}:after_candidate",
            )
            result["gpu_state_after_candidate"] = gpu_state_after_candidate
            result["gpu_exclusivity_gate"] = {
                "status": "passed",
                "before_candidate": gpu_gate_before,
                "after_candidate": gpu_gate_after,
            }
            result["input_versions_unchanged"] = True
            result["experiment_identity"] = experiment_identity(
                protocol_hash=protocol_id["sha256"],
                plan_hash=plan_id["sha256"],
                operator_hash=identity["payload"]["operator_fingerprint"],
                runtime_hash=runtime_id["sha256"],
                input_content_hash=lifecycle.input_content_hash,
            )
            for row in result["measurements"]:
                row["config_hash"] = artifact["config_hash"]
                row["measurement_contract_hash"] = artifact[
                    "measurement_contract_hash"
                ]
            case_result = {
                key: value
                for key, value in result.items()
                if key not in {"measurements", "summary"}
            }
            case_artifact = {
                "schema_version": 2,
                "artifact_type": "indextopk_benchmark_case",
                "status": "complete",
                "sealed_at_utc": utc_now(),
                "run_id": run_id,
                "config_hash": artifact["config_hash"],
                "measurement_contract_hash": artifact["measurement_contract_hash"],
                "variant_fingerprint": identity["fingerprint"],
                "case": case_result,
                "measurements": result["measurements"],
                "summary": result["summary"],
            }
            case_path = case_paths[(case.query_tokens, case.context_tokens)]
            write_json_atomic(case_path, case_artifact)
            case_record = {
                "query_tokens": case.query_tokens,
                "context_tokens": case.context_tokens,
                "path": str(case_path),
                "sha256": file_sha256(case_path),
                "bytes": case_path.stat().st_size,
                "status": "complete",
            }
            artifact["case_artifacts"].append(case_record)
            artifact["cases"].append(case_result)
            artifact["measurements"].extend(result["measurements"])
            del lifecycle, candidate_graph, result
            gc.collect()
            torch.cuda.empty_cache()
        artifact["summary"] = summarize_measurements(artifact["measurements"])
        artifact["status"] = "complete"
        artifact["completed_at_utc"] = utc_now()
        write_measurements(csv_path, artifact["measurements"])
        artifact["measurement_csv"] = {
            "path": str(csv_path),
            "sha256": file_sha256(csv_path),
            "bytes": csv_path.stat().st_size,
        }
        write_json_atomic(destination, artifact)
        return artifact
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["completed_at_utc"] = utc_now()
        artifact["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        if runtime is not None:
            artifact["runtime"] = runtime
        write_json_atomic(destination, artifact)
        raise
