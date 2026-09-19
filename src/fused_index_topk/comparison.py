"""CPU-only, correctness-gated comparison of two formal benchmark artifacts."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import canonical_hash, load_json, write_json_atomic
from .summary import summarize_measurements, validate_measurements

CaseKey = tuple[int, int]


def _case_key(value: Mapping[str, Any]) -> CaseKey:
    return int(value["query_tokens"]), int(value["context_tokens"])


def _case_label(key: CaseKey) -> str:
    return f"Q={key[0]}, N={key[1]}"


def _configured_case_keys(workload: Mapping[str, Any]) -> set[CaseKey]:
    raw_cases = workload.get("benchmark_cases")
    if isinstance(raw_cases, list):
        return {
            _case_key(item)
            for item in raw_cases
            if isinstance(item, Mapping)
        }
    query_tokens = int(workload.get("query_tokens", -1))
    return {
        (query_tokens, int(context_tokens))
        for context_tokens in workload.get("context_lengths", [])
    }


def _load_benchmark(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    payload = load_json(value) if isinstance(value, (str, Path)) else dict(value)
    if payload.get("artifact_type") != "indextopk_benchmark":
        raise ValueError("comparison input is not a FusedIndexTopK benchmark")
    return payload


def _formal_samples(payload: Mapping[str, Any]) -> dict[CaseKey, list[float]]:
    rows = validate_measurements(payload.get("measurements", []))
    samples: dict[CaseKey, list[float]] = defaultdict(list)
    for row in rows:
        if row["pass"] == "formal_kernel_sum" and row["scope_id"] == "operator_total":
            samples[_case_key(row)].append(float(row["latency_ms"]))
    if not samples:
        raise ValueError("benchmark contains no formal Kineto/CUPTI kernel-sum samples")
    return dict(samples)


def _event_samples(payload: Mapping[str, Any]) -> dict[CaseKey, list[float]]:
    rows = validate_measurements(payload.get("measurements", []))
    samples: dict[CaseKey, list[float]] = defaultdict(list)
    for row in rows:
        if row["pass"] == "cuda_event_total" and row["scope_id"] == "operator_total":
            samples[_case_key(row)].append(float(row["latency_ms"]))
    if not samples:
        raise ValueError("benchmark contains no supplemental CUDA-event samples")
    return dict(samples)


def _case_map(payload: Mapping[str, Any]) -> dict[CaseKey, Mapping[str, Any]]:
    result: dict[CaseKey, Mapping[str, Any]] = {}
    for item in payload.get("cases", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("case"), Mapping):
            continue
        key = _case_key(item["case"])
        if key in result:
            raise ValueError(f"duplicate benchmark case for {_case_label(key)}")
        result[key] = item
    return result


def _fixture_gate_errors(
    gate: Any,
    *,
    expected_order: Sequence[str],
    label: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(gate, Mapping) or gate.get("status") != "passed":
        return [f"{label} is not passed"]
    if gate.get("order") != list(expected_order):
        errors.append(f"{label} fixture order is invalid")
    checks = gate.get("checks")
    if not isinstance(checks, Mapping) or len(checks) != len(expected_order):
        return [*errors, f"{label} fixture checks are incomplete"]
    realized_order: list[str] = []
    for position in range(len(expected_order)):
        check = checks.get(f"{position}:{expected_order[position]}")
        if not isinstance(check, Mapping):
            errors.append(f"{label} is missing fixture check {position}")
            continue
        realized_order.append(str(check.get("fixture_id")))
        for name in ("contract_check", "exact_reference_check"):
            record = check.get(name)
            if not isinstance(record, Mapping) or record.get("status") != "passed":
                errors.append(f"{label} fixture {position} has no passed {name}")
    if realized_order != list(expected_order):
        errors.append(f"{label} realized fixture order is invalid")
    return errors


def _input_evidence_errors(item: Mapping[str, Any], *, case_key: CaseKey) -> list[str]:
    errors: list[str] = []
    label = _case_label(case_key)
    content_hash = item.get("input_content_hash")
    fixtures = item.get("input_fixtures")
    lifecycle = item.get("lifecycle")
    if not isinstance(fixtures, Mapping) or not isinstance(lifecycle, Mapping):
        return [f"case {label} has no structured A/B input evidence"]
    fixture_hashes: dict[str, str] = {}
    for fixture_id in ("A", "B"):
        fixture = fixtures.get(fixture_id)
        content = fixture.get("content") if isinstance(fixture, Mapping) else None
        manifest = content.get("manifest") if isinstance(content, Mapping) else None
        digest = content.get("sha256") if isinstance(content, Mapping) else None
        if not isinstance(manifest, Mapping) or canonical_hash(manifest) != digest:
            errors.append(f"case {label} fixture {fixture_id} content hash is invalid")
            continue
        fixture_hashes[fixture_id] = str(digest)
    if set(fixture_hashes) == {"A", "B"}:
        if fixture_hashes["A"] == fixture_hashes["B"]:
            errors.append(f"case {label} A/B fixture contents are identical")
        if canonical_hash(fixture_hashes) != content_hash:
            errors.append(f"case {label} combined input content hash is invalid")
    if lifecycle.get("fixtures") != fixtures:
        errors.append(f"case {label} lifecycle fixture evidence differs")
    if lifecycle.get("input_content_hash") != content_hash:
        errors.append(f"case {label} lifecycle input hash differs")
    anti_precompute = lifecycle.get("anti_precompute_gate")
    if (
        not isinstance(anti_precompute, Mapping)
        or anti_precompute.get("candidate_prepare_fixture") != "A"
        or anti_precompute.get("required_execution_fixtures") != ["A", "B"]
        or anti_precompute.get("same_shape_different_content") is not True
    ):
        errors.append(f"case {label} anti-precompute contract is invalid")
    return errors


def benchmark_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Validate a sealed benchmark before any statistics are computed."""

    errors: list[str] = []
    schema_version = payload.get("schema_version")
    if schema_version not in {1, 2, 3}:
        errors.append("unsupported benchmark schema_version")
    if payload.get("status") != "complete":
        errors.append("benchmark status is not complete")
    contract = payload.get("measurement_contract")
    if not isinstance(contract, Mapping):
        errors.append("measurement_contract is missing")
        return errors
    if canonical_hash(contract) != payload.get("measurement_contract_hash"):
        errors.append("measurement_contract hash does not match its contents")

    variant = payload.get("variant")
    identity = variant.get("identity") if isinstance(variant, Mapping) else None
    if not isinstance(identity, Mapping) or not isinstance(identity.get("payload"), Mapping):
        errors.append("variant identity is missing")
        variant_fingerprint = None
    else:
        variant_fingerprint = identity.get("fingerprint")
        if canonical_hash(identity["payload"]) != variant_fingerprint:
            errors.append("variant identity fingerprint does not match its payload")
    plugin_id = variant.get("plugin_id") if isinstance(variant, Mapping) else None
    correctness = payload.get("correctness")
    if not isinstance(correctness, Mapping) or correctness.get("status") != "passed":
        errors.append("correctness gate is not passed")
    elif correctness.get("variant_fingerprint") != variant_fingerprint:
        errors.append("correctness gate is bound to a different variant fingerprint")

    workload = contract.get("workload")
    timing = contract.get("timing")
    if not isinstance(workload, Mapping) or not isinstance(timing, Mapping):
        errors.append("measurement contract lacks workload or timing")
        return errors
    expected_cases = _configured_case_keys(workload)
    try:
        cases = _case_map(payload)
    except (KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        cases = {}
    if set(cases) != expected_cases:
        errors.append("case matrix does not match measurement contract")
    if schema_version >= 2:
        identities: dict[str, Mapping[str, Any]] = {}
        for name in ("protocol_identity", "plan_identity", "runtime_identity"):
            identity_record = payload.get(name)
            if not isinstance(identity_record, Mapping):
                errors.append(f"{name} is missing")
                continue
            identity_payload = identity_record.get("payload")
            if (
                not isinstance(identity_payload, Mapping)
                or canonical_hash(identity_payload) != identity_record.get("sha256")
            ):
                errors.append(f"{name} hash does not match its payload")
                continue
            identities[name] = identity_record
        if identities.get("protocol_identity") != contract.get("protocol_identity"):
            errors.append("protocol identity differs from the measurement contract")
        if identities.get("plan_identity") != contract.get("plan_identity"):
            errors.append("plan identity differs from the measurement contract")
        if isinstance(identity, Mapping):
            variant_identity_payload = identity.get("payload")
            identity_core = (
                variant_identity_payload.get("core_fingerprint")
                if isinstance(variant_identity_payload, Mapping)
                else None
            )
            if identity_core != contract.get("core_fingerprint"):
                errors.append("variant core fingerprint differs from the measurement contract")
        correctness_runtime = (
            correctness.get("runtime_identity_sha256")
            if isinstance(correctness, Mapping)
            else None
        )
        runtime_sha = (payload.get("runtime_identity") or {}).get("sha256")
        if correctness_runtime != runtime_sha:
            errors.append("correctness gate is bound to a different runtime identity")
        case_artifacts = payload.get("case_artifacts")
        if not isinstance(case_artifacts, list) or len(case_artifacts) != len(
            expected_cases
        ):
            errors.append("per-case atomic artifact manifest is incomplete")
        elif {
            (
                int(item.get("query_tokens", -1)),
                int(item.get("context_tokens", -1)),
            )
            for item in case_artifacts
            if isinstance(item, Mapping) and item.get("status") == "complete"
        } != expected_cases:
            errors.append("per-case atomic artifact manifest does not cover the matrix")
        else:
            paths: set[str] = set()
            for record in case_artifacts:
                path = record.get("path") if isinstance(record, Mapping) else None
                digest = record.get("sha256") if isinstance(record, Mapping) else None
                size = record.get("bytes") if isinstance(record, Mapping) else None
                if not isinstance(path, str) or not path or path in paths:
                    errors.append("per-case atomic artifact manifest has an invalid path")
                else:
                    paths.add(path)
                if not isinstance(digest, str) or len(digest) != 64:
                    errors.append("per-case atomic artifact manifest has an invalid hash")
                if not isinstance(size, int) or size <= 0:
                    errors.append("per-case atomic artifact manifest has an invalid size")
    for case_key, item in cases.items():
        label = _case_label(case_key)
        if schema_version >= 2:
            exclusivity = item.get("gpu_exclusivity_gate")
            if (
                not isinstance(exclusivity, Mapping)
                or exclusivity.get("status") != "passed"
                or any(
                    not isinstance(exclusivity.get(name), Mapping)
                    or exclusivity[name].get("status") != "passed"
                    or exclusivity[name].get("other_processes") != []
                    for name in ("before_candidate", "after_candidate")
                )
            ):
                errors.append(f"case {label} has no passed GPU exclusivity gate")
        for check_name in (
            "exact_reference_check",
            "postflight_exact_reference_check",
            "large_case_contract_check",
            "postflight_contract_check",
        ):
            check = item.get(check_name)
            if not isinstance(check, Mapping) or check.get("status") != "passed":
                errors.append(f"case {label} has no passed {check_name}")
        if not item.get("input_fingerprint"):
            errors.append(f"case {label} has no input fingerprint")
        if schema_version >= 2:
            content_hash = item.get("input_content_hash")
            if not content_hash or content_hash != item.get("input_fingerprint"):
                errors.append(f"case {label} has invalid input content identity")
            errors.extend(_input_evidence_errors(item, case_key=case_key))
            lifecycle = item.get("lifecycle")
            checks = lifecycle.get("integrity_checks") if isinstance(lifecycle, Mapping) else None
            phases = {
                check.get("phase")
                for check in checks or []
                if isinstance(check, Mapping) and check.get("status") == "passed"
            }
            if not {"before_candidate", "candidate_prepare", "benchmark_postflight"}.issubset(
                phases
            ):
                errors.append(f"case {label} has incomplete lifecycle integrity gates")
            errors.extend(
                _fixture_gate_errors(
                    item.get("preflight_gate"),
                    expected_order=("A", "B"),
                    label=f"case {label} preflight gate",
                )
            )
            errors.extend(
                _fixture_gate_errors(
                    item.get("postflight_gate"),
                    expected_order=("B", "A"),
                    label=f"case {label} postflight gate",
                )
            )
            experiment = item.get("experiment_identity")
            components = (
                experiment.get("components") if isinstance(experiment, Mapping) else None
            )
            expected_components = {
                "ProtocolHash": (payload.get("protocol_identity") or {}).get("sha256"),
                "PlanHash": (payload.get("plan_identity") or {}).get("sha256"),
                "OperatorHash": (
                    identity.get("payload", {}).get("operator_fingerprint")
                    if isinstance(identity, Mapping)
                    else None
                ),
                "RuntimeHash": (payload.get("runtime_identity") or {}).get("sha256"),
                "InputContentHash": content_hash,
            }
            if (
                components != expected_components
                or not isinstance(experiment, Mapping)
                or canonical_hash(components) != experiment.get("sha256")
            ):
                errors.append(f"case {label} experiment identity is invalid")
        if schema_version >= 3:
            kineto = item.get("kineto")
            topology = kineto.get("topology_gate") if isinstance(kineto, Mapping) else None
            expected_trials = int(timing.get("kineto_trials", -1))
            if (
                not isinstance(topology, Mapping)
                or topology.get("status") != "passed"
                or topology.get("trial_count") != expected_trials
                or not isinstance(topology.get("operator_kernel_count"), int)
                or int(topology.get("operator_kernel_count", 0)) <= 0
                or not isinstance(topology.get("operator_activity_signature_sha256"), str)
                or len(str(topology.get("operator_activity_signature_sha256"))) != 64
            ):
                errors.append(f"case {label} has no valid Kineto topology gate")

    try:
        rows = validate_measurements(payload.get("measurements", []))
    except (TypeError, ValueError) as error:
        errors.append(f"invalid measurement table: {error}")
        return errors
    seen: set[tuple[Any, ...]] = set()
    pass_trials: dict[tuple[CaseKey, str], set[int]] = defaultdict(set)
    stage_trials: dict[tuple[CaseKey, int], set[str]] = defaultdict(set)
    for row in rows:
        case_key = _case_key(row)
        label = _case_label(case_key)
        case = cases.get(case_key)
        key = (
            row.get("run_id"),
            row["case_id"],
            row["pass"],
            row["trial_id"],
            row["stage_id"],
        )
        if key in seen:
            errors.append(f"duplicate measurement primary key: {key}")
        seen.add(key)
        if row.get("variant_id") != plugin_id:
            errors.append("measurement row variant_id differs from artifact variant")
        if row.get("run_id") != payload.get("run_id"):
            errors.append("measurement row run_id differs from artifact run_id")
        if row.get("config_hash") != payload.get("config_hash"):
            errors.append("measurement row config_hash differs from artifact")
        if row.get("measurement_contract_hash") != payload.get(
            "measurement_contract_hash"
        ):
            errors.append("measurement row contract hash differs from artifact")
        if case is None:
            errors.append(f"measurement uses unconfigured case {label}")
            continue
        if row["case_id"] != case["case"]["case_id"]:
            errors.append(f"case {label} measurement case_id mismatch")
        if row.get("graph_fingerprint") != case.get("graph_fingerprint"):
            errors.append(f"case {label} measurement graph fingerprint mismatch")
        pass_trials[(case_key, str(row["pass"]))].add(int(row["trial_id"]))
        if row["pass"] == "kineto_stage_kernel_sum":
            stage_trials[(case_key, int(row["trial_id"]))].add(str(row["stage_id"]))
        if row["pass"] in {"formal_kernel_sum", "cuda_event_total", "clean_total"}:
            if schema_version >= 2:
                expected_fixture = "A" if int(row["trial_id"]) % 2 == 0 else "B"
                if row.get("fixture_id") != expected_fixture:
                    errors.append(
                        f"case {label} trial {row['trial_id']} fixture schedule differs"
                    )

    for case_key in expected_cases:
        label = _case_label(case_key)
        if schema_version >= 3:
            expected_kineto = set(range(int(timing.get("kineto_trials", -1))))
            expected_event = set(range(int(timing.get("event_trials", -1))))
            for pass_name in (
                "formal_kernel_sum",
                "kineto_activity_sum",
                "kineto_device_span",
            ):
                if pass_trials[(case_key, pass_name)] != expected_kineto:
                    errors.append(
                        f"case {label} {pass_name} trial IDs are incomplete or duplicated"
                    )
            if pass_trials[(case_key, "cuda_event_total")] != expected_event:
                errors.append(
                    f"case {label} cuda_event_total trial IDs are incomplete or duplicated"
                )
            stages = {
                str(node["stage_id"])
                for node in cases.get(case_key, {}).get("graph", {}).get("nodes", [])
                if node.get("profile", True)
            }
            for trial in expected_kineto:
                if stage_trials[(case_key, trial)] != stages:
                    errors.append(
                        f"case {label} Kineto trial {trial} does not cover graph stages"
                    )
        else:
            expected_clean = set(range(int(timing.get("clean_trials", -1))))
            if pass_trials[(case_key, "clean_total")] != expected_clean:
                errors.append(
                    f"case {label} clean trial IDs are incomplete or duplicated"
                )
    return list(dict.fromkeys(errors))


def _bootstrap_speedup(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    repetitions: int = 2000,
    seed: int = 20260805,
) -> tuple[float, float]:
    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(repetitions):
        baseline_draw = [generator.choice(baseline) for _ in baseline]
        candidate_draw = [generator.choice(candidate) for _ in candidate]
        denominator = statistics.median(candidate_draw)
        ratios.append(statistics.median(baseline_draw) / denominator)
    ratios.sort()
    return (
        ratios[int(0.025 * (len(ratios) - 1))],
        ratios[int(0.975 * (len(ratios) - 1))],
    )


def _runtime_signature(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime", {})
    selected = runtime.get("selected_device", {}) if isinstance(runtime, Mapping) else {}
    smi_rows = runtime.get("nvidia_smi", []) if isinstance(runtime, Mapping) else []
    smi = smi_rows[0] if isinstance(smi_rows, list) and smi_rows else {}
    imports = runtime.get("imports", {}) if isinstance(runtime, Mapping) else {}
    return {
        "gpu_name": selected.get("name"),
        "capability": selected.get("capability"),
        "multiprocessor_count": selected.get("multiprocessor_count"),
        "total_memory_bytes": selected.get("total_memory_bytes"),
        "gpu_uuid": smi.get("uuid") if isinstance(smi, Mapping) else None,
        "driver_version": smi.get("driver_version") if isinstance(smi, Mapping) else None,
        "memory_total": smi.get("memory.total") if isinstance(smi, Mapping) else None,
        "power_limit": smi.get("power.limit") if isinstance(smi, Mapping) else None,
        "deep_gemm_import": (
            imports.get("deep_gemm") if isinstance(imports, Mapping) else None
        ),
        "torch": runtime.get("torch") if isinstance(runtime, Mapping) else None,
        "torch_cuda_build": (
            runtime.get("torch_cuda_build") if isinstance(runtime, Mapping) else None
        ),
    }


def compare_benchmarks(
    baseline: str | Path | Mapping[str, Any],
    candidate: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Compare CUPTI kernel sums with CUDA-event latency as a guardrail."""

    base = _load_benchmark(baseline)
    cand = _load_benchmark(candidate)
    comparability_failures = [
        *(f"baseline: {error}" for error in benchmark_validation_errors(base)),
        *(f"candidate: {error}" for error in benchmark_validation_errors(cand)),
    ]
    if base.get("measurement_contract_hash") != cand.get("measurement_contract_hash"):
        comparability_failures.append("measurement contract hashes differ")
    if _runtime_signature(base) != _runtime_signature(cand):
        comparability_failures.append("GPU/PyTorch runtime signatures differ")
    try:
        base_samples = _formal_samples(base)
        cand_samples = _formal_samples(cand)
        base_event_samples = _event_samples(base)
        cand_event_samples = _event_samples(cand)
    except ValueError as error:
        comparability_failures.append(str(error))
        base_samples, cand_samples = {}, {}
        base_event_samples, cand_event_samples = {}, {}
    if set(base_samples) != set(cand_samples):
        comparability_failures.append("formal Q/N case matrices differ")
    if set(base_event_samples) != set(cand_event_samples) or set(base_event_samples) != set(
        base_samples
    ):
        comparability_failures.append("supplemental CUDA-event case matrices differ")
    try:
        base_cases = _case_map(base)
        cand_cases = _case_map(cand)
    except ValueError as error:
        comparability_failures.append(str(error))
        base_cases, cand_cases = {}, {}
    if {
        case_key: item.get("input_fingerprint") for case_key, item in base_cases.items()
    } != {
        case_key: item.get("input_fingerprint") for case_key, item in cand_cases.items()
    }:
        comparability_failures.append("input fingerprints differ")
    result: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "indextopk_comparison",
        "baseline": {
            "run_id": base.get("run_id"),
            "variant": base.get("variant", {}).get("plugin_id"),
        },
        "candidate": {
            "run_id": cand.get("run_id"),
            "variant": cand.get("variant", {}).get("plugin_id"),
        },
        "comparability": {
            "status": "failed" if comparability_failures else "passed",
            "failures": comparability_failures,
        },
        "formal_metric": "formal_kernel_sum/operator_total/kineto_cupti",
        "guardrail_metric": "cuda_event_total/operator_total/direct_cuda_event",
        "comparison_quality": "unpaired_independent_runs",
        "cases": [],
    }
    if comparability_failures:
        result["status"] = "incomparable"
        return result

    unstable = False
    case_statuses: list[str] = []
    for case_key in sorted(base_samples):
        query_tokens, context_tokens = case_key
        baseline_values = base_samples[case_key]
        candidate_values = cand_samples[case_key]
        baseline_summary = summarize_measurements(
            [
                row
                for row in base["measurements"]
                if row["pass"] == "formal_kernel_sum"
                and _case_key(row) == case_key
            ]
        )[0]
        candidate_summary = summarize_measurements(
            [
                row
                for row in cand["measurements"]
                if row["pass"] == "formal_kernel_sum"
                and _case_key(row) == case_key
            ]
        )[0]
        baseline_event_summary = summarize_measurements(
            [
                row
                for row in base["measurements"]
                if row["pass"] == "cuda_event_total" and _case_key(row) == case_key
            ]
        )[0]
        candidate_event_summary = summarize_measurements(
            [
                row
                for row in cand["measurements"]
                if row["pass"] == "cuda_event_total" and _case_key(row) == case_key
            ]
        )[0]
        baseline_median = float(baseline_summary["median_ms"])
        candidate_median = float(candidate_summary["median_ms"])
        speedup = baseline_median / candidate_median
        ci_low, ci_high = _bootstrap_speedup(
            baseline_values,
            candidate_values,
            seed=20260805 + query_tokens * 31 + context_tokens,
        )
        stable = (
            len(baseline_values) >= 20
            and len(candidate_values) >= 20
            and baseline_summary["normalized_mad"] <= 0.05
            and candidate_summary["normalized_mad"] <= 0.05
            and baseline_summary["p90_ms"] / baseline_median <= 1.10
            and candidate_summary["p90_ms"] / candidate_median <= 1.10
        )
        unstable |= not stable
        p95_ratio = candidate_summary["p95_ms"] / baseline_summary["p95_ms"]
        event_speedup = (
            baseline_event_summary["median_ms"]
            / candidate_event_summary["median_ms"]
        )
        event_p95_ratio = (
            candidate_event_summary["p95_ms"] / baseline_event_summary["p95_ms"]
        )
        if (
            speedup >= 1.03
            and ci_low > 1.0
            and p95_ratio <= 1.05
            and event_p95_ratio <= 1.05
        ):
            case_status = "improved"
        elif speedup <= 0.97 and ci_high < 1.0:
            case_status = "regressed"
        else:
            case_status = "inconclusive"
        case_statuses.append(case_status)
        result["cases"].append(
            {
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "speedup": speedup,
                "bootstrap_95pct_speedup_ci": [ci_low, ci_high],
                "candidate_to_baseline_p95_ratio": p95_ratio,
                "baseline_cuda_event": baseline_event_summary,
                "candidate_cuda_event": candidate_event_summary,
                "cuda_event_speedup": event_speedup,
                "candidate_to_baseline_cuda_event_p95_ratio": event_p95_ratio,
                "stability": "passed" if stable else "failed",
                "status": case_status,
            }
        )

    if unstable:
        result["status"] = "unstable"
    elif all(status == "improved" for status in case_statuses):
        result["status"] = "improved"
    elif any(status == "regressed" for status in case_statuses):
        result["status"] = "regressed"
    else:
        result["status"] = "inconclusive"
    return result


def write_comparison(
    path: str | Path,
    baseline: str | Path | Mapping[str, Any],
    candidate: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    result = compare_benchmarks(baseline, candidate)
    write_json_atomic(path, result)
    return result
