from __future__ import annotations

from fused_index_topk.artifacts import canonical_hash
from fused_index_topk.comparison import compare_benchmarks


def _artifact(variant: str, latency: float, *, contract: str = "same") -> dict[str, object]:
    protocol_payload = {"protocol": contract}
    protocol_identity = {
        "payload": protocol_payload,
        "sha256": canonical_hash(protocol_payload),
    }
    plan_payload = {"plan": contract}
    plan_identity = {
        "payload": plan_payload,
        "sha256": canonical_hash(plan_payload),
    }
    runtime_payload = {"runtime": "same"}
    runtime_identity = {
        "payload": runtime_payload,
        "sha256": canonical_hash(runtime_payload),
    }
    contract_payload = {
        "workload": {
            "benchmark_cases": [{"query_tokens": 4096, "context_tokens": 8192}],
        },
        "timing": {"kineto_trials": 30, "event_trials": 20},
        "core_fingerprint": "core",
        "protocol_identity": protocol_identity,
        "plan_identity": plan_identity,
        "contract_tag": contract,
    }
    contract_hash = canonical_hash(contract_payload)
    identity_payload = {
        "configured_id": variant,
        "core_fingerprint": "core",
        "operator_fingerprint": f"operator-{variant}",
    }
    identity = {
        "payload": identity_payload,
        "fingerprint": canonical_hash(identity_payload),
    }
    graph_hash = f"graph-{variant}"
    config_hash = "same-config"
    def row(pass_name: str, trial: int, value: float) -> dict[str, object]:
        event = pass_name == "cuda_event_total"
        return {
                "variant_id": variant,
                "run_id": variant,
                "config_hash": config_hash,
                "measurement_contract_hash": contract_hash,
                "graph_fingerprint": graph_hash,
                "case_id": "prefill-8192",
                "query_tokens": 4096,
                "context_tokens": 8192,
                "top_k": 2048,
                "pass": pass_name,
                "trial_id": trial,
                "scope_id": "operator_total",
                "stage_id": "operator_total",
                "semantic_ops": "indexer,topk,output",
                "timing_source": "direct_cuda_event" if event else "kineto_cupti",
                "latency_ms": value * (1.0 + (trial % 3 - 1) * 0.001),
                "derived": False,
                "fixture_id": "A" if trial % 2 == 0 else "B",
            }

    rows = []
    for trial in range(30):
        rows.extend(
            (
                row("formal_kernel_sum", trial, latency),
                row("kineto_activity_sum", trial, latency * 1.01),
                row("kineto_device_span", trial, latency * 1.02),
            )
        )
    rows.extend(row("cuda_event_total", trial, latency * 1.05) for trial in range(20))

    fixture_hashes = {
        fixture_id: canonical_hash({"fixture": fixture_id}) for fixture_id in ("A", "B")
    }
    fixtures = {
        fixture_id: {
            "content": {
                "manifest": {"fixture": fixture_id},
                "sha256": fixture_hashes[fixture_id],
            }
        }
        for fixture_id in ("A", "B")
    }
    input_hash = canonical_hash(fixture_hashes)

    def gate(order: tuple[str, ...]) -> dict[str, object]:
        return {
            "status": "passed",
            "order": list(order),
            "checks": {
                f"{position}:{fixture_id}": {
                    "fixture_id": fixture_id,
                    "contract_check": {"status": "passed"},
                    "exact_reference_check": {"status": "passed"},
                }
                for position, fixture_id in enumerate(order)
            },
        }

    lifecycle = {
        "fixtures": fixtures,
        "input_content_hash": input_hash,
        "integrity_checks": [
            {"status": "passed", "phase": phase}
            for phase in ("before_candidate", "candidate_prepare", "benchmark_postflight")
        ],
        "anti_precompute_gate": {
            "candidate_prepare_fixture": "A",
            "required_execution_fixtures": ["A", "B"],
            "same_shape_different_content": True,
        },
    }
    components = {
        "ProtocolHash": protocol_identity["sha256"],
        "PlanHash": plan_identity["sha256"],
        "OperatorHash": identity_payload["operator_fingerprint"],
        "RuntimeHash": runtime_identity["sha256"],
        "InputContentHash": input_hash,
    }
    case = {
        "case": {
            "case_id": "prefill-8192",
            "query_tokens": 4096,
            "context_tokens": 8192,
        },
        "graph_fingerprint": graph_hash,
        "graph": {"nodes": []},
        "input_fixtures": fixtures,
        "input_fingerprint": input_hash,
        "input_content_hash": input_hash,
        "lifecycle": lifecycle,
        "preflight_gate": gate(("A", "B")),
        "postflight_gate": gate(("B", "A")),
        "large_case_contract_check": {"status": "passed"},
        "exact_reference_check": {"status": "passed"},
        "postflight_contract_check": {"status": "passed"},
        "postflight_exact_reference_check": {"status": "passed"},
        "gpu_exclusivity_gate": {
            "status": "passed",
            "before_candidate": {"status": "passed", "other_processes": []},
            "after_candidate": {"status": "passed", "other_processes": []},
        },
        "kineto": {
            "topology_gate": {
                "status": "passed",
                "trial_count": 30,
                "operator_kernel_count": 1,
                "operator_activity_count": 1,
                "operator_activity_signature_sha256": "b" * 64,
                "stage_kernel_counts": {},
                "stage_activity_counts": {},
            }
        },
        "experiment_identity": {
            "components": components,
            "sha256": canonical_hash(components),
        },
    }
    return {
        "schema_version": 3,
        "artifact_type": "indextopk_benchmark",
        "status": "complete",
        "run_id": variant,
        "config_hash": config_hash,
        "measurement_contract": contract_payload,
        "measurement_contract_hash": contract_hash,
        "correctness": {
            "status": "passed",
            "variant_fingerprint": identity["fingerprint"],
            "runtime_identity_sha256": runtime_identity["sha256"],
        },
        "variant": {"plugin_id": variant, "identity": identity},
        "cases": [case],
        "case_artifacts": [
            {
                "query_tokens": 4096,
                "context_tokens": 8192,
                "path": f"fake-{variant}.json",
                "sha256": "a" * 64,
                "bytes": 1,
                "status": "complete",
            }
        ],
        "protocol_identity": protocol_identity,
        "plan_identity": plan_identity,
        "runtime_identity": runtime_identity,
        "runtime": {
            "selected_device": {
                "name": "NVIDIA H20-3e",
                "capability": [9, 0],
                "multiprocessor_count": 78,
                "total_memory_bytes": 150109880320,
            },
            "torch": "2.10.0+cu130",
            "torch_cuda_build": "13.0",
        },
        "measurements": rows,
    }


def test_comparison_uses_only_formal_direct_total() -> None:
    result = compare_benchmarks(_artifact("base", 10.0), _artifact("fast", 8.0))
    assert result["status"] == "improved"
    assert result["cases"][0]["speedup"] == 1.25


def test_comparison_rejects_different_contracts() -> None:
    result = compare_benchmarks(
        _artifact("base", 10.0, contract="a"),
        _artifact("candidate", 8.0, contract="b"),
    )
    assert result["status"] == "incomparable"


def test_comparison_reports_regression() -> None:
    result = compare_benchmarks(_artifact("base", 10.0), _artifact("slow", 12.0))
    assert result["status"] == "regressed"


def test_comparison_rejects_duplicate_formal_trial() -> None:
    candidate = _artifact("candidate", 8.0)
    candidate["measurements"].append(dict(candidate["measurements"][0]))
    result = compare_benchmarks(_artifact("base", 10.0), candidate)
    assert result["status"] == "incomparable"
    assert any("duplicate measurement" in item for item in result["comparability"]["failures"])


def test_comparison_requires_full_size_exact_reference() -> None:
    candidate = _artifact("candidate", 8.0)
    candidate["cases"][0]["exact_reference_check"] = {"status": "not_evaluated"}
    result = compare_benchmarks(_artifact("base", 10.0), candidate)
    assert result["status"] == "incomparable"


def test_comparison_requires_passed_kineto_topology_gate() -> None:
    candidate = _artifact("candidate", 8.0)
    candidate["cases"][0]["kineto"]["topology_gate"]["status"] = "failed"
    result = compare_benchmarks(_artifact("base", 10.0), candidate)
    assert result["status"] == "incomparable"
    assert any(
        "Kineto topology gate" in item
        for item in result["comparability"]["failures"]
    )
