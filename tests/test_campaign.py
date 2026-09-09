from __future__ import annotations

import json
from pathlib import Path

import pytest

from index_topk_perflab.artifacts import canonical_hash, write_json_atomic
from index_topk_perflab.campaign import (
    build_campaign_plan,
    seal_campaign,
    write_campaign_plan,
)
from index_topk_perflab.config import load_config
from index_topk_perflab.runner import file_sha256
from index_topk_perflab.summary import summarize_measurements

ROOT = Path(__file__).resolve().parents[1]


def _benchmark(slot: dict, latency: float, plan: dict) -> dict:
    protocol_identity = plan["protocol_identity"]
    plan_identity = plan["formal_plan_identity"]
    runtime_identity = {"payload": {"runtime": "test"}}
    runtime_identity["sha256"] = canonical_hash(runtime_identity["payload"])
    contract = {
        "workload": {
            "benchmark_cases": [
                {"query_tokens": 4096, "context_tokens": 8192}
            ]
        },
        "timing": {"kineto_trials": 30, "event_trials": 20},
        "core_fingerprint": "core",
        "protocol_identity": protocol_identity,
        "plan_identity": plan_identity,
    }
    contract_hash = canonical_hash(contract)
    identity_payload = {
        "configured_id": slot["variant"],
        "core_fingerprint": "core",
        "operator_fingerprint": f"operator-{slot['variant']}",
    }
    identity = {
        "payload": identity_payload,
        "fingerprint": canonical_hash(identity_payload),
    }
    graph_hash = f"graph-{slot['variant']}"
    def row(pass_name: str, trial: int, value: float) -> dict:
        event = pass_name == "cuda_event_total"
        return {
            "variant_id": slot["variant"],
            "run_id": slot["run_id"],
            "config_hash": "config",
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
        fixture_id: canonical_hash({"fixture": fixture_id})
        for fixture_id in ("A", "B")
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

    def gate(order: tuple[str, ...], phase: str) -> dict:
        return {
            "status": "passed",
            "phase": phase,
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
            for phase in (
                "before_candidate",
                "candidate_prepare",
                "benchmark_preflight",
                "benchmark_postflight",
            )
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
        "preflight_gate": gate(("A", "B"), "benchmark_preflight"),
        "postflight_gate": gate(("B", "A"), "benchmark_postflight"),
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
    summary = summarize_measurements(rows)
    benchmark_path = Path(slot["artifact_path"])
    case_path = benchmark_path.with_name("benchmark.cases") / "Q4096-N8192.json"
    sealed_case = {
        "schema_version": 2,
        "artifact_type": "indextopk_benchmark_case",
        "status": "complete",
        "run_id": slot["run_id"],
        "config_hash": "config",
        "measurement_contract_hash": contract_hash,
        "variant_fingerprint": identity["fingerprint"],
        "case": case,
        "measurements": rows,
        "summary": summary,
    }
    write_json_atomic(case_path, sealed_case)
    return {
        "schema_version": 3,
        "artifact_type": "indextopk_benchmark",
        "status": "complete",
        "run_id": slot["run_id"],
        "config_hash": "config",
        "measurement_contract": contract,
        "measurement_contract_hash": contract_hash,
        "correctness": {
            "status": "passed",
            "variant_fingerprint": identity["fingerprint"],
            "runtime_identity_sha256": runtime_identity["sha256"],
        },
        "variant": {"plugin_id": slot["variant"], "identity": identity},
        "cases": [case],
        "case_artifacts": [
            {
                "query_tokens": 4096,
                "context_tokens": 8192,
                "path": str(case_path),
                "sha256": file_sha256(case_path),
                "bytes": case_path.stat().st_size,
                "status": "complete",
            }
        ],
        "protocol_identity": protocol_identity,
        "plan_identity": plan_identity,
        "runtime_identity": runtime_identity,
        "runtime": {},
        "measurements": rows,
        "summary": summary,
    }


def test_three_campaign_plan_is_abba_baab_and_uses_campaign_level_ci(tmp_path) -> None:
    config_path = ROOT / "configs" / "r13a_h20_release.json"
    config = load_config(config_path)
    plan = build_campaign_plan(
        config,
        config_path=config_path,
        candidate_variant="deepgemm_fused_candidate_topk_r13a_nsweep",
        run_id="unit-campaign",
        output_root=tmp_path,
    )
    assert plan["campaign_orders"] == ["ABBA", "BAAB", "ABBA"]
    assert len(plan["runs"]) == 12

    plan_path = tmp_path / "plan.json"
    write_campaign_plan(plan_path, plan)
    for slot in plan["runs"]:
        latency = 10.0 if slot["arm"] == "A" else 8.0
        write_json_atomic(slot["artifact_path"], _benchmark(slot, latency, plan))

    output = tmp_path / "campaign.json"
    result = seal_campaign(plan_path, output)
    assert result["status"] == "improved"
    assert result["uncertainty_unit"] == "independent_campaign_block"
    assert result["within_run_bootstrap_used_for_formal_ci"] is False
    assert result["cases"][0]["campaign_count"] == 3
    assert result["cases"][0]["median_speedup"] == 1.25


def _written_campaign(tmp_path: Path) -> tuple[Path, dict]:
    config_path = ROOT / "configs" / "r13a_h20_release.json"
    config = load_config(config_path)
    plan = build_campaign_plan(
        config,
        config_path=config_path,
        candidate_variant="deepgemm_fused_candidate_topk_r13a_nsweep",
        run_id="invalid-campaign",
        output_root=tmp_path,
    )
    plan_path = tmp_path / "invalid-plan.json"
    write_campaign_plan(plan_path, plan)
    for slot in plan["runs"]:
        write_json_atomic(slot["artifact_path"], _benchmark(slot, 10.0, plan))
    return plan_path, plan


def test_campaign_rejects_legacy_benchmark_schema(tmp_path) -> None:
    plan_path, plan = _written_campaign(tmp_path)
    benchmark_path = Path(plan["runs"][0]["artifact_path"])
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    write_json_atomic(benchmark_path, payload)
    with pytest.raises(ValueError, match="requires benchmark schema v3"):
        seal_campaign(plan_path, tmp_path / "campaign.json")


def test_campaign_rejects_tampered_atomic_case(tmp_path) -> None:
    plan_path, plan = _written_campaign(tmp_path)
    benchmark_path = Path(plan["runs"][0]["artifact_path"])
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    case_path = Path(payload["case_artifacts"][0]["path"])
    case_path.write_text(case_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="atomic case artifact hash/size mismatch"):
        seal_campaign(plan_path, tmp_path / "campaign.json")


def test_campaign_rejects_missing_per_case_gpu_exclusivity_gate(tmp_path) -> None:
    plan_path, plan = _written_campaign(tmp_path)
    benchmark_path = Path(plan["runs"][0]["artifact_path"])
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    payload["cases"][0].pop("gpu_exclusivity_gate")

    case_record = payload["case_artifacts"][0]
    case_path = Path(case_record["path"])
    sealed_case = json.loads(case_path.read_text(encoding="utf-8"))
    sealed_case["case"].pop("gpu_exclusivity_gate")
    write_json_atomic(case_path, sealed_case)
    case_record["sha256"] = file_sha256(case_path)
    case_record["bytes"] = case_path.stat().st_size
    write_json_atomic(benchmark_path, payload)

    with pytest.raises(ValueError, match="no passed GPU exclusivity gate"):
        seal_campaign(plan_path, tmp_path / "campaign.json")
