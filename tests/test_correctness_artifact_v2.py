from __future__ import annotations

import json
from pathlib import Path

import pytest

from fused_index_topk.artifacts import canonical_hash, write_json_atomic
from fused_index_topk.config import load_config
from fused_index_topk.contract import plan_identity, protocol_identity
from fused_index_topk.provenance import experiment_identity, variant_identity
from fused_index_topk.registry import load_variant
from fused_index_topk.runner import file_sha256, verify_correctness_artifact

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fused_index_topk_h20.json"


def _artifact() -> tuple[dict, str, str]:
    config = load_config(CONFIG)
    # This is a CPU-only artifact schema fixture. Keep it independent of the
    # selected H20 baseline and its locked external source checkout.
    variant_id = "deepgemm_torch_unfused"
    options = config.variant_options(variant_id)
    plugin = load_variant(config.variant_factory(variant_id), options=options)
    identity = variant_identity(config, variant_id, plugin, options=options)
    protocol = protocol_identity(config)
    plan = plan_identity(config)
    runtime = {"payload": {"runtime": "unit"}}
    runtime["sha256"] = canonical_hash(runtime["payload"])
    cases = []
    for case in config.correctness_cases():
        fixture_hashes = {
            fixture_id: canonical_hash(
                {
                    "query_tokens": case.query_tokens,
                    "context_tokens": case.context_tokens,
                    "fixture_id": fixture_id,
                }
            )
            for fixture_id in ("A", "B")
        }
        fixtures = {
            fixture_id: {
                "content": {
                    "manifest": {
                        "query_tokens": case.query_tokens,
                        "context_tokens": case.context_tokens,
                        "fixture_id": fixture_id,
                    },
                    "sha256": fixture_hashes[fixture_id],
                }
            }
            for fixture_id in ("A", "B")
        }
        input_hash = canonical_hash(fixture_hashes)
        components = experiment_identity(
            protocol_hash=protocol["sha256"],
            plan_hash=plan["sha256"],
            operator_hash=identity["payload"]["operator_fingerprint"],
            runtime_hash=runtime["sha256"],
            input_content_hash=input_hash,
        )
        checks = {
            f"{position}:{fixture_id}": {
                "fixture_id": fixture_id,
                "contract_check": {"status": "passed"},
                "exact_reference_check": {"status": "passed"},
            }
            for position, fixture_id in enumerate(("A", "B", "A"))
        }
        cases.append(
            {
                "case": {
                    "case_id": case.case_id,
                    "query_tokens": case.query_tokens,
                    "context_tokens": case.context_tokens,
                },
                "input_fixtures": fixtures,
                "input_fingerprint": input_hash,
                "input_content_hash": input_hash,
                "experiment_identity": components,
                "lifecycle": {
                    "input_content_hash": input_hash,
                    "integrity_checks": [
                        {"status": "passed", "phase": phase}
                        for phase in ("before_candidate", "candidate_prepare", "correctness")
                    ],
                    "anti_precompute_gate": {
                        "candidate_prepare_fixture": "A",
                        "required_execution_fixtures": ["A", "B"],
                        "same_shape_different_content": True,
                    },
                },
                "reference_check": {
                    "status": "passed",
                    "fixtures": {
                        fixture_id: {
                            "contract_check": {"status": "passed"},
                            "cutoff_tie_check": {"status": "passed"},
                        }
                        for fixture_id in ("A", "B")
                    },
                },
                "candidate_check": {
                    "status": "passed",
                    "order": ["A", "B", "A"],
                    "checks": checks,
                },
            }
        )
    return (
        {
            "schema_version": 2,
            "artifact_type": "indextopk_correctness",
            "status": "passed",
            "run_id": "unit",
            "config_sha256": file_sha256(CONFIG),
            "variant": {"plugin_id": variant_id, "identity": identity},
            "protocol_identity": protocol,
            "plan_identity": plan,
            "runtime_identity": runtime,
            "cases": cases,
        },
        variant_id,
        identity["fingerprint"],
    )


def test_v2_correctness_artifact_is_bound_to_complete_ab_evidence(tmp_path) -> None:
    artifact, variant_id, fingerprint = _artifact()
    path = tmp_path / "correctness.json"
    write_json_atomic(path, artifact)
    record = verify_correctness_artifact(
        path,
        config_path=CONFIG,
        plugin_id=variant_id,
        variant_fingerprint=fingerprint,
    )
    assert record["runtime_identity_sha256"] == artifact["runtime_identity"]["sha256"]


def test_v2_correctness_artifact_rejects_missing_b_execution(tmp_path) -> None:
    artifact, variant_id, fingerprint = _artifact()
    artifact["cases"][0]["candidate_check"]["checks"] = {
        "0:A": artifact["cases"][0]["candidate_check"]["checks"]["0:A"]
    }
    path = tmp_path / "correctness.json"
    write_json_atomic(path, artifact)
    with pytest.raises(ValueError, match="did not execute A and B"):
        verify_correctness_artifact(
            path,
            config_path=CONFIG,
            plugin_id=variant_id,
            variant_fingerprint=fingerprint,
        )


def test_correctness_reuse_ignores_profile_only_config_change(tmp_path) -> None:
    artifact, variant_id, fingerprint = _artifact()
    artifact_path = tmp_path / "correctness.json"
    write_json_atomic(artifact_path, artifact)
    raw_config = json.loads(CONFIG.read_text(encoding="utf-8"))
    raw_config["profiling"]["ncu_metrics"].append("unit_test_metric")
    changed_config = tmp_path / "profile-only-change.json"
    write_json_atomic(changed_config, raw_config)
    record = verify_correctness_artifact(
        artifact_path,
        config_path=changed_config,
        plugin_id=variant_id,
        variant_fingerprint=fingerprint,
    )
    assert record["config_file_match"] is False
