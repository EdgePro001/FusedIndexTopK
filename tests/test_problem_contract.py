from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from index_topk_perflab.config import load_config
from index_topk_perflab.contract import (
    plan_identity,
    profiling_plan_identity,
    protocol_identity,
    resolved_problem_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_resolved_fusion_problem_is_explicit_and_machine_readable() -> None:
    config = load_config(ROOT / "configs" / "fused_index_topk_h20.json")
    contract = resolved_problem_contract(config)

    assert contract["version"] == "fusion-v1"
    assert contract["dimensions"] == {
        "batch_size": 1,
        "query_token_values": [4096],
        "indexer_heads": 64,
        "head_dim": 128,
        "top_k": 2048,
        "benchmark_cases": [
            {"query_tokens": 4096, "context_tokens": 6144},
            {"query_tokens": 4096, "context_tokens": 8192},
            {"query_tokens": 4096, "context_tokens": 12288},
            {"query_tokens": 4096, "context_tokens": 16384},
        ],
        "context_token_values": [6144, 8192, 12288, 16384],
        "irregular_benchmark_cases": [
            {"query_tokens": 4096, "context_tokens": 6144},
            {"query_tokens": 4096, "context_tokens": 12288},
        ],
        "correctness_cases": [
            {"query_tokens": 4096, "context_tokens": 6144},
            {"query_tokens": 4096, "context_tokens": 8192},
            {"query_tokens": 4096, "context_tokens": 12288},
            {"query_tokens": 4096, "context_tokens": 16384},
        ],
        "correctness_kv_tile_remainders": {
            "q4096-kv6144": 0,
            "q4096-kv8192": 0,
            "q4096-kv12288": 0,
            "q4096-kv16384": 0,
        },
    }
    assert contract["inputs"]["q"]["dtype"] == "float8_e4m3fn"
    assert contract["inputs"]["kv_scales"]["dtype"] == "float32"
    assert contract["input_recipe"]["version"] == "contiguous-prefill-fp8-v2"
    assert contract["input_recipe"]["fixtures"] == ["A", "B"]
    assert contract["input_recipe"]["generation_order"] == "independent_rng_streams"
    assert contract["input_recipe"]["storage"].startswith("compact per case")
    assert contract["score"]["dtype"] == "float32"
    assert contract["score"]["valid_range"] == "[0,N-Q+q+1)"
    assert contract["selection"]["mode"] == "exact"
    assert contract["selection"]["tie_policy"] == "exact_score_threshold"
    assert contract["output"] == {
        "name": "indices",
        "shape": "[Q,1,K]",
        "dtype": "int32",
        "layout": "contiguous",
        "padding_index": -1,
        "scores_returned": False,
    }
    assert contract["formal_measurement"]["method"] == "deepgemm_kineto_cupti_v1"
    assert contract["formal_measurement"]["kineto_trials"] == 30
    assert contract["formal_measurement"]["event_trials"] == 20
    assert contract["formal_measurement"]["l2_flush_bytes"] == 8_000_000_000
    assert len(protocol_identity(config)["sha256"]) == 64
    assert len(plan_identity(config)["sha256"]) == 64


def test_formal_plan_identity_excludes_diagnostic_profile_controls() -> None:
    config = load_config(ROOT / "configs" / "fused_index_topk_h20.json")
    changed = replace(
        config,
        profiling=replace(
            config.profiling,
            ncu_metrics=(*config.profiling.ncu_metrics, "unit_test_metric"),
        ),
    )
    assert plan_identity(changed) == plan_identity(config)
    assert profiling_plan_identity(
        changed,
        mode="ncu",
        target_length=16384,
        stage="indexer",
    ) != profiling_plan_identity(
        config,
        mode="ncu",
        target_length=16384,
        stage="indexer",
    )
