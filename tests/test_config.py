from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fused_index_topk.config import ConfigError, build_prefill_cases, load_config, parse_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fused_index_topk_h20.json"
FUSION_CONFIG = CONFIG


def _raw() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _fusion_raw() -> dict:
    return json.loads(FUSION_CONFIG.read_text(encoding="utf-8"))


def test_load_frozen_prefill_config_and_build_cases() -> None:
    config = load_config(CONFIG)
    assert config.name == "fused_index_topk_h20_v2"
    assert config.exact_reference_variant == "deepgemm_torch_unfused"
    assert config.baseline_variant == "deepgemm_deepselect_topk"
    assert config.target.compute_capability == (9, 0)
    assert config.target.gpu_name == "NVIDIA H20-3e"
    assert config.target.multiprocessor_count == 78

    benchmark = config.benchmark_cases()
    assert [(case.query_tokens, case.context_tokens) for case in benchmark] == [
        (4096, 8192),
        (4096, 16384),
        (4096, 32768),
        (4096, 65536),
        (4096, 131072),
        (4096, 163840),
    ]
    assert benchmark[-1].query_start == 159744
    assert benchmark[-1].top_k == 2048
    assert build_prefill_cases(config, purpose="profile")[0].context_tokens == 16384
    assert [
        (case.query_tokens, case.context_tokens)
        for case in config.correctness_cases()
    ] == [
        (4096, 8192),
        (4096, 16384),
        (4096, 32768),
        (4096, 65536),
        (4096, 131072),
        (4096, 163840),
    ]


def test_load_explicit_fusion_v1_contract() -> None:
    config = load_config(FUSION_CONFIG)
    workload = config.workload
    assert config.schema_version == 4
    assert workload.q_dtype == "float8_e4m3fn"
    assert workload.kv_dtype == "float8_e4m3fn"
    assert workload.kv_scale_dtype == "float32"
    assert workload.weight_dtype == "float32"
    assert workload.range_dtype == "int32"
    assert workload.score_dtype == "float32"
    assert workload.output_dtype == "int32"
    assert workload.selection == "exact"
    assert workload.output_order == "unordered"
    assert workload.padding_index == -1
    assert workload.tie_policy == "exact_score_threshold"
    assert workload.causal_range == "[0,N-Q+q+1)"
    assert len(config.benchmark_cases()) == 6
    assert len(config.correctness_cases()) == 6
    assert config.profile_lengths("nsys") == (16384, 65536, 163840)
    assert config.profile_lengths("ncu") == (16384, 65536, 163840)
    assert config.timing.warmup_iterations == 10
    assert config.timing.method == "deepgemm_kineto_cupti_v1"
    assert config.timing.event_trials == 20
    assert config.timing.kineto_trials == 30
    assert config.timing.l2_flush_bytes == 8_000_000_000
    assert config.timing.cooldown_seconds == 0.0
    assert config.timing.campaign_repetitions == 3
    assert config.timing.campaign_orders == ("ABBA", "BAAB", "ABBA")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("q_dtype", "bfloat16"),
        ("score_dtype", "float16"),
        ("selection", "approximate"),
        ("output_order", "sorted"),
        ("padding_index", 0),
        ("tie_policy", "unspecified"),
        ("causal_range", "[0,N-Q+q)"),
    ],
)
def test_fusion_v1_rejects_semantic_drift(field: str, value: object) -> None:
    raw = _fusion_raw()
    raw["workload"][field] = value
    with pytest.raises(ConfigError, match=field):
        parse_config(raw)


def test_config_is_strict_about_unknown_fields() -> None:
    raw = _raw()
    raw["workload"]["typo"] = 1
    with pytest.raises(ConfigError, match="unknown field"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update(schema_version=5), "unsupported schema_version"),
        (lambda raw: raw["workload"].update(batch_size=2), "batch_size=1"),
        (lambda raw: raw["workload"].update(causal=False), "causal=true"),
        (lambda raw: raw["workload"].update(output_dtype="int64"), "output_dtype='int32'"),
        (lambda raw: raw["workload"].update(top_k=99999), "smallest configured"),
        (
            lambda raw: raw["profiling"]["nsys_cases"][-1].update(
                context_tokens=99999
            ),
            "subset",
        ),
        (lambda raw: raw.update(baseline_variant="missing"), "baseline_variant"),
        (
            lambda raw: raw.update(exact_reference_variant="missing"),
            "exact_reference_variant",
        ),
    ],
)
def test_invalid_contracts_are_rejected(mutate, message: str) -> None:
    raw = copy.deepcopy(_raw())
    mutate(raw)
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_case_factory_rejects_unconfigured_length_or_purpose() -> None:
    config = load_config(CONFIG)
    with pytest.raises(ConfigError, match="not configured"):
        config.make_case(12345)
    assert config.make_case(8192).query_tokens == 4096
    with pytest.raises(ConfigError, match="purpose"):
        build_prefill_cases(config, purpose="decode")
