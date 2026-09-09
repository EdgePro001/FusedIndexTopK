from __future__ import annotations

from pathlib import Path

from index_topk_perflab.benchmark import BenchmarkProtocol
from index_topk_perflab.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_accepts_typed_config() -> None:
    config = load_config(ROOT / "configs" / "r13a_h20_release.json")
    protocol = BenchmarkProtocol.from_config(config)
    assert protocol.warmup_iterations == 10
    assert protocol.method == "deepgemm_kineto_cupti_v1"
    assert protocol.event_trials == 20
    assert protocol.kineto_trials == 30
    assert protocol.l2_flush_bytes == 8_000_000_000
    assert protocol.cooldown_seconds == 0.0


def test_formal_fusion_protocol_keeps_diagnostics_separate() -> None:
    config = load_config(ROOT / "configs" / "r13a_h20_release.json")
    protocol = BenchmarkProtocol.from_config(config)
    assert protocol.event_trials == 20
    assert protocol.kineto_trials == 30
