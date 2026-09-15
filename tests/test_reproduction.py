from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from index_topk_perflab.config import load_config
from index_topk_perflab.reproduction import (
    build_smoke_config,
    parse_nvidia_smi_rows,
    render_report,
    select_visible_gpu,
)

ROOT = Path(__file__).resolve().parents[1]


def test_parse_and_select_visible_gpu() -> None:
    rows = parse_nvidia_smi_rows(
        "0, NVIDIA H20-3e, GPU-aaa, 143156, 580.65.06\n"
        "1, NVIDIA H20-3e, GPU-bbb, 143156, 580.65.06\n"
    )
    assert len(rows) == 2
    assert select_visible_gpu(rows, "0")["uuid"] == "GPU-aaa"
    assert select_visible_gpu(rows, "GPU-bbb")["index"] == "1"
    assert select_visible_gpu(rows, "0,1") is None


def test_build_smoke_config_limits_the_matrix_without_mutating_source() -> None:
    loaded = load_config(ROOT / "configs" / "fused_index_topk_h20.json")
    assert loaded.name == "fused_index_topk_h20_v1"
    raw = json.loads((ROOT / "configs" / "fused_index_topk_h20.json").read_text())
    original = deepcopy(raw)
    smoke = build_smoke_config(raw)
    assert raw == original
    assert smoke["workload"]["correctness_cases"] == [
        {"query_tokens": 4096, "context_tokens": 6144}
    ]
    assert smoke["workload"]["benchmark_cases"] == [
        {"query_tokens": 4096, "context_tokens": 6144}
    ]
    assert smoke["name"].endswith("_smoke_n6144")


def test_smoke_config_narrows_profiling_and_stays_loadable(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "configs" / "fused_index_topk_h20.json").read_text())
    smoke = build_smoke_config(raw)

    # parse_config rejects profiling cases that are not a subset of the
    # benchmark cases, so the smoke matrix must narrow both together.
    for field in ("nsys_cases", "ncu_cases"):
        assert smoke["profiling"][field] == [{"query_tokens": 4096, "context_tokens": 6144}]

    smoke_path = tmp_path / "smoke-config.json"
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    parsed = load_config(smoke_path)
    assert parsed.name.endswith("_smoke_n6144")
    assert [case.context_tokens for case in parsed.workload.benchmark_cases] == [6144]
    assert [case.context_tokens for case in parsed.workload.correctness_cases] == [6144]


def test_render_report_includes_status_and_failure() -> None:
    report = render_report(
        {
            "status": "failed",
            "run_id": "smoke-test",
            "started_at_utc": "start",
            "completed_at_utc": "end",
            "level": "smoke",
            "stages": [{"name": "preflight", "status": "failed"}],
            "source": {"revision": "abc", "dirty": True},
            "failure": "preflight failed",
        }
    )
    assert "Status: **failed**" in report
    assert "preflight failed" in report
    assert "Dirty worktree: `True`" in report
