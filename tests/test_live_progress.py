from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIVE_PROGRESS = runpy.run_path(str(ROOT / "scripts" / "live_progress.py"))
_artifact_status = LIVE_PROGRESS["_artifact_status"]
_case_metrics = LIVE_PROGRESS["_case_metrics"]
_case_path = LIVE_PROGRESS["_case_path"]
_command_from_args = LIVE_PROGRESS["_command_from_args"]
_load_cases = LIVE_PROGRESS["_load_cases"]
_run = LIVE_PROGRESS["_run"]


def test_progress_uses_the_frozen_benchmark_order() -> None:
    cases = _load_cases(ROOT / "configs" / "fused_index_topk_h20.json")
    assert cases[0] == (4096, 8192)
    assert cases[-1] == (4096, 163840)
    assert len(cases) == 6


def test_case_path_matches_runner_atomic_layout(tmp_path: Path) -> None:
    artifact = tmp_path / "variant" / "benchmark.json"
    assert _case_path(artifact, 4096, 524288) == (
        tmp_path / "variant" / "benchmark.cases" / "Q4096-N524288.json"
    )


def test_case_metrics_requires_both_operator_totals(tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "summary": [
                    {
                        "pass": "formal_kernel_sum",
                        "scope_id": "operator_total",
                        "stage_id": "operator_total",
                        "count": 30,
                        "median_ms": 1.25,
                        "p95_ms": 1.30,
                    },
                    {
                        "pass": "cuda_event_total",
                        "scope_id": "operator_total",
                        "stage_id": "operator_total",
                        "count": 20,
                        "median_ms": 1.28,
                        "p95_ms": 1.34,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assert _case_metrics(case_path) == {
        "formal_count": 30,
        "formal_median_ms": 1.25,
        "formal_p95_ms": 1.30,
        "event_count": 20,
        "event_median_ms": 1.28,
        "event_p95_ms": 1.34,
    }

    payload = json.loads(case_path.read_text(encoding="utf-8"))
    payload["summary"].pop()
    case_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="complete operator summary"):
        _case_metrics(case_path)


def test_command_and_artifact_status_validation(tmp_path: Path) -> None:
    assert _command_from_args(["--", "python3", "-V"]) == ["python3", "-V"]
    with pytest.raises(ValueError, match="command is required"):
        _command_from_args(["--"])
    artifact = tmp_path / "artifact.json"
    assert _artifact_status(artifact) is None
    artifact.write_text('{"status":"passed"}', encoding="utf-8")
    assert _artifact_status(artifact) == "passed"


def test_progress_runner_observes_atomic_case_and_final_artifact(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {"workload": {"benchmark_cases": [{"query_tokens": 2048, "context_tokens": 4096}]}}
        ),
        encoding="utf-8",
    )
    artifact = tmp_path / "run" / "benchmark.json"
    case_payload = {
        "status": "complete",
        "summary": [
            {
                "pass": "formal_kernel_sum",
                "scope_id": "operator_total",
                "stage_id": "operator_total",
                "count": 30,
                "median_ms": 1.0,
                "p95_ms": 1.1,
            },
            {
                "pass": "cuda_event_total",
                "scope_id": "operator_total",
                "stage_id": "operator_total",
                "count": 20,
                "median_ms": 1.05,
                "p95_ms": 1.15,
            },
        ],
    }
    writer = """
import json
import sys
import time
from pathlib import Path

artifact = Path(sys.argv[1])
case_payload = json.loads(sys.argv[2])
case_dir = artifact.with_name(f"{artifact.stem}.cases")
case_dir.mkdir(parents=True)
(case_dir / "Q2048-N4096.json").write_text(json.dumps(case_payload))
time.sleep(0.3)
artifact.write_text(json.dumps({"status": "complete"}))
"""
    args = SimpleNamespace(
        kind="benchmark",
        label="test benchmark",
        artifact=artifact,
        config=config,
        log=None,
        poll_seconds=0.2,
        command=[sys.executable, "-c", writer, str(artifact), json.dumps(case_payload)],
    )

    assert _run(args) == 0
    assert artifact.with_name("benchmark.run.log").is_file()
