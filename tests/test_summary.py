from __future__ import annotations

import json

import pytest

from index_topk_perflab.summary import (
    read_measurements,
    summarize_measurements,
    validate_measurements,
    write_measurements,
    write_run_tables,
)


def _row(
    latency_ms: float,
    *,
    trial_id: int,
    pass_name: str = "formal_kernel_sum",
    stage_id: str = "operator_total",
) -> dict[str, object]:
    stage = pass_name == "kineto_stage_kernel_sum"
    event = pass_name == "cuda_event_total"
    return {
        "variant_id": "baseline",
        "case_id": "prefill-8k",
        "query_tokens": 4096,
        "context_tokens": 8192,
        "top_k": 2048,
        "pass": pass_name,
        "trial_id": trial_id,
        "scope_id": "stage" if stage else "operator_total",
        "stage_id": stage_id,
        "semantic_ops": stage_id,
        "timing_source": "direct_cuda_event" if event else "kineto_cupti",
        "latency_ms": latency_ms,
        "derived": False,
    }


def test_summary_keeps_formal_and_stage_diagnostic_separate() -> None:
    rows = [
        _row(1.0, trial_id=0),
        _row(3.0, trial_id=1),
        _row(0.5, trial_id=0, pass_name="kineto_stage_kernel_sum", stage_id="indexer"),
        _row(0.7, trial_id=1, pass_name="kineto_stage_kernel_sum", stage_id="indexer"),
    ]
    summary = summarize_measurements(rows)
    assert len(summary) == 2
    formal = next(item for item in summary if item["pass"] == "formal_kernel_sum")
    stage = next(item for item in summary if item["pass"] == "kineto_stage_kernel_sum")
    assert formal["count"] == 2
    assert formal["median_ms"] == pytest.approx(2.0)
    assert formal["p25_ms"] == pytest.approx(1.5)
    assert formal["pstdev_ms"] == pytest.approx(1.0)
    assert stage["median_ms"] == pytest.approx(0.6)


def test_jsonl_and_csv_round_trip(tmp_path) -> None:
    rows = [_row(1.25, trial_id=0), _row(1.5, trial_id=1)]
    jsonl = write_measurements(tmp_path / "rows.jsonl", rows)
    csv_path = write_measurements(tmp_path / "rows.csv", rows)
    assert read_measurements(jsonl) == validate_measurements(rows)
    assert read_measurements(csv_path) == validate_measurements(rows)


def test_read_measurements_accepts_run_result_json(tmp_path) -> None:
    path = tmp_path / "run-result.json"
    path.write_text(json.dumps({"measurements": [_row(2.0, trial_id=0)]}), encoding="utf-8")
    assert read_measurements(path)[0]["latency_ms"] == 2.0


def test_invalid_formal_row_is_rejected() -> None:
    row = _row(1.0, trial_id=0)
    row["derived"] = True
    with pytest.raises(ValueError, match="invalid formal"):
        validate_measurements([row])


def test_write_run_tables_emits_long_tables_and_summary(tmp_path) -> None:
    rows = [_row(1.0, trial_id=0), _row(2.0, trial_id=1)]
    paths = write_run_tables(
        {
            "schema_version": 1,
            "run_id": "unit-test",
            "measurements": rows,
            "summary": summarize_measurements(rows),
        },
        tmp_path,
    )
    assert set(paths) == {"measurements_jsonl", "measurements_csv", "summary", "run"}
    assert all(path.exists() for path in paths.values())
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary[0]["median_ms"] == pytest.approx(1.5)
