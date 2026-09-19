from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from fused_index_topk.kineto import (
    extract_active_trials,
    extract_range_timing,
    operator_range_name,
    stage_range_name,
    validate_trial_topology,
)


@dataclass
class _TimeRange:
    start: float
    end: float


@dataclass
class _Event:
    name: str
    device_type: str
    id: int
    start: float
    end: float
    cpu_children: list["_Event"] = field(default_factory=list)

    @property
    def time_range(self) -> _TimeRange:
        return _TimeRange(self.start, self.end)


def _profile_events() -> list[_Event]:
    launch_a = _Event("cudaLaunchKernel", "CPU", 101, 1.0, 1.1)
    launch_b = _Event("cudaLaunchKernel", "CPU", 102, 1.2, 1.3)
    stage = _Event(
        stage_range_name(1, 0, "indexer"),
        "CPU",
        11,
        0.5,
        1.5,
        [launch_a, launch_b],
    )
    operator = _Event(
        operator_range_name(1, 0),
        "CPU",
        10,
        0.0,
        2.0,
        [stage],
    )
    return [
        operator,
        stage,
        _Event(operator.name, "CUDA", 10, 0.0, 100.0),
        _Event(stage.name, "CUDA", 11, 0.5, 90.0),
        _Event("scorer", "CUDA", 101, 10.0, 30.0),
        _Event("topk", "CUDA", 102, 32.0, 42.0),
        _Event("unrelated", "CUDA", 999, 5.0, 500.0),
    ]


def test_extract_range_uses_correlation_and_excludes_range_annotations() -> None:
    timing = extract_range_timing(_profile_events(), operator_range_name(1, 0))
    assert timing.kernel_count == 2
    assert timing.kernel_sum_us == pytest.approx(30.0)
    assert timing.activity_sum_us == pytest.approx(30.0)
    assert timing.device_span_us == pytest.approx(32.0)
    assert timing.gap_us == pytest.approx(2.0)
    assert [item.name for item in timing.activities] == ["scorer", "topk"]


def test_extract_range_rejects_cpu_event_id_collisions() -> None:
    launch = _Event("cudaLaunchKernel", "CPU", 201, 1.1, 1.2)
    ordinary_cpu_op = _Event("aten::mul", "CPU", 101, 1.0, 1.3, [launch])
    operator = _Event(
        operator_range_name(1, 0),
        "CPU",
        10,
        0.0,
        2.0,
        [ordinary_cpu_op],
    )
    events = [
        operator,
        ordinary_cpu_op,
        launch,
        # ID 101 collides with an ordinary CPU event.  This models the
        # out-of-range L2 scrub that was observed on the H20 trace.
        _Event("l2_scrub", "CUDA", 101, 2.0, 500.0),
        _Event("operator_kernel", "CUDA", 201, 510.0, 530.0),
    ]
    timing = extract_range_timing(events, operator.name)
    assert timing.kernel_count == 1
    assert timing.kernel_sum_us == pytest.approx(20.0)
    assert [item.name for item in timing.activities] == ["operator_kernel"]


def test_extract_active_trials_returns_operator_and_stage_records() -> None:
    rows = extract_active_trials(
        _profile_events(),
        trials=1,
        stage_ids=("indexer",),
    )
    assert rows[0]["operator"]["kernel_sum_us"] == pytest.approx(30.0)
    assert rows[0]["stages"]["indexer"]["kernel_count"] == 2


def test_topology_gate_requires_stable_complete_stage_coverage() -> None:
    rows = extract_active_trials(
        _profile_events(),
        trials=1,
        stage_ids=("indexer",),
    )
    gate = validate_trial_topology(rows, stage_ids=("indexer",))
    assert gate["status"] == "passed"
    assert gate["operator_kernel_count"] == 2
    assert gate["stage_kernel_counts"] == {"indexer": 2}

    unstable = [rows[0], {**rows[0], "operator": {**rows[0]["operator"]}}]
    unstable[1]["operator"]["activities"] = list(
        reversed(unstable[1]["operator"]["activities"])
    )
    with pytest.raises(ValueError, match="topology changed"):
        validate_trial_topology(unstable, stage_ids=("indexer",))

    incomplete = [{**rows[0], "stages": {"indexer": {**rows[0]["stages"]["indexer"]}}}]
    incomplete[0]["stages"]["indexer"]["kernel_count"] = 1
    with pytest.raises(ValueError, match="kernel coverage"):
        validate_trial_topology(incomplete, stage_ids=("indexer",))


def test_extract_range_rejects_missing_or_empty_ranges() -> None:
    with pytest.raises(ValueError, match="expected one"):
        extract_range_timing([], operator_range_name(1, 0))

    empty = _Event(operator_range_name(1, 0), "CPU", 1, 0.0, 1.0)
    with pytest.raises(ValueError, match="no correlated"):
        extract_range_timing([empty], operator_range_name(1, 0))
