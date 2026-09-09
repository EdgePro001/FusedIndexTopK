"""DeepGEMM-compatible Kineto/CUPTI activity attribution.

DeepGEMM's ``bench_kineto`` profiles CUDA activities after an 8 GB memset and
reports name-matched kernel averages.  An IndexTopK operator may launch many
kernels, so this module keeps the same collection model while attributing all
CUDA activities to one explicit operator or stage range through Kineto
correlation IDs.  The parsing code deliberately has no torch import so it can
be unit-tested on CPU-only machines.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

KINETO_RANGE_PREFIX = "ITK::KINETO::"
KINETO_OPERATOR_PREFIX = f"{KINETO_RANGE_PREFIX}OPERATOR::"
KINETO_STAGE_PREFIX = f"{KINETO_RANGE_PREFIX}STAGE::"


def operator_range_name(phase: int, trial_id: int) -> str:
    return f"{KINETO_OPERATOR_PREFIX}P{phase}::T{trial_id:04d}"


def stage_range_name(phase: int, trial_id: int, stage_id: str) -> str:
    return f"{KINETO_STAGE_PREFIX}P{phase}::T{trial_id:04d}::{stage_id}"


def _device_type_name(event: Any) -> str:
    return str(getattr(event, "device_type", "")).rsplit(".", maxsplit=1)[-1]


def _event_id(event: Any) -> int | None:
    value = getattr(event, "id", None)
    return int(value) if value is not None else None


def _time_range(event: Any) -> tuple[float, float]:
    value = getattr(event, "time_range", None)
    if value is None:
        raise ValueError(f"Kineto event {getattr(event, 'name', '<unknown>')!r} has no time range")
    start = float(value.start)
    end = float(value.end)
    if end < start:
        raise ValueError("Kineto event has a negative duration")
    return start, end


_CUDA_RUNTIME_ACTIVITY_PREFIXES = (
    "cudaLaunch",
    "cuLaunch",
    "cudaGraphLaunch",
    "cuGraphLaunch",
    "cudaMemcpy",
    "cuMemcpy",
    "cudaMemset",
    "cuMemset",
    "cudaMemPrefetch",
    "cuMemPrefetch",
)


def _is_cuda_runtime_activity(event: Any) -> bool:
    if _device_type_name(event) != "CPU":
        return False
    name = str(getattr(event, "name", ""))
    return name.startswith(_CUDA_RUNTIME_ACTIVITY_PREFIXES)


def _correlated_cuda_ids(root: Any) -> set[int]:
    """Collect only CUDA runtime/driver correlation IDs below a CPU range.

    FunctionEvent ``id`` values for ordinary CPU operators share a numeric
    namespace with CUPTI correlation IDs.  Treating every descendant ID as a
    CUDA correlation can therefore pull an unrelated kernel (notably the L2
    scrub immediately before a timed range) into the operator.  CUDA runtime
    and driver launch/memory calls are the actual correlation boundary.
    """

    result: set[int] = set()
    stack = list(getattr(root, "cpu_children", ()) or ())
    while stack:
        child = stack.pop()
        if _is_cuda_runtime_activity(child):
            identifier = _event_id(child)
            if identifier is not None:
                result.add(identifier)
        stack.extend(getattr(child, "cpu_children", ()) or ())
    return result


def _is_memory_activity(name: str) -> bool:
    normalized = name.lower()
    return normalized.startswith("[cuda memcpy") or normalized.startswith("[cuda memset")


@dataclass(frozen=True)
class ActivityRecord:
    name: str
    start_us: float
    end_us: float
    duration_us: float
    kind: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_us": self.start_us,
            "end_us": self.end_us,
            "duration_us": self.duration_us,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class RangeTiming:
    name: str
    kernel_sum_us: float
    activity_sum_us: float
    device_span_us: float
    gap_us: float
    kernel_count: int
    activity_count: int
    activities: tuple[ActivityRecord, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "range": self.name,
            "kernel_sum_us": self.kernel_sum_us,
            "activity_sum_us": self.activity_sum_us,
            "device_span_us": self.device_span_us,
            "gap_us": self.gap_us,
            "kernel_count": self.kernel_count,
            "activity_count": self.activity_count,
            "activities": [item.as_mapping() for item in self.activities],
        }


def extract_range_timing(events: Sequence[Any], range_name: str) -> RangeTiming:
    """Return all CUDA activities correlated to one CPU ``record_function`` range."""

    ranges = [
        event
        for event in events
        if getattr(event, "name", None) == range_name and _device_type_name(event) == "CPU"
    ]
    if len(ranges) != 1:
        raise ValueError(
            f"expected one CPU Kineto range {range_name!r}, found {len(ranges)}"
        )
    correlated_ids = _correlated_cuda_ids(ranges[0])
    if not correlated_ids:
        raise ValueError(f"Kineto range {range_name!r} contains no correlated CUDA launches")

    records: list[ActivityRecord] = []
    seen: set[tuple[int | None, str, float, float]] = set()
    for event in events:
        if _device_type_name(event) != "CUDA" or _event_id(event) not in correlated_ids:
            continue
        name = str(getattr(event, "name", ""))
        if not name or name.startswith(KINETO_RANGE_PREFIX):
            continue
        start, end = _time_range(event)
        if end == start:
            continue
        identity = (_event_id(event), name, start, end)
        if identity in seen:
            continue
        seen.add(identity)
        records.append(
            ActivityRecord(
                name=name,
                start_us=start,
                end_us=end,
                duration_us=end - start,
                kind="memory" if _is_memory_activity(name) else "kernel",
            )
        )
    records.sort(key=lambda item: (item.start_us, item.end_us, item.name))
    if not records:
        raise ValueError(f"Kineto range {range_name!r} has no CUDA activity records")

    activity_sum = sum(item.duration_us for item in records)
    kernel_records = [item for item in records if item.kind == "kernel"]
    kernel_sum = sum(item.duration_us for item in kernel_records)
    device_span = max(item.end_us for item in records) - min(item.start_us for item in records)
    return RangeTiming(
        name=range_name,
        kernel_sum_us=kernel_sum,
        activity_sum_us=activity_sum,
        device_span_us=device_span,
        gap_us=max(0.0, device_span - activity_sum),
        kernel_count=len(kernel_records),
        activity_count=len(records),
        activities=tuple(records),
    )


def extract_active_trials(
    events: Iterable[Any],
    *,
    trials: int,
    stage_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Extract active-phase operator and stage timings from a Kineto profile."""

    materialized = list(events)
    result: list[dict[str, Any]] = []
    for trial_id in range(trials):
        operator = extract_range_timing(
            materialized,
            operator_range_name(1, trial_id),
        )
        stages = {
            stage_id: extract_range_timing(
                materialized,
                stage_range_name(1, trial_id, stage_id),
            ).as_mapping()
            for stage_id in stage_ids
        }
        result.append(
            {
                "trial_id": trial_id,
                "operator": operator.as_mapping(),
                "stages": stages,
            }
        )
    return result


def validate_trial_topology(
    trials: Sequence[dict[str, Any]],
    *,
    stage_ids: Sequence[str],
) -> dict[str, Any]:
    """Reject unstable or incomplete CUDA activity attribution.

    All fixtures have identical shapes, so a formal case must launch the same
    CUDA activity sequence on every active profiler trial.  Requiring the
    operator record to equal the union of its stage records also catches an
    incorrectly attributed scrub or an unlabelled stage.
    """

    if not trials:
        raise ValueError("Kineto topology gate requires at least one trial")

    def signature(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(activity["kind"]), str(activity["name"]))
            for activity in record["activities"]
        )

    expected_operator = signature(trials[0]["operator"])
    if not expected_operator:
        raise ValueError("Kineto operator topology is empty")
    expected_stages = {
        stage_id: signature(trials[0]["stages"][stage_id]) for stage_id in stage_ids
    }
    for trial in trials:
        trial_id = int(trial["trial_id"])
        operator = trial["operator"]
        if signature(operator) != expected_operator:
            raise ValueError(f"Kineto operator topology changed at trial {trial_id}")
        for stage_id in stage_ids:
            if signature(trial["stages"][stage_id]) != expected_stages[stage_id]:
                raise ValueError(
                    f"Kineto stage {stage_id!r} topology changed at trial {trial_id}"
                )

        stages = [trial["stages"][stage_id] for stage_id in stage_ids]
        if sum(int(stage["kernel_count"]) for stage in stages) != int(
            operator["kernel_count"]
        ):
            raise ValueError(f"Kineto stage kernel coverage failed at trial {trial_id}")
        if sum(int(stage["activity_count"]) for stage in stages) != int(
            operator["activity_count"]
        ):
            raise ValueError(f"Kineto stage activity coverage failed at trial {trial_id}")
        if abs(
            sum(float(stage["kernel_sum_us"]) for stage in stages)
            - float(operator["kernel_sum_us"])
        ) > 1e-6:
            raise ValueError(f"Kineto stage kernel timing coverage failed at trial {trial_id}")
        if abs(
            sum(float(stage["activity_sum_us"]) for stage in stages)
            - float(operator["activity_sum_us"])
        ) > 1e-6:
            raise ValueError(f"Kineto stage activity timing coverage failed at trial {trial_id}")

    serialized = json.dumps(
        expected_operator,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "status": "passed",
        "trial_count": len(trials),
        "operator_kernel_count": int(trials[0]["operator"]["kernel_count"]),
        "operator_activity_count": int(trials[0]["operator"]["activity_count"]),
        "operator_activity_signature_sha256": hashlib.sha256(serialized).hexdigest(),
        "stage_kernel_counts": {
            stage_id: int(trials[0]["stages"][stage_id]["kernel_count"])
            for stage_id in stage_ids
        },
        "stage_activity_counts": {
            stage_id: int(trials[0]["stages"][stage_id]["activity_count"])
            for stage_id in stage_ids
        },
    }
