"""CPU-only measurement validation, aggregation, and table I/O."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifacts import write_json_atomic

MEASUREMENT_COLUMNS = (
    "variant_id",
    "case_id",
    "query_tokens",
    "context_tokens",
    "top_k",
    "pass",
    "trial_id",
    "scope_id",
    "stage_id",
    "semantic_ops",
    "timing_source",
    "latency_ms",
    "derived",
)

DEFAULT_GROUP_KEYS = (
    "variant_id",
    "case_id",
    "query_tokens",
    "context_tokens",
    "top_k",
    "pass",
    "scope_id",
    "stage_id",
    "semantic_ops",
    "timing_source",
    "derived",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"cannot parse boolean measurement value: {value!r}")


def normalize_measurement(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and coerce one long-table measurement row."""

    missing = set(MEASUREMENT_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"measurement row is missing columns: {sorted(missing)}")
    result = dict(row)
    for key in ("query_tokens", "context_tokens", "top_k", "trial_id"):
        result[key] = int(result[key])
    result["latency_ms"] = float(result["latency_ms"])
    result["derived"] = _as_bool(result["derived"])
    if not math.isfinite(result["latency_ms"]) or result["latency_ms"] < 0:
        raise ValueError("latency_ms must be finite and non-negative")
    if result["trial_id"] < 0:
        raise ValueError("trial_id must be non-negative")

    rules = {
        "formal_kernel_sum": {
            "scope_id": "operator_total",
            "stage_id": "operator_total",
            "timing_source": "kineto_cupti",
            "derived": False,
        },
        "kineto_activity_sum": {
            "scope_id": "operator_total",
            "stage_id": "operator_total",
            "timing_source": "kineto_cupti",
            "derived": False,
        },
        "kineto_device_span": {
            "scope_id": "operator_total",
            "stage_id": "operator_total",
            "timing_source": "kineto_cupti",
            "derived": False,
        },
        "cuda_event_total": {
            "scope_id": "operator_total",
            "stage_id": "operator_total",
            "timing_source": "direct_cuda_event",
            "derived": False,
        },
        "kineto_stage_kernel_sum": {
            "scope_id": "stage",
            "timing_source": "kineto_cupti",
            "derived": False,
        },
        # Historical R1/R2 artifacts remain readable but are never mixed with
        # the schema-v4 CUPTI formal metric.
        "clean_total": {
            "scope_id": "discovery_total",
            "stage_id": "discovery_total",
            "timing_source": "direct_cuda_event",
            "derived": False,
        },
        "attribution": {
            "scope_id": "stage",
            "timing_source": "instrumented_cuda_event",
            "derived": False,
        },
    }
    expected = rules.get(str(result["pass"]))
    if expected is None:
        raise ValueError(f"unknown measurement pass: {result['pass']!r}")
    if result["latency_ms"] <= 0:
        raise ValueError(f"{result['pass']} latency_ms must be strictly positive")
    mismatches = {
        key: (result[key], value)
        for key, value in expected.items()
        if result[key] != value
    }
    if mismatches:
        raise ValueError(f"invalid {result['pass']} row: {mismatches}")
    return result


def validate_measurements(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_measurement(row) for row in rows]


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("quantile fraction must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_measurements(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_keys: Sequence[str] = DEFAULT_GROUP_KEYS,
) -> list[dict[str, Any]]:
    """Aggregate long-table rows without importing torch, pandas, or CUDA."""

    normalized = validate_measurements(rows)
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in normalized:
        try:
            key = tuple(row[column] for column in group_keys)
        except KeyError as error:
            raise ValueError(f"unknown summary group key: {error.args[0]!r}") from error
        groups[key].append(row["latency_ms"])

    result: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        values = sorted(groups[key])
        median = float(statistics.median(values))
        mad = float(statistics.median(abs(value - median) for value in values))
        summary = dict(zip(group_keys, key, strict=True))
        summary.update(
            {
                "count": len(values),
                "min_ms": float(values[0]),
                "p10_ms": _quantile(values, 0.10),
                "p25_ms": _quantile(values, 0.25),
                "median_ms": median,
                "mean_ms": float(statistics.fmean(values)),
                "p75_ms": _quantile(values, 0.75),
                "p90_ms": _quantile(values, 0.90),
                "p95_ms": _quantile(values, 0.95),
                "max_ms": float(values[-1]),
                "pstdev_ms": float(statistics.pstdev(values)),
                "mad_ms": mad,
                "normalized_mad": mad / median if median else 0.0,
            }
        )
        result.append(summary)
    return result


def read_measurements(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON, JSONL, or CSV long table and normalize its rows."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    elif suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            if "measurements" not in payload:
                raise ValueError("JSON object does not contain a measurements member")
            rows = payload["measurements"]
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError("measurement JSON must be an array or run-result object")
    elif suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unsupported measurement table extension: {suffix!r}")
    return validate_measurements(rows)


def write_measurements(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write normalized rows as JSONL or CSV based on ``path`` suffix."""

    destination = Path(path)
    normalized = validate_measurements(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() not in {".jsonl", ".csv"}:
        raise ValueError("write_measurements supports only .jsonl and .csv")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            if destination.suffix.lower() == ".jsonl":
                for row in normalized:
                    handle.write(
                        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                    )
            else:
                extras = sorted(
                    {key for row in normalized for key in row}
                    - set(MEASUREMENT_COLUMNS)
                )
                fieldnames = [*MEASUREMENT_COLUMNS, *extras]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_summary(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    payload = summarize_measurements(rows)
    write_json_atomic(destination, payload)
    return destination


def write_run_tables(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Persist the long table and CPU-readable summary for one benchmark run."""

    if "measurements" not in result:
        raise ValueError("benchmark result does not contain measurements")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rows = validate_measurements(result["measurements"])
    jsonl_path = write_measurements(directory / "measurements.jsonl", rows)
    csv_path = write_measurements(directory / "measurements.csv", rows)
    summary_path = write_summary(directory / "summary.json", rows)

    manifest = {
        key: value
        for key, value in result.items()
        if key not in {"measurements", "summary"}
    }
    manifest_path = directory / "run.json"
    write_json_atomic(manifest_path, manifest)
    return {
        "measurements_jsonl": jsonl_path,
        "measurements_csv": csv_path,
        "summary": summary_path,
        "run": manifest_path,
    }
