"""CPU-only structural validation for exported Nsys and NCU evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty profiler export: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_ncu_csv(
    path: str | Path,
    *,
    requested_metrics: Iterable[str],
    expected_nvtx_label: str,
    stage: str,
) -> dict[str, Any]:
    """Require real kernel rows and finite values for every requested metric."""

    source = Path(path)
    rows = list(csv.reader(source.read_text(encoding="utf-8").splitlines()))
    header_index = next(
        (
            index
            for index, row in enumerate(rows)
            if "Kernel Name" in row and any(name.startswith("gpu__time_duration") for name in row)
        ),
        None,
    )
    if header_index is None:
        raise ValueError("NCU CSV has no raw kernel header")
    header = rows[header_index]
    columns = {name: index for index, name in enumerate(header)}
    required = tuple(dict.fromkeys(str(metric) for metric in requested_metrics))
    missing = [metric for metric in required if metric not in columns]
    if missing:
        raise ValueError(f"NCU CSV is missing requested metrics: {missing}")
    duration_columns = [
        index for name, index in columns.items() if name.startswith("gpu__time_duration")
    ]
    kernel_column = columns["Kernel Name"]
    data_rows = [
        row
        for row in rows[header_index + 1 :]
        if len(row) > kernel_column and row[kernel_column].strip()
    ]
    if not data_rows:
        raise ValueError("NCU CSV contains no kernel rows")
    if not any(
        any(index < len(row) and _finite_number(row[index]) for index in duration_columns)
        for row in data_rows
    ):
        raise ValueError("NCU CSV contains no finite GPU duration")
    metric_counts = {
        metric: sum(
            columns[metric] < len(row) and _finite_number(row[columns[metric]])
            for row in data_rows
        )
        for metric in required
    }
    empty = [metric for metric, count in metric_counts.items() if count == 0]
    if empty:
        raise ValueError(f"NCU CSV has no finite values for requested metrics: {empty}")
    label_matches = sum(expected_nvtx_label in " ".join(row) for row in data_rows)
    if label_matches == 0:
        raise ValueError(f"NCU CSV does not contain NVTX label {expected_nvtx_label!r}")
    return {
        "schema_version": 1,
        "status": "passed",
        "tool": "ncu",
        "stage": stage,
        "expected_nvtx_label": expected_nvtx_label,
        "nvtx_matching_kernel_rows": label_matches,
        "kernel_rows": len(data_rows),
        "unique_kernel_names": sorted({row[kernel_column] for row in data_rows}),
        "requested_metric_finite_value_counts": metric_counts,
        "source": _source_record(source),
    }


def validate_nsys_stats(
    path: str | Path,
    *,
    expected_nvtx_label: str,
    stage: str = "pipeline",
) -> dict[str, Any]:
    """Require the untruncated pipeline label in Nsys CSV stats output."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    matches = text.count(expected_nvtx_label)
    if matches == 0:
        raise ValueError(f"Nsys stats do not contain NVTX label {expected_nvtx_label!r}")
    nonempty_rows = sum(bool(line.strip()) for line in text.splitlines())
    return {
        "schema_version": 1,
        "status": "passed",
        "tool": "nsys",
        "stage": stage,
        "expected_nvtx_label": expected_nvtx_label,
        "nvtx_label_occurrences": matches,
        "nonempty_stats_rows": nonempty_rows,
        "source": _source_record(source),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verification evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="tool", required=True)
    for tool in ("nsys", "ncu"):
        command = subparsers.add_parser(tool)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--expected-nvtx-label", required=True)
        command.add_argument("--stage", required=True)
        command.add_argument("--output", type=Path, required=True)
        if tool == "ncu":
            command.add_argument("--metrics", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.tool == "ncu":
        payload = validate_ncu_csv(
            args.input,
            requested_metrics=(item for item in args.metrics.split(",") if item),
            expected_nvtx_label=args.expected_nvtx_label,
            stage=args.stage,
        )
    else:
        payload = validate_nsys_stats(
            args.input,
            expected_nvtx_label=args.expected_nvtx_label,
            stage=args.stage,
        )
    _write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
