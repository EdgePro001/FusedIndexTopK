#!/usr/bin/env python3
"""Audit FusedIndexTopK release fault-injection results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from index_topk_perflab.artifacts import canonical_hash, load_json, write_json_atomic

EXPECTED = {
    "natural_layer0_normal_a": {
        "fast_failure_rows": 0,
        "determinism_runs": 20,
        "zero_weights": False,
    },
    "natural_layer60_hard_b": {
        "fast_failure_rows": 0,
        "determinism_runs": 20,
        "zero_weights": False,
    },
    "forced_underflow_all_rows": {
        "fast_failure_rows": 4096,
        "forced_underflow_rows": 4096,
        "determinism_runs": 2,
        "zero_weights": False,
    },
    "forced_working_overflow_all_rows": {
        "fast_failure_rows": 4096,
        "forced_working_overflow_rows": 4096,
        "determinism_runs": 2,
        "zero_weights": False,
    },
    "all_equal_scores": {
        "fast_failure_rows": 4096,
        "determinism_runs": 20,
        "zero_weights": True,
    },
}


def _audit_case(name: str, result: dict[str, Any], expected: dict[str, Any]) -> None:
    if result.get("operator") != "fused_index_topk":
        raise ValueError(f"{name} did not test FusedIndexTopK")
    for key in (
        "failure_rows",
        "value_mismatch_rows",
        "duplicate_output_rows",
        "semantic_mismatch_runs",
    ):
        if int(result.get(key, -1)) != 0:
            raise ValueError(f"{name} failed {key}: {result.get(key)}")
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"{name} expected {key}={value!r}, received {result.get(key)!r}"
            )
    source = result.get("final_variant_source_identity", {})
    if canonical_hash(source.get("payload")) != source.get("sha256"):
        raise ValueError(f"{name} has an invalid final source identity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    results: dict[str, Any] = {}
    source_hashes: set[str] = set()
    for name, expected in EXPECTED.items():
        path = root / f"{name}.json"
        result = load_json(path)
        _audit_case(name, result, expected)
        source_hashes.add(result["final_variant_source_identity"]["sha256"])
        results[name] = {
            "path": str(path),
            "fast_failure_rows": result["fast_failure_rows"],
            "repaired_failure_rows": result["failure_rows"],
            "candidate_count_min": result["candidate_count_min"],
            "candidate_count_max": result["candidate_count_max"],
            "determinism_runs": result["determinism_runs"],
            "bitwise_order_change_runs": result["bitwise_order_change_runs"],
            "unordered_set_change_runs": result["unordered_set_change_runs"],
            "semantic_mismatch_runs": result["semantic_mismatch_runs"],
            "value_mismatch_rows": result["value_mismatch_rows"],
            "duplicate_output_rows": result["duplicate_output_rows"],
            "zero_weights": result["zero_weights"],
        }
    if len(source_hashes) != 1:
        raise ValueError("fault-injection cases did not use one final source identity")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "method": "FusedIndexTopK path-specific fault injection and repeated exactness",
        "case_count": len(results),
        "all_semantic_exactness_gates_passed": True,
        "final_variant_source_sha256": next(iter(source_hashes)),
        "cases": results,
        "interpretation": (
            "ID order and equal-cutoff ID choice may vary under the declared "
            "unordered score-threshold contract; every repetition remained exact"
        ),
    }
    write_json_atomic(args.output.resolve(), payload)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
