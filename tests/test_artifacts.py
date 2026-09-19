from __future__ import annotations

import json
import math

import pytest

from fused_index_topk.artifacts import (
    canonical_hash,
    canonical_json,
    load_json,
    write_json_atomic,
)


def test_canonical_hash_is_key_order_independent_and_value_sensitive() -> None:
    left = {"b": [2, 3], "a": {"value": 1}}
    right = {"a": {"value": 1}, "b": [2, 3]}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_hash(left) == canonical_hash(right)
    assert canonical_hash(left) != canonical_hash({"a": {"value": 2}, "b": [2, 3]})
    assert len(canonical_hash(left)) == 64


def test_atomic_json_is_sorted_round_trippable_and_has_newline(tmp_path) -> None:
    destination = tmp_path / "nested" / "run.json"
    payload = {"z": 1, "a": {"unicode": "索引"}}
    assert write_json_atomic(destination, payload) == destination
    text = destination.read_text(encoding="utf-8")
    assert text.startswith('{\n  "a"')
    assert text.endswith("\n")
    assert load_json(destination) == payload
    assert not list(destination.parent.glob(".*.tmp"))


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_non_finite_value_never_overwrites_existing_artifact(tmp_path, invalid: float) -> None:
    destination = tmp_path / "run.json"
    write_json_atomic(destination, {"status": "complete"})
    before = destination.read_bytes()
    with pytest.raises(ValueError):
        write_json_atomic(destination, {"latency_ms": invalid})
    assert destination.read_bytes() == before
    assert json.loads(before) == {"status": "complete"}


def test_canonical_hash_rejects_non_json_objects() -> None:
    with pytest.raises(TypeError):
        canonical_hash({"bad": object()})
