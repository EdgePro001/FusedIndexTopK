from __future__ import annotations

from fused_index_topk.inputs import input_stream_seeds


def test_rng_streams_are_stable_independent_and_fixture_specific() -> None:
    first = input_stream_seeds(20260803, "A")
    assert first == input_stream_seeds(20260803, "A")
    assert set(first) == {"kv", "q", "weights"}
    assert len(set(first.values())) == 3
    assert first != input_stream_seeds(20260803, "B")


def test_rng_fixture_id_is_strict() -> None:
    try:
        input_stream_seeds(1, "C")
    except ValueError as error:
        assert "fixture_id" in str(error)
    else:
        raise AssertionError("invalid fixture ID was accepted")
