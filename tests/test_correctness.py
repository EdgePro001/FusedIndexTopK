from __future__ import annotations

import copy

import pytest

from index_topk_perflab.api import PrefillCase
from index_topk_perflab.correctness import (
    CorrectnessError,
    compare_topk_indices,
    validate_topk_indices,
)


class FakeTensor:
    def __init__(self, data, *, dtype: str = "int32", contiguous: bool = True) -> None:
        self._data = data
        self.dtype = dtype
        self.shape = (len(data), len(data[0]), len(data[0][0]))
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous

    def tolist(self):
        return copy.deepcopy(self._data)


def _padding_case() -> PrefillCase:
    return PrefillCase(
        case_id="small",
        query_tokens=3,
        context_tokens=3,
        top_k=4,
        seed=1,
    )


def _valid_data():
    return [
        [[0, -1, -1, -1]],
        [[1, 0, -1, -1]],
        [[2, 0, 1, -1]],
    ]


def test_validate_causal_padded_int32_output() -> None:
    summary = validate_topk_indices(FakeTensor(_valid_data()), _padding_case())
    assert summary == {
        "status": "passed",
        "rows": 3,
        "top_k": 4,
        "valid_ids": 6,
        "padding_ids": 6,
        "dtype": "int32",
        "contiguous": True,
    }


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (FakeTensor(_valid_data(), dtype="int64"), "dtype"),
        (FakeTensor(_valid_data(), contiguous=False), "contiguous"),
        (
            FakeTensor([[[0, -1, -1, -1]], [[1, 1, -1, -1]], [[2, 0, 1, -1]]]),
            "duplicate",
        ),
        (
            FakeTensor([[[1, -1, -1, -1]], [[1, 0, -1, -1]], [[2, 0, 1, -1]]]),
            "outside",
        ),
        (
            FakeTensor([[[0, -1, -1, -1]], [[1, 0, -1, -1]], [[2, 0, -1, -1]]]),
            "expected 3 valid IDs",
        ),
        (
            FakeTensor([[[0, -2, -1, -1]], [[1, 0, -1, -1]], [[2, 0, 1, -1]]]),
            "invalid negative",
        ),
    ],
)
def test_contract_violations_are_rejected(tensor: FakeTensor, message: str) -> None:
    with pytest.raises(CorrectnessError, match=message):
        validate_topk_indices(tensor, _padding_case())


def test_shape_must_be_q_one_k() -> None:
    tensor = FakeTensor(_valid_data())
    tensor.shape = (1, 3, 4)
    with pytest.raises(CorrectnessError, match="shape"):
        validate_topk_indices(tensor, _padding_case())


def test_compare_accepts_different_order_but_same_sets() -> None:
    baseline = FakeTensor(_valid_data())
    candidate = FakeTensor(
        [
            [[-1, 0, -1, -1]],
            [[-1, 0, 1, -1]],
            [[1, -1, 2, 0]],
        ]
    )
    result = compare_topk_indices(candidate, baseline, _padding_case())
    assert result["comparison"] == "unordered_exact_id_set"


def test_compare_reports_set_difference() -> None:
    case = PrefillCase(
        case_id="set-mismatch",
        query_tokens=1,
        context_tokens=5,
        top_k=3,
        seed=1,
    )
    # query_start=4, so any three unique IDs in [0, 5) satisfy the contract.
    baseline = FakeTensor([[[0, 1, 2]]])
    candidate = FakeTensor([[[0, 1, 3]]])
    with pytest.raises(CorrectnessError, match="missing=.*2.*unexpected=.*3"):
        compare_topk_indices(candidate, baseline, case)


def test_compare_accepts_only_score_threshold_equivalent_tied_ids() -> None:
    case = PrefillCase(
        case_id="cutoff-tie",
        query_tokens=1,
        context_tokens=5,
        top_k=3,
        seed=1,
    )
    baseline = FakeTensor([[[0, 1, 2]]])
    candidate = FakeTensor([[[0, 1, 3]]])
    tie_rows = {
        0: {
            "boundary_ids": (2, 3),
            "required_boundary_selections": 1,
            "strictly_greater_count": 2,
        }
    }

    result = compare_topk_indices(
        candidate,
        baseline,
        case,
        cutoff_tie_rows=tie_rows,
    )

    assert result["comparison"] == "unordered_exact_topk_score_threshold"
    assert result["cutoff_tie_rows_available"] == 1
    assert result["cutoff_tie_rows_substituted"] == 1


def test_compare_rejects_below_cutoff_id_on_tied_row() -> None:
    case = PrefillCase(
        case_id="invalid-cutoff-tie",
        query_tokens=1,
        context_tokens=5,
        top_k=3,
        seed=1,
    )
    baseline = FakeTensor([[[0, 1, 2]]])
    candidate = FakeTensor([[[0, 3, 4]]])
    tie_rows = {
        0: {
            "boundary_ids": (2, 3),
            "required_boundary_selections": 1,
            "strictly_greater_count": 2,
        }
    }

    with pytest.raises(CorrectnessError, match="not an exact score-threshold TopK"):
        compare_topk_indices(
            candidate,
            baseline,
            case,
            cutoff_tie_rows=tie_rows,
        )
