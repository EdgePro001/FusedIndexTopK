"""CPU-side validation for the canonical exact causal TopK output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .api import PrefillCase


class CorrectnessError(AssertionError):
    """Raised when a plugin result violates the shared output contract."""


def build_score_threshold_oracle(
    scores: object,
    baseline_indices: object,
    case: PrefillCase,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Build a compact exact-TopK oracle for rows with cutoff-score ties.

    An exact TopK is not a unique ID set when several legal keys have the K-th
    score.  The protocol therefore requires every ID above the cutoff and the
    necessary number of IDs equal to the cutoff; IDs below it remain invalid.
    Rows without a cutoff tie still use exact unordered ID-set comparison.

    Only tied rows retain their boundary IDs.  The full ``[Q, N]`` score tensor
    can consequently be released before candidate preparation, including at
    the 2M-token formal shape.
    """

    import torch

    expected_shape = (case.query_tokens, case.context_tokens)
    if not isinstance(scores, torch.Tensor):
        raise CorrectnessError("baseline scores must be a torch.Tensor")
    if tuple(scores.shape) != expected_shape:
        raise CorrectnessError(
            f"baseline scores shape must be {expected_shape}, received {tuple(scores.shape)}"
        )
    if scores.dtype != torch.float32:
        raise CorrectnessError("baseline scores must use float32")
    validate_topk_indices(baseline_indices, case, name="baseline cutoff oracle")

    valid_counts = torch.arange(
        case.query_start + 1,
        case.context_tokens + 1,
        device=scores.device,
        dtype=torch.int64,
    )
    eligible = valid_counts > case.top_k
    rows_checked = int(eligible.sum().item())
    rows_skipped = case.query_tokens - rows_checked
    if rows_checked == 0:
        return (
            {
                "status": "passed",
                "policy": "exact_score_threshold",
                "rows_checked": 0,
                "reference_selection_rows_validated": 0,
                "rows_skipped_valid_count_le_k": rows_skipped,
                "rows_with_cutoff_ties": 0,
                "minimum_cutoff_margin": None,
                "minimum_positive_cutoff_margin": None,
                "tie_rows": [],
            },
            {},
        )
    if case.context_tokens <= case.top_k:
        raise CorrectnessError("eligible cutoff rows require N > K")

    top_values = torch.topk(
        scores,
        case.top_k + 1,
        dim=-1,
        largest=True,
        sorted=True,
    ).values[eligible]
    boundary = top_values[:, case.top_k - 1 : case.top_k + 1]
    if not bool(torch.isfinite(boundary).all().item()):
        raise CorrectnessError("eligible TopK cutoff scores must be finite")
    margins = boundary[:, 0] - boundary[:, 1]
    tied = margins == 0
    if bool((margins < 0).any().item()):
        raise CorrectnessError("TopK cutoff values are not monotonically ordered")
    baseline_rows = baseline_indices[:, 0, :]
    eligible_rows = eligible.nonzero(as_tuple=False).flatten()
    selected_ids = baseline_rows[eligible].to(torch.int64)
    selected_scores = scores[eligible].gather(1, selected_ids)
    cutoff_scores = boundary[:, 0]
    below_cutoff = selected_scores < cutoff_scores[:, None]
    if bool(below_cutoff.any().item()):
        position = int(below_cutoff.nonzero(as_tuple=False)[0, 0].item())
        row = int(eligible_rows[position].item())
        raise CorrectnessError(
            f"exact reference row {row} selected an ID below the TopK cutoff"
        )
    expected_greater = (top_values[:, : case.top_k] > cutoff_scores[:, None]).sum(dim=1)
    selected_greater = (selected_scores > cutoff_scores[:, None]).sum(dim=1)
    wrong_greater_count = selected_greater != expected_greater
    if bool(wrong_greater_count.any().item()):
        position = int(wrong_greater_count.nonzero(as_tuple=False)[0, 0].item())
        row = int(eligible_rows[position].item())
        raise CorrectnessError(
            f"exact reference row {row} omitted an ID above the TopK cutoff"
        )

    tie_rows: dict[int, dict[str, Any]] = {}
    tie_summaries: list[dict[str, Any]] = []
    for tied_position in tied.nonzero(as_tuple=False).flatten().tolist():
        row = int(eligible_rows[tied_position].item())
        causal_end = int(valid_counts[row].item())
        cutoff_score = boundary[tied_position, 0]
        boundary_ids = (
            (scores[row, :causal_end] == cutoff_score)
            .nonzero(as_tuple=False)
            .flatten()
            .to(device="cpu", dtype=torch.int64)
            .tolist()
        )
        selected = {
            int(value)
            for value in baseline_rows[row].to(device="cpu", dtype=torch.int64).tolist()
            if int(value) >= 0
        }
        boundary_set = set(boundary_ids)
        selected_boundary = selected & boundary_set
        strictly_greater = int((scores[row, :causal_end] > cutoff_score).sum().item())
        required_boundary = case.top_k - strictly_greater
        if len(selected) != case.top_k:
            raise CorrectnessError(f"baseline cutoff oracle row {row} has invalid width")
        if len(selected - boundary_set) != strictly_greater:
            raise CorrectnessError(
                f"baseline cutoff oracle row {row} omitted a score above the cutoff"
            )
        if len(selected_boundary) != required_boundary:
            raise CorrectnessError(
                f"baseline cutoff oracle row {row} selected an invalid number of tied IDs"
            )
        if len(boundary_ids) <= required_boundary:
            raise CorrectnessError(
                f"baseline cutoff oracle row {row} was marked tied without alternatives"
            )
        tie_rows[row] = {
            "boundary_ids": tuple(int(value) for value in boundary_ids),
            "required_boundary_selections": required_boundary,
            "strictly_greater_count": strictly_greater,
        }
        tie_summaries.append(
            {
                "row": row,
                "causal_end": causal_end,
                "cutoff_score": float(cutoff_score.item()),
                "boundary_candidate_count": len(boundary_ids),
                "required_boundary_selections": required_boundary,
                "strictly_greater_count": strictly_greater,
            }
        )
    positive = margins[margins > 0]
    summary = {
        "status": "passed",
        "policy": "exact_score_threshold",
        "rows_checked": rows_checked,
        "reference_selection_rows_validated": rows_checked,
        "rows_skipped_valid_count_le_k": rows_skipped,
        "minimum_cutoff_margin": float(margins.min().item()),
        "minimum_positive_cutoff_margin": (
            float(positive.min().item()) if positive.numel() else None
        ),
        "rows_with_cutoff_ties": len(tie_rows),
        "tie_rows": tie_summaries,
    }
    return summary, tie_rows


def _shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise CorrectnessError("indices must expose a shape attribute")
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as error:
        raise CorrectnessError("indices.shape must contain integer dimensions") from error


def _dtype_name(value: object) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise CorrectnessError("indices must expose a dtype attribute")
    return str(dtype).lower().split(".")[-1]


def _is_contiguous(value: object) -> bool:
    method = getattr(value, "is_contiguous", None)
    if callable(method):
        return bool(method())
    flags = getattr(value, "flags", None)
    if flags is not None and hasattr(flags, "c_contiguous"):
        return bool(flags.c_contiguous)
    raise CorrectnessError("indices must expose is_contiguous() or C-contiguous flags")


def _rows(value: object) -> list[list[list[int]]]:
    materialized: Any = value
    detach = getattr(materialized, "detach", None)
    if callable(detach):
        materialized = detach()
    cpu = getattr(materialized, "cpu", None)
    if callable(cpu):
        materialized = cpu()
    tolist = getattr(materialized, "tolist", None)
    if not callable(tolist):
        raise CorrectnessError("indices must expose tolist()")
    data = tolist()
    if not isinstance(data, list):
        raise CorrectnessError("indices.tolist() must return a nested list")
    return data


def _validated_sets(indices: object, case: PrefillCase, *, name: str) -> list[frozenset[int]]:
    if not case.causal:
        raise CorrectnessError("the current correctness contract requires a causal Prefill case")
    expected_shape = (case.query_tokens, 1, case.top_k)
    actual_shape = _shape(indices)
    if actual_shape != expected_shape:
        raise CorrectnessError(
            f"{name} shape must be {expected_shape}, received {actual_shape}"
        )
    if _dtype_name(indices) != "int32":
        raise CorrectnessError(f"{name} dtype must be int32")
    if not _is_contiguous(indices):
        raise CorrectnessError(f"{name} must be contiguous")

    nested = _rows(indices)
    if len(nested) != case.query_tokens:
        raise CorrectnessError(f"{name}.tolist() is inconsistent with its shape")
    selected_sets: list[frozenset[int]] = []
    for row_index, outer_row in enumerate(nested):
        if not isinstance(outer_row, list) or len(outer_row) != 1:
            raise CorrectnessError(f"{name} row {row_index} must contain one index group")
        row = outer_row[0]
        if not isinstance(row, list) or len(row) != case.top_k:
            raise CorrectnessError(f"{name} row {row_index} has an invalid materialized width")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, int):
                raise CorrectnessError(f"{name} row {row_index} contains a non-integer ID")

        causal_end = case.query_start + row_index + 1
        invalid_sentinels = [value for value in row if value < 0 and value != -1]
        if invalid_sentinels:
            raise CorrectnessError(
                f"{name} row {row_index} contains invalid negative IDs: {invalid_sentinels[:4]}"
            )
        valid = [value for value in row if value != -1]
        expected_valid = min(case.top_k, causal_end)
        if len(valid) != expected_valid:
            raise CorrectnessError(
                f"{name} row {row_index} expected {expected_valid} valid IDs and "
                f"{case.top_k - expected_valid} -1 slots, received {len(valid)} valid IDs"
            )
        out_of_range = [value for value in valid if value < 0 or value >= causal_end]
        if out_of_range:
            raise CorrectnessError(
                f"{name} row {row_index} contains IDs outside [0, {causal_end}): "
                f"{out_of_range[:4]}"
            )
        selected = frozenset(valid)
        if len(selected) != len(valid):
            raise CorrectnessError(f"{name} row {row_index} contains duplicate valid IDs")
        selected_sets.append(selected)
    return selected_sets


def _is_torch_tensor(value: object) -> bool:
    return type(value).__module__.startswith("torch") and hasattr(value, "device")


def _validate_torch_tensor(indices: object, case: PrefillCase, *, name: str) -> dict[str, Any]:
    """Vectorized CUDA/CPU Tensor path; avoids materializing millions of Python ints."""

    import torch

    if not case.causal:
        raise CorrectnessError("the current correctness contract requires a causal Prefill case")
    expected_shape = (case.query_tokens, 1, case.top_k)
    if tuple(indices.shape) != expected_shape:
        raise CorrectnessError(
            f"{name} shape must be {expected_shape}, received {tuple(indices.shape)}"
        )
    if indices.dtype != torch.int32:
        raise CorrectnessError(f"{name} dtype must be int32")
    if not indices.is_contiguous():
        raise CorrectnessError(f"{name} must be contiguous")

    rows = indices[:, 0, :]
    if bool((rows < -1).any().item()):
        raise CorrectnessError(f"{name} contains a negative ID other than -1")
    valid = rows >= 0
    causal_end = torch.arange(
        case.query_start + 1,
        case.context_tokens + 1,
        device=rows.device,
        dtype=torch.int64,
    )
    expected_counts = causal_end.clamp(max=case.top_k)
    counts = valid.sum(dim=1)
    mismatched = counts != expected_counts
    if bool(mismatched.any().item()):
        row = int(mismatched.nonzero()[0].item())
        raise CorrectnessError(
            f"{name} row {row} expected {int(expected_counts[row].item())} valid IDs, "
            f"received {int(counts[row].item())}"
        )
    out_of_range = valid & (rows.to(torch.int64) >= causal_end[:, None])
    if bool(out_of_range.any().item()):
        location = out_of_range.nonzero()[0]
        row = int(location[0].item())
        column = int(location[1].item())
        raise CorrectnessError(
            f"{name} row {row} ID {int(rows[row, column].item())} is outside "
            f"[0, {int(causal_end[row].item())})"
        )

    sentinel = torch.iinfo(torch.int32).max
    normalized = torch.where(valid, rows, torch.full_like(rows, sentinel))
    ordered = torch.sort(normalized, dim=1).values
    duplicates = (ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] != sentinel)
    if bool(duplicates.any().item()):
        row = int(duplicates.nonzero()[0, 0].item())
        raise CorrectnessError(f"{name} row {row} contains duplicate valid IDs")
    valid_total = int(counts.sum().item())
    return {
        "status": "passed",
        "rows": case.query_tokens,
        "top_k": case.top_k,
        "valid_ids": valid_total,
        "padding_ids": case.query_tokens * case.top_k - valid_total,
        "dtype": "int32",
        "contiguous": True,
    }


def validate_topk_indices(
    indices: object,
    case: PrefillCase,
    *,
    name: str = "indices",
) -> dict[str, Any]:
    """Validate shape, dtype, layout, causal bounds, padding, and uniqueness."""

    if _is_torch_tensor(indices):
        return _validate_torch_tensor(indices, case, name=name)
    selected = _validated_sets(indices, case, name=name)
    return {
        "status": "passed",
        "rows": case.query_tokens,
        "top_k": case.top_k,
        "valid_ids": sum(len(row) for row in selected),
        "padding_ids": case.query_tokens * case.top_k - sum(len(row) for row in selected),
        "dtype": "int32",
        "contiguous": True,
    }


def compare_topk_indices(
    candidate: object,
    baseline: object,
    case: PrefillCase,
    *,
    cutoff_tie_rows: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare exact TopK results as unordered per-row selections.

    The released API uses ``sorted=False``. Position-by-position equality would
    therefore reject semantically identical implementations.  When a compact
    cutoff oracle is supplied, a differing tied row is accepted only if it
    retains every score-above-cutoff ID and substitutes solely among IDs equal
    to the cutoff.
    """

    def validate_tied_row(
        row: int,
        actual: frozenset[int],
        expected: frozenset[int],
    ) -> None:
        if cutoff_tie_rows is None or row not in cutoff_tie_rows:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise CorrectnessError(
                f"candidate row {row} differs from baseline; "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        tie = cutoff_tie_rows[row]
        boundary = frozenset(int(value) for value in tie["boundary_ids"])
        mandatory = expected - boundary
        missing_mandatory = sorted(mandatory - actual)
        below_cutoff = sorted(actual - mandatory - boundary)
        required_boundary = int(tie["required_boundary_selections"])
        selected_boundary = len(actual & boundary)
        if missing_mandatory or below_cutoff or selected_boundary != required_boundary:
            raise CorrectnessError(
                f"candidate row {row} is not an exact score-threshold TopK; "
                f"missing_above_cutoff={missing_mandatory[:8]}, "
                f"below_cutoff={below_cutoff[:8]}, "
                f"selected_at_cutoff={selected_boundary}, "
                f"required_at_cutoff={required_boundary}"
            )

    comparison = (
        "unordered_exact_topk_score_threshold"
        if cutoff_tie_rows is not None
        else "unordered_exact_id_set"
    )

    if _is_torch_tensor(candidate) and _is_torch_tensor(baseline):
        import torch

        _validate_torch_tensor(candidate, case, name="candidate")
        _validate_torch_tensor(baseline, case, name="baseline")
        sentinel = torch.iinfo(torch.int32).max
        candidate_values = candidate[:, 0, :]
        baseline_values = baseline[:, 0, :]
        candidate_rows = torch.where(
            candidate_values >= 0,
            candidate_values,
            torch.full_like(candidate_values, sentinel),
        )
        baseline_rows = torch.where(
            baseline_values >= 0,
            baseline_values,
            torch.full_like(baseline_values, sentinel),
        )
        candidate_ordered = torch.sort(candidate_rows, dim=1).values
        baseline_ordered = torch.sort(baseline_rows, dim=1).values
        differences = candidate_ordered != baseline_ordered
        differing_rows = differences.any(dim=1).nonzero(as_tuple=False).flatten().tolist()
        for row_value in differing_rows:
            row = int(row_value)
            actual = frozenset(
                int(value)
                for value in candidate_values[row]
                .to(device="cpu", dtype=torch.int64)
                .tolist()
                if int(value) >= 0
            )
            expected = frozenset(
                int(value)
                for value in baseline_values[row]
                .to(device="cpu", dtype=torch.int64)
                .tolist()
                if int(value) >= 0
            )
            validate_tied_row(row, actual, expected)
        return {
            "status": "passed",
            "rows_compared": case.query_tokens,
            "comparison": comparison,
            "cutoff_tie_rows_available": len(cutoff_tie_rows or {}),
            "cutoff_tie_rows_substituted": len(differing_rows),
        }

    candidate_sets = _validated_sets(candidate, case, name="candidate")
    baseline_sets = _validated_sets(baseline, case, name="baseline")
    for row_index, (actual, expected) in enumerate(
        zip(candidate_sets, baseline_sets, strict=True)
    ):
        if actual != expected:
            validate_tied_row(row_index, actual, expected)
    return {
        "status": "passed",
        "rows_compared": case.query_tokens,
        "comparison": comparison,
        "cutoff_tie_rows_available": len(cutoff_tie_rows or {}),
        "cutoff_tie_rows_substituted": sum(
            actual != expected
            for actual, expected in zip(candidate_sets, baseline_sets, strict=True)
        ),
    }


# Short aliases for callers that already carry the TopK context in their names.
validate_indices = validate_topk_indices
compare_with_baseline = compare_topk_indices
