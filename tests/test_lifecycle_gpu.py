from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from index_topk_perflab.api import (  # noqa: E402
    ExecutionContext,
    PrefillCase,
    PrefillInputs,
    PreparedGraph,
    RunMode,
    StageNode,
    StageSpec,
    VariantDescriptor,
)
from index_topk_perflab.correctness import (  # noqa: E402
    CorrectnessError,
    build_score_threshold_oracle,
    compare_topk_indices,
)
from index_topk_perflab.lifecycle import CaseLifecycle  # noqa: E402


def _descriptor(plugin_id: str) -> VariantDescriptor:
    return VariantDescriptor(
        plugin_id=plugin_id,
        display_name=plugin_id,
        api_version="1.0",
        implementation_version="1",
        mode="test",
        description="lifecycle GPU test",
        implementation="torch",
        exact_topk=True,
        supported_arches=("sm90",),
    )


def _indices(inputs: PrefillInputs) -> tuple[object, object]:
    case = inputs.case
    base = torch.arange(case.context_tokens, device="cuda", dtype=torch.float32)
    scores = base if inputs.fixture_id == "A" else -base
    logits = scores.repeat(case.query_tokens, 1)
    positions = torch.arange(case.context_tokens, device="cuda")
    logits = logits.masked_fill(positions[None, :] >= inputs.k_end[:, None], -torch.inf)
    indices = torch.topk(logits, case.top_k, dim=-1).indices.to(torch.int32)
    return logits, indices.unsqueeze(1).contiguous()


class _Baseline:
    descriptor = _descriptor("test_baseline")

    def supports(self, case: PrefillCase) -> bool:
        return True

    def prepare(self, case, inputs, *, options, mode):
        def run(context: ExecutionContext, artifacts: dict) -> None:
            logits, indices = _indices(artifacts["inputs"])
            artifacts["logits"] = logits
            artifacts["indices"] = indices

        return PreparedGraph(
            self.descriptor,
            (
                StageNode(
                    StageSpec(
                        "baseline",
                        (),
                        ("inputs",),
                        ("logits", "indices"),
                        ("indexer", "topk", "output"),
                    ),
                    run,
                ),
            ),
            {"inputs": inputs},
        )


class _Candidate(_Baseline):
    descriptor = _descriptor("test_candidate")


class _CachedCandidate(_Baseline):
    descriptor = _descriptor("test_cached_candidate")

    def prepare(self, case, inputs, *, options, mode):
        _, cached = _indices(inputs)

        def run(context: ExecutionContext, artifacts: dict) -> None:
            artifacts["indices"] = cached

        return PreparedGraph(
            self.descriptor,
            (
                StageNode(
                    StageSpec(
                        "cached",
                        (),
                        ("inputs",),
                        ("indices",),
                        ("indexer", "topk", "output"),
                    ),
                    run,
                ),
            ),
            {"inputs": inputs},
        )


def _lifecycle() -> CaseLifecycle:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("lifecycle GPU gate requires an FP8-capable SM90 GPU")
    case = PrefillCase(
        "lifecycle-gpu",
        query_tokens=2,
        context_tokens=4,
        top_k=2,
        seed=7,
        indexer_heads=2,
        head_dim=4,
    )
    workload = SimpleNamespace(
        q_dtype="float8_e4m3fn",
        kv_dtype="float8_e4m3fn",
        kv_scale_dtype="float32",
        weight_dtype="float32",
        range_dtype="int32",
    )
    lifecycle = CaseLifecycle(
        case=case,
        workload=workload,
        reference=_Baseline(),
        reference_options={},
        mode=RunMode.CORRECTNESS,
    )
    lifecycle.build_oracles()
    return lifecycle


def test_ab_fixture_gate_accepts_real_recomputation() -> None:
    lifecycle = _lifecycle()
    lifecycle.prepare_candidate(_Candidate(), {})
    assert lifecycle.check_candidate(phase="test", order=("A", "B", "A"))["status"] == (
        "passed"
    )


def test_ab_fixture_gate_rejects_prepare_time_cached_result() -> None:
    lifecycle = _lifecycle()
    lifecycle.prepare_candidate(_CachedCandidate(), {})
    with pytest.raises(CorrectnessError):
        lifecycle.check_candidate(phase="test", order=("A", "B"))


def test_cuda_score_threshold_oracle_accepts_equivalent_tied_id() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA score-threshold gate requires a GPU")
    case = PrefillCase("tie-gpu", query_tokens=1, context_tokens=5, top_k=3, seed=7)
    scores = torch.tensor([[5.0, 4.0, 3.0, 3.0, 1.0]], device="cuda")
    baseline = torch.tensor([[[0, 1, 2]]], device="cuda", dtype=torch.int32)
    candidate = torch.tensor([[[0, 1, 3]]], device="cuda", dtype=torch.int32)

    summary, tie_rows = build_score_threshold_oracle(scores, baseline, case)
    comparison = compare_topk_indices(
        candidate,
        baseline,
        case,
        cutoff_tie_rows=tie_rows,
    )

    assert summary["rows_with_cutoff_ties"] == 1
    assert summary["tie_rows"][0]["boundary_candidate_count"] == 2
    assert comparison["status"] == "passed"
    assert comparison["cutoff_tie_rows_substituted"] == 1


def test_cuda_score_threshold_oracle_rejects_below_cutoff_id() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA score-threshold gate requires a GPU")
    case = PrefillCase("tie-gpu-invalid", query_tokens=1, context_tokens=5, top_k=3, seed=7)
    scores = torch.tensor([[5.0, 4.0, 3.0, 3.0, 1.0]], device="cuda")
    baseline = torch.tensor([[[0, 1, 2]]], device="cuda", dtype=torch.int32)
    candidate = torch.tensor([[[0, 3, 4]]], device="cuda", dtype=torch.int32)
    _, tie_rows = build_score_threshold_oracle(scores, baseline, case)

    with pytest.raises(CorrectnessError, match="not an exact score-threshold TopK"):
        compare_topk_indices(
            candidate,
            baseline,
            case,
            cutoff_tie_rows=tie_rows,
        )


def test_cuda_score_threshold_oracle_rejects_invalid_reference_selection() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA score-threshold gate requires a GPU")
    case = PrefillCase("reference-invalid", query_tokens=1, context_tokens=5, top_k=3, seed=7)
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]], device="cuda")
    invalid_reference = torch.tensor([[[0, 1, 4]]], device="cuda", dtype=torch.int32)

    with pytest.raises(CorrectnessError, match="below the TopK cutoff"):
        build_score_threshold_oracle(scores, invalid_reference, case)
