"""Reusable physical stages for released DeepGEMM-based operators.

The core plugin contract never requires these intermediates.  They merely make
it possible to replace one physical stage while keeping the terminal contract
and timing boundary unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..api import ExecutionContext, PrefillCase, PrefillInputs, StageNode, StageSpec


def deepgemm_indexer_stage(deep_gemm: Any) -> StageNode:
    def run_indexer(context: ExecutionContext, artifacts: dict[str, Any]) -> None:
        standard: PrefillInputs = artifacts["inputs"]
        artifacts["logits"] = deep_gemm.fp8_mqa_logits(
            standard.q,
            (standard.kv, standard.kv_scales),
            standard.weights,
            standard.k_start,
            standard.k_end,
            clean_logits=True,
        )

    return StageNode(
        StageSpec(
            stage_id="indexer",
            dependencies=(),
            consumes=("inputs",),
            produces=("logits",),
            semantic_ops=("indexer",),
            description="DeepGEMM FP8 MQA scorer plus causal cleanup",
            kernel_regexes=("sm90_fp8_mqa_logits", "clean_logits"),
        ),
        run_indexer,
    )


def torch_exact_topk_stage(torch: Any, *, sorted_output: bool = False) -> StageNode:
    def run_topk(context: ExecutionContext, artifacts: dict[str, Any]) -> None:
        logits = artifacts["logits"]
        values, ids = torch.topk(
            logits,
            min(context.case.top_k, logits.shape[-1]),
            dim=-1,
            sorted=sorted_output,
        )
        artifacts["topk_values"] = values
        artifacts["topk_ids"] = ids

    return StageNode(
        StageSpec(
            stage_id="topk",
            dependencies=("indexer",),
            consumes=("logits",),
            produces=("topk_values", "topk_ids"),
            semantic_ops=("topk",),
            description="PyTorch CUDA exact TopK",
            kernel_regexes=("mbtopk", "radixFindKthValues", "gatherTopK"),
        ),
        run_topk,
    )


def supports_frozen_deepgemm_case(case: PrefillCase) -> bool:
    """Return whether the released DeepGEMM scorer supports this case."""

    return (
        case.batch_size == 1
        and case.causal
        and case.indexer_heads == 64
        and case.head_dim == 128
        and case.query_tokens > 0
        and case.context_tokens >= case.query_tokens
        and case.context_tokens >= case.top_k > 0
    )


def physical_rows_with_clean_tail(torch: Any, logits: Any) -> Any:
    """Expose DeepGEMM physical rows and initialize their small padded tail."""

    if logits.ndim != 2 or logits.stride(1) != 1 or logits.storage_offset() != 0:
        raise RuntimeError(
            "external TopK requires a zero-offset 2D logits view with unit inner stride"
        )
    rows, logical_columns = logits.shape
    row_stride = int(logits.stride(0))
    if row_stride < logical_columns:
        raise RuntimeError("logits leading stride is smaller than its logical width")
    physical = torch.as_strided(logits, (rows, row_stride), (row_stride, 1))
    if row_stride != logical_columns:
        physical[:, logical_columns:].fill_(-torch.inf)
    return physical


TopKLauncher = Callable[[Any, Any, Any], None]


def external_topk_stage(
    torch: Any,
    *,
    description: str,
    launcher: TopKLauncher,
    output_values: Any,
    output_ids: Any,
    kernel_regexes: tuple[str, ...],
) -> StageNode:
    """Build a timed exact-TopK stage around one precompiled launcher."""

    def run_topk(context: ExecutionContext, artifacts: dict[str, Any]) -> None:
        physical = physical_rows_with_clean_tail(torch, artifacts["logits"])
        launcher(physical, output_ids, output_values)
        artifacts["topk_values"] = output_values
        artifacts["topk_ids"] = output_ids

    return StageNode(
        StageSpec(
            stage_id="topk",
            dependencies=("indexer",),
            consumes=("logits",),
            produces=("topk_values", "topk_ids"),
            semantic_ops=("topk",),
            description=description,
            kernel_regexes=("fill",) + kernel_regexes,
        ),
        run_topk,
    )


def int32_output_stage(torch: Any) -> StageNode:
    def run_output(context: ExecutionContext, artifacts: dict[str, Any]) -> None:
        standard: PrefillInputs = artifacts["inputs"]
        values = artifacts["topk_values"]
        ids = artifacts["topk_ids"]
        valid = torch.isfinite(values) & (ids >= 0) & (ids < standard.k_end[:, None])
        padded = torch.where(valid, ids, torch.full_like(ids, -1))
        artifacts["indices"] = padded.to(torch.int32).unsqueeze(1).contiguous()

    return StageNode(
        StageSpec(
            stage_id="output",
            dependencies=("topk",),
            consumes=("inputs", "topk_values", "topk_ids"),
            produces=("indices",),
            semantic_ops=("output",),
            description="Causal validity, -1 padding, INT32 cast, and contiguous pack",
        ),
        run_output,
    )
