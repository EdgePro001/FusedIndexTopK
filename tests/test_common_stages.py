from __future__ import annotations

import sys

from index_topk_perflab.variants.common import (
    deepgemm_indexer_stage,
    int32_output_stage,
    torch_exact_topk_stage,
)


def test_reusable_unfused_stage_specs_are_cuda_lazy() -> None:
    torch_before = sys.modules.get("torch")
    indexer = deepgemm_indexer_stage(object())
    topk = torch_exact_topk_stage(object())
    output = int32_output_stage(object())
    assert [node.spec.stage_id for node in (indexer, topk, output)] == [
        "indexer",
        "topk",
        "output",
    ]
    assert topk.spec.dependencies == ("indexer",)
    assert output.spec.produces == ("indices",)
    assert sys.modules.get("torch") is torch_before
