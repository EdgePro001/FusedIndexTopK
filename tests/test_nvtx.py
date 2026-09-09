from __future__ import annotations

import pytest

from index_topk_perflab.api import (
    PrefillCase,
    PreparedGraph,
    StageNode,
    StageSpec,
    VariantDescriptor,
)
from index_topk_perflab.nvtx import (
    graph_labels,
    ncu_push_pop_filter,
    nvtx_range,
    pipeline_label,
)


def _graph() -> PreparedGraph:
    descriptor = VariantDescriptor(
        plugin_id="fused_v1",
        display_name="fused",
        api_version="1.0",
        implementation_version="1",
        mode="fused",
        description="test",
        implementation="python",
        exact_topk=True,
        supported_arches=(),
    )
    node = StageNode(
        StageSpec(
            "indexer_topk_fused",
            (),
            ("inputs",),
            ("indices",),
            ("indexer", "topk", "output"),
        ),
        lambda context, artifacts: artifacts.update(indices=[]),
    )
    return PreparedGraph(descriptor, (node,), {"inputs": object()})


def test_graph_labels_are_stable_and_ncu_filter_has_push_pop_suffix() -> None:
    case = PrefillCase("prefill-8k", 4096, 8192, 2048, 7)
    labels = graph_labels(_graph(), case)
    assert labels.pipeline == "ITK::RUN::fused_v1::prefill-8k"
    stage = labels.stages["indexer_topk_fused"]
    assert stage == "ITK::STAGE::fused_v1::prefill-8k::indexer_topk_fused"
    assert ncu_push_pop_filter(stage) == f"{stage}/"


def test_nvtx_components_reject_filter_metacharacters() -> None:
    with pytest.raises(ValueError):
        pipeline_label("bad/variant", "prefill-8k")


def test_nvtx_range_always_pops() -> None:
    calls: list[tuple[str, str | None]] = []

    class Nvtx:
        @staticmethod
        def range_push(label: str) -> None:
            calls.append(("push", label))

        @staticmethod
        def range_pop() -> None:
            calls.append(("pop", None))

    class Cuda:
        nvtx = Nvtx()

    class FakeTorch:
        cuda = Cuda()

    with pytest.raises(RuntimeError), nvtx_range(FakeTorch, "ITK::RUN::x::y"):
        raise RuntimeError("boom")
    assert calls == [("push", "ITK::RUN::x::y"), ("pop", None)]
