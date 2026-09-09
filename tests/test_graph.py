from __future__ import annotations

import pytest

from index_topk_perflab.api import (
    ExecutionContext,
    PrefillCase,
    PreparedGraph,
    RunMode,
    StageNode,
    StageSpec,
    VariantDescriptor,
)
from index_topk_perflab.graph import (
    execute_graph,
    graph_fingerprint,
    topological_nodes,
    validate_graph,
)


def _descriptor(plugin_id: str = "test_variant") -> VariantDescriptor:
    return VariantDescriptor(
        plugin_id=plugin_id,
        display_name="test",
        api_version="1.0",
        implementation_version="1",
        mode="test",
        description="CPU-only graph test",
        implementation="python",
        exact_topk=True,
        supported_arches=(),
    )


def _context() -> ExecutionContext:
    case = PrefillCase("tiny", 2, 4, 2, 7)
    return ExecutionContext(case=case, mode=RunMode.CORRECTNESS, iteration=0)


def test_unfused_graph_executes_in_dependency_order() -> None:
    calls: list[str] = []

    def indexer(context: ExecutionContext, artifacts: dict[str, object]) -> None:
        calls.append("indexer")
        artifacts["logits"] = [3, 1]

    def topk(context: ExecutionContext, artifacts: dict[str, object]) -> None:
        calls.append("topk")
        artifacts["ids"] = [0]

    def output(context: ExecutionContext, artifacts: dict[str, object]) -> None:
        calls.append("output")
        artifacts["indices"] = artifacts["ids"]

    # Intentionally pass nodes in non-topological order.
    graph = PreparedGraph(
        descriptor=_descriptor(),
        nodes=(
            StageNode(
                StageSpec(
                    "output",
                    ("topk",),
                    ("ids",),
                    ("indices",),
                    ("output",),
                ),
                output,
            ),
            StageNode(
                StageSpec("topk", ("indexer",), ("logits",), ("ids",), ("topk",)),
                topk,
            ),
            StageNode(
                StageSpec("indexer", (), ("inputs",), ("logits",), ("indexer",)),
                indexer,
            ),
        ),
        initial_artifacts={"inputs": object()},
    )

    assert [node.spec.stage_id for node in topological_nodes(graph)] == [
        "indexer",
        "topk",
        "output",
    ]
    artifacts = execute_graph(graph, _context())
    assert calls == ["indexer", "topk", "output"]
    assert artifacts["indices"] == [0]


def test_single_node_fused_graph_is_valid() -> None:
    def fused(context: ExecutionContext, artifacts: dict[str, object]) -> None:
        artifacts["indices"] = "terminal"

    graph = PreparedGraph(
        descriptor=_descriptor("fused_test"),
        nodes=(
            StageNode(
                StageSpec(
                    "indexer_topk_fused",
                    (),
                    ("inputs",),
                    ("indices",),
                    ("indexer", "topk", "output"),
                ),
                fused,
            ),
        ),
        initial_artifacts={"inputs": object()},
    )
    validate_graph(graph)
    assert execute_graph(graph, _context())["indices"] == "terminal"


def test_graph_rejects_duplicate_artifact_producers() -> None:
    def noop(context: ExecutionContext, artifacts: dict[str, object]) -> None:
        return None

    graph = PreparedGraph(
        descriptor=_descriptor(),
        nodes=(
            StageNode(
                StageSpec("one", (), ("inputs",), ("x",), ("indexer",)),
                noop,
            ),
            StageNode(
                StageSpec("two", ("one",), ("x",), ("x", "indices"), ("topk",)),
                noop,
            ),
        ),
        initial_artifacts={"inputs": object()},
    )
    with pytest.raises(ValueError, match="produced twice"):
        validate_graph(graph)


def test_graph_rejects_cycles() -> None:
    def noop(context: ExecutionContext, artifacts: dict[str, object]) -> None:
        return None

    graph = PreparedGraph(
        descriptor=_descriptor(),
        nodes=(
            StageNode(
                StageSpec("one", ("two",), (), ("x",), ("indexer",)),
                noop,
            ),
            StageNode(
                StageSpec("two", ("one",), (), ("indices",), ("topk",)),
                noop,
            ),
        ),
        initial_artifacts={"inputs": object()},
    )
    with pytest.raises(ValueError, match="cycle"):
        validate_graph(graph)


def test_graph_rejects_artifact_from_unrelated_stage() -> None:
    def noop(context, artifacts) -> None:
        return None

    graph = PreparedGraph(
        descriptor=_descriptor(),
        nodes=(
            StageNode(
                StageSpec("indexer", (), ("inputs",), ("logits",), ("indexer",)),
                noop,
            ),
            StageNode(
                StageSpec("topk", (), ("logits",), ("indices",), ("topk",)),
                noop,
            ),
        ),
        initial_artifacts={"inputs": object()},
    )
    with pytest.raises(ValueError, match="outside its dependency chain"):
        validate_graph(graph)


def test_graph_requires_semantics_on_terminal_dependency_chain() -> None:
    def noop(context, artifacts) -> None:
        return None

    graph = PreparedGraph(
        descriptor=_descriptor(),
        nodes=(
            StageNode(
                StageSpec("dummy_indexer", (), ("inputs",), ("x",), ("indexer",)),
                noop,
            ),
            StageNode(
                StageSpec("dummy_topk", ("dummy_indexer",), ("x",), ("y",), ("topk",)),
                noop,
            ),
            StageNode(
                StageSpec("actual", (), ("inputs",), ("indices",), ("output",)),
                noop,
            ),
        ),
        initial_artifacts={"inputs": object()},
    )
    with pytest.raises(ValueError, match="dependency closure"):
        validate_graph(graph)


def test_graph_fingerprint_is_stable_for_metadata_key_order() -> None:
    def fused(context: ExecutionContext, artifacts: dict[str, object]) -> None:
        artifacts["indices"] = 1

    node = StageNode(
        StageSpec(
            "fused",
            (),
            ("inputs",),
            ("indices",),
            ("indexer", "topk"),
        ),
        fused,
    )
    first = PreparedGraph(_descriptor(), (node,), {"inputs": 1}, metadata={"a": 1, "b": 2})
    second = PreparedGraph(_descriptor(), (node,), {"inputs": 2}, metadata={"b": 2, "a": 1})
    assert graph_fingerprint(first) == graph_fingerprint(second)
