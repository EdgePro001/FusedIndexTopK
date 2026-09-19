"""Validation, stable serialization, and single-stream execution of variant DAGs."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Callable, ContextManager, Iterable

from .api import ArtifactStore, ExecutionContext, PreparedGraph, StageNode

_SAFE_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_KNOWN_SEMANTICS = frozenset({"indexer", "topk", "output"})


def topological_nodes(graph: PreparedGraph) -> tuple[StageNode, ...]:
    nodes = {node.spec.stage_id: node for node in graph.nodes}
    indegree = {name: 0 for name in nodes}
    children: dict[str, list[str]] = {name: [] for name in nodes}
    for node in graph.nodes:
        for dependency in node.spec.dependencies:
            if dependency not in nodes:
                raise ValueError(
                    f"stage {node.spec.stage_id!r} depends on missing stage {dependency!r}"
                )
            indegree[node.spec.stage_id] += 1
            children[dependency].append(node.spec.stage_id)

    ready = sorted(name for name, value in indegree.items() if value == 0)
    ordered: list[StageNode] = []
    while ready:
        name = ready.pop(0)
        ordered.append(nodes[name])
        for child in sorted(children[name]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(ordered) != len(nodes):
        raise ValueError("variant stage graph contains a cycle")
    return tuple(ordered)


def validate_graph(graph: PreparedGraph) -> tuple[StageNode, ...]:
    if not _SAFE_ID.fullmatch(graph.descriptor.plugin_id):
        raise ValueError(f"unsafe plugin_id: {graph.descriptor.plugin_id!r}")
    stage_ids = [node.spec.stage_id for node in graph.nodes]
    if len(stage_ids) != len(set(stage_ids)):
        raise ValueError("stage IDs must be unique")
    for stage_id in stage_ids:
        if not _SAFE_ID.fullmatch(stage_id):
            raise ValueError(f"unsafe stage_id: {stage_id!r}")

    ordered = topological_nodes(graph)
    ancestors: dict[str, set[str]] = {}
    for node in ordered:
        direct = set(node.spec.dependencies)
        ancestors[node.spec.stage_id] = direct | {
            ancestor
            for dependency in direct
            for ancestor in ancestors[dependency]
        }

    declared_producers: dict[str, str] = {}
    for node in ordered:
        for artifact in node.spec.produces:
            if artifact in declared_producers or artifact in graph.initial_artifacts:
                owner = declared_producers.get(artifact, "initial_artifacts")
                raise ValueError(f"artifact {artifact!r} is produced twice; first owner={owner}")
            declared_producers[artifact] = node.spec.stage_id

    available = set(graph.initial_artifacts)
    completed: set[str] = set()
    for node in ordered:
        spec = node.spec
        if not set(spec.dependencies).issubset(completed):
            raise ValueError(f"stage {spec.stage_id!r} is not topologically executable")
        missing = set(spec.consumes) - available
        if missing:
            raise ValueError(
                f"stage {spec.stage_id!r} consumes missing artifacts: {sorted(missing)}"
            )
        unrelated = {
            artifact: declared_producers[artifact]
            for artifact in spec.consumes
            if artifact in declared_producers
            and declared_producers[artifact] not in ancestors[spec.stage_id]
        }
        if unrelated:
            raise ValueError(
                f"stage {spec.stage_id!r} consumes artifacts outside its dependency chain: "
                f"{unrelated}"
            )
        for artifact in spec.produces:
            available.add(artifact)
        unknown = set(spec.semantic_ops) - _KNOWN_SEMANTICS
        if unknown:
            raise ValueError(f"stage {spec.stage_id!r} has unknown semantic ops: {sorted(unknown)}")
        completed.add(spec.stage_id)
    if graph.terminal_artifact not in declared_producers:
        raise ValueError(f"terminal artifact {graph.terminal_artifact!r} is never produced")
    terminal_owner = declared_producers[graph.terminal_artifact]
    terminal_closure = {terminal_owner, *ancestors[terminal_owner]}
    terminal_semantics = {
        op
        for node in ordered
        if node.spec.stage_id in terminal_closure
        for op in node.spec.semantic_ops
    }
    if not {"indexer", "topk"}.issubset(terminal_semantics):
        raise ValueError(
            "terminal artifact dependency closure must cover both indexer and topk semantics"
        )
    dead_comparable = {
        node.spec.stage_id
        for node in ordered
        if node.spec.stage_id not in terminal_closure
        and {"indexer", "topk"}.intersection(node.spec.semantic_ops)
    }
    if dead_comparable:
        raise ValueError(
            "indexer/topk semantic stages must contribute to the terminal artifact: "
            f"{sorted(dead_comparable)}"
        )
    return ordered


def graph_mapping(graph: PreparedGraph) -> dict[str, Any]:
    ordered = validate_graph(graph)
    return {
        "descriptor": asdict(graph.descriptor),
        "nodes": [asdict(node.spec) for node in ordered],
        "initial_artifact_keys": sorted(graph.initial_artifacts),
        "terminal_artifact": graph.terminal_artifact,
        "metadata": dict(graph.metadata),
    }


def graph_fingerprint(graph: PreparedGraph) -> str:
    payload = json.dumps(
        graph_mapping(graph),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


StageContextFactory = Callable[[StageNode], ContextManager[Any]]


def execute_graph(
    graph: PreparedGraph,
    context: ExecutionContext,
    *,
    stage_context: StageContextFactory | None = None,
) -> ArtifactStore:
    """Validate, then execute one graph iteration on the caller's CUDA stream."""

    artifacts: ArtifactStore = dict(graph.initial_artifacts)
    return execute_prevalidated(
        validate_graph(graph),
        context,
        artifacts,
        stage_context=stage_context,
        validate_produced=True,
    )


def execute_prevalidated(
    nodes: Iterable[StageNode],
    context: ExecutionContext,
    artifacts: ArtifactStore,
    *,
    stage_context: StageContextFactory | None = None,
    validate_produced: bool = False,
) -> ArtifactStore:
    """Execute an already validated graph without changing its launch path.

    Formal timing and profiler capture both use this executor after validation,
    so their physical stage call sequence stays identical.  Expensive graph and
    artifact-contract checks remain outside the measured/captured interval.
    """

    ordered = tuple(nodes)
    if stage_context is None and not validate_produced:
        for node in ordered:
            node.run(context, artifacts)
        return artifacts

    for node in ordered:
        manager = stage_context(node) if stage_context is not None else nullcontext()
        with manager:
            node.run(context, artifacts)
        if validate_produced:
            missing = set(node.spec.produces) - set(artifacts)
            if missing:
                raise RuntimeError(
                    f"stage {node.spec.stage_id!r} did not produce declared artifacts: "
                    f"{sorted(missing)}"
                )
    return artifacts


def semantic_coverage(nodes: Iterable[StageNode]) -> tuple[str, ...]:
    return tuple(sorted({op for node in nodes for op in node.spec.semantic_ops}))
