#!/usr/bin/env python3
"""Audit and summarize the frozen-stack R13a release qualification campaign."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from index_topk_perflab.artifacts import canonical_hash, load_json, write_json_atomic

LAYERS = (0, 15, 30, 45, 60)
SPLITS = ("test_normal", "test_hard")
CONTEXTS = (6144, 8192, 12288, 16384)
VARIANTS = ("flashinfer", "fused_r13a_nsweep", "torch")
BLOCKS = (
    ("block0_frt", ("flashinfer", "fused_r13a_nsweep", "torch")),
    ("block1_rtf", ("fused_r13a_nsweep", "torch", "flashinfer")),
    ("block2_tfr", ("torch", "flashinfer", "fused_r13a_nsweep")),
)
PROTOCOL = {
    "method": "deepgemm_kineto_cupti_v1",
    "warmup_iterations": 10,
    "event_trials": 20,
    "kineto_trials": 30,
    "l2_flush_bytes": 8_000_000_000,
}
BASELINE = "flashinfer"
CANDIDATE = "fused_r13a_nsweep"


def _quantile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics(values: Iterable[float]) -> dict[str, float | int]:
    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("cannot summarize an empty sample")
    median = float(statistics.median(sample))
    mad = float(statistics.median(abs(value - median) for value in sample))
    return {
        "count": len(sample),
        "min": min(sample),
        "median": median,
        "mean": float(statistics.fmean(sample)),
        "p95": _quantile(sample, 0.95),
        "max": max(sample),
        "mad": mad,
        "normalized_mad": mad / median if median else 0.0,
    }


def _bootstrap_median_ci(
    values: Iterable[float],
    *,
    seed: int,
    repetitions: int = 10_000,
) -> dict[str, float | int]:
    sample = [float(value) for value in values]
    if len(sample) < 3:
        raise ValueError("bootstrap requires at least three observations")
    generator = random.Random(seed)
    estimates = sorted(
        statistics.median(generator.choices(sample, k=len(sample)))
        for _ in range(repetitions)
    )
    return {
        "method": "deterministic_nonparametric_bootstrap_of_median",
        "repetitions": repetitions,
        "seed": seed,
        "lower_95": _quantile(estimates, 0.025),
        "upper_95": _quantile(estimates, 0.975),
    }


def _measurement_values(
    result: dict[str, Any],
    *,
    pass_name: str,
    fixture: str | None = None,
) -> list[float]:
    return [
        float(row["latency_ms"])
        for row in result["measurements"]
        if row["pass"] == pass_name
        and row["stage_id"] == "operator_total"
        and (fixture is None or row["fixture_id"] == fixture)
    ]


def _require_passed(value: Any, label: str) -> None:
    if not isinstance(value, dict) or value.get("status") != "passed":
        raise ValueError(f"{label} did not pass")


def _runtime_class(identity: dict[str, Any]) -> str:
    payload = deepcopy(identity["payload"])
    payload["device"].pop("physical_uuid", None)
    return canonical_hash(payload)


def _validate_result(
    result: dict[str, Any],
    row: dict[str, Any],
    block: dict[str, Any],
    *,
    label: str,
) -> None:
    if result.get("runtime_identity", {}).get("sha256") != block["runtime_identity"][
        "sha256"
    ]:
        raise ValueError(f"{label} runtime identity differs from its block")
    if result.get("protocol") != PROTOCOL | {"cooldown_seconds": 0.0}:
        raise ValueError(f"{label} result protocol is not the frozen release protocol")
    for gate_name in (
        "preflight_gate",
        "postflight_gate",
        "postflight_contract_check",
        "postflight_exact_reference_check",
    ):
        _require_passed(result.get(gate_name), f"{label} {gate_name}")
    _require_passed(result.get("kineto", {}).get("topology_gate"), f"{label} topology")
    for phase in ("before", "after"):
        _require_passed(
            result.get("gpu_exclusivity", {}).get(phase),
            f"{label} GPU exclusivity {phase}",
        )
    integrity = result.get("lifecycle", {}).get("integrity_checks", [])
    if len(integrity) < 4 or any(item.get("status") != "passed" for item in integrity):
        raise ValueError(f"{label} input integrity history is incomplete")
    for item in integrity:
        for fixture in item.get("fixtures", {}).values():
            if (
                fixture.get("sha256_before") != fixture.get("sha256_after")
                or fixture.get("versions_unchanged") is not True
            ):
                raise ValueError(f"{label} mutated an input fixture")

    source = result.get("source_identity")
    if not isinstance(source, dict):
        raise ValueError(f"{label} is missing source identity")
    candidate_source = source.get("candidate", {})
    if candidate_source.get("sha256") != row.get("source_identity_sha256"):
        raise ValueError(f"{label} source identity does not match its summary row")
    if canonical_hash(candidate_source.get("payload")) != candidate_source.get("sha256"):
        raise ValueError(f"{label} candidate source identity is internally inconsistent")
    reference_source = source.get("reference", {})
    if canonical_hash(reference_source.get("payload")) != reference_source.get("sha256"):
        raise ValueError(f"{label} reference source identity is internally inconsistent")
    if source.get("framework_sha256") != row.get("framework_sha256"):
        raise ValueError(f"{label} framework identity does not match its summary row")

    expected_counts = {
        "formal_kernel_sum": (PROTOCOL["kineto_trials"], PROTOCOL["kineto_trials"] // 2),
        "cuda_event_total": (PROTOCOL["event_trials"], PROTOCOL["event_trials"] // 2),
    }
    for pass_name, (total, per_fixture) in expected_counts.items():
        if len(_measurement_values(result, pass_name=pass_name)) != total:
            raise ValueError(f"{label} has the wrong {pass_name} trial count")
        for fixture in ("A", "B"):
            if (
                len(
                    _measurement_values(
                        result,
                        pass_name=pass_name,
                        fixture=fixture,
                    )
                )
                != per_fixture
            ):
                raise ValueError(
                    f"{label} has the wrong {pass_name}/{fixture} trial count"
                )


def _variant_latency(
    runs: dict[tuple[int, str, int, str, str], dict[str, Any]],
    *,
    layer: int,
    split: str,
    context: int,
    variant: str,
    pass_name: str,
) -> dict[str, Any]:
    raw: list[float] = []
    block_medians: list[float] = []
    fixture_medians: dict[str, list[float]] = {"A": [], "B": []}
    for block_name, _ in BLOCKS:
        result = runs[(layer, split, context, block_name, variant)]
        block_values = _measurement_values(result, pass_name=pass_name)
        raw.extend(block_values)
        block_medians.append(float(statistics.median(block_values)))
        for fixture in ("A", "B"):
            fixture_medians[fixture].append(
                float(
                    statistics.median(
                        _measurement_values(
                            result,
                            pass_name=pass_name,
                            fixture=fixture,
                        )
                    )
                )
            )
    block_median = float(statistics.median(block_medians))
    return {
        "raw_trials": _statistics(raw),
        "block_medians_ms": block_medians,
        "median_of_block_medians_ms": block_median,
        "block_range_ms": [min(block_medians), max(block_medians)],
        "block_range_percent_of_median": (
            100.0 * (max(block_medians) - min(block_medians)) / block_median
        ),
        "fixture_medians_ms": fixture_medians,
    }


def _paired_reductions(
    runs: dict[tuple[int, str, int, str, str], dict[str, Any]],
    *,
    layer: int,
    split: str,
    context: int,
    pass_name: str,
    baseline_variant: str = BASELINE,
) -> list[float]:
    reductions: list[float] = []
    for block_name, _ in BLOCKS:
        candidate = runs[(layer, split, context, block_name, CANDIDATE)]
        baseline = runs[(layer, split, context, block_name, baseline_variant)]
        for fixture in ("A", "B"):
            candidate_ms = statistics.median(
                _measurement_values(
                    candidate,
                    pass_name=pass_name,
                    fixture=fixture,
                )
            )
            baseline_ms = statistics.median(
                _measurement_values(
                    baseline,
                    pass_name=pass_name,
                    fixture=fixture,
                )
            )
            reductions.append(100.0 * (baseline_ms - candidate_ms) / baseline_ms)
    return reductions


def _write_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "layer",
        "split",
        "context_tokens",
        "variant",
        "cupti_median_ms",
        "cupti_p95_ms",
        "event_median_ms",
        "event_p95_ms",
        "cupti_reduction_vs_flashinfer_percent",
        "event_reduction_vs_flashinfer_percent",
        "cupti_reduction_vs_torch_percent",
        "event_reduction_vs_torch_percent",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            for variant in VARIANTS:
                latency = cell["variants"][variant]
                writer.writerow(
                    {
                        "layer": cell["layer"],
                        "split": cell["split"],
                        "context_tokens": cell["context_tokens"],
                        "variant": variant,
                        "cupti_median_ms": latency["cupti"][
                            "median_of_block_medians_ms"
                        ],
                        "cupti_p95_ms": latency["cupti"]["raw_trials"]["p95"],
                        "event_median_ms": latency["cuda_event"][
                            "median_of_block_medians_ms"
                        ],
                        "event_p95_ms": latency["cuda_event"]["raw_trials"]["p95"],
                        "cupti_reduction_vs_flashinfer_percent": (
                            cell["comparison_vs_flashinfer"]["cupti"][
                                "reduction_percent"
                            ]
                            if variant == CANDIDATE
                            else ""
                        ),
                        "event_reduction_vs_flashinfer_percent": (
                            cell["comparison_vs_flashinfer"]["cuda_event"][
                                "reduction_percent"
                            ]
                            if variant == CANDIDATE
                            else ""
                        ),
                        "cupti_reduction_vs_torch_percent": (
                            cell["comparison_vs_torch"]["cupti"][
                                "reduction_percent"
                            ]
                            if variant == CANDIDATE
                            else ""
                        ),
                        "event_reduction_vs_torch_percent": (
                            cell["comparison_vs_torch"]["cuda_event"][
                                "reduction_percent"
                            ]
                            if variant == CANDIDATE
                            else ""
                        ),
                    }
                )


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    aggregate = payload["aggregate_vs_flashinfer"]
    torch_aggregate = payload["aggregate_vs_torch"]
    gates = payload["acceptance_gates"]
    lines = [
        "# R13a frozen-stack release qualification",
        "",
        f"Status: **{gates['frozen_stack_status']}**",
        "",
        "Primary metric: CUPTI sum of all CUDA kernels in the complete operator. "
        "CUDA Event is an independent whole-pipeline guardrail.",
        "",
        "## Aggregate result",
        "",
        "| metric | CUPTI | CUDA Event |",
        "|---|---:|---:|",
        (
            "| winning cells | "
            f"{aggregate['cupti']['winning_cells']}/{aggregate['cell_count']} | "
            f"{aggregate['cuda_event']['winning_cells']}/{aggregate['cell_count']} |"
        ),
        (
            "| median latency reduction | "
            f"{aggregate['cupti']['cell_reductions_percent']['median']:.3f}% | "
            f"{aggregate['cuda_event']['cell_reductions_percent']['median']:.3f}% |"
        ),
        (
            "| median reduction vs PyTorch | "
            f"{torch_aggregate['cupti']['cell_reductions_percent']['median']:.3f}% | "
            f"{torch_aggregate['cuda_event']['cell_reductions_percent']['median']:.3f}% |"
        ),
        (
            "| bootstrap 95% CI of median reduction | "
            f"[{aggregate['cupti']['bootstrap_95']['lower_95']:.3f}%, "
            f"{aggregate['cupti']['bootstrap_95']['upper_95']:.3f}%] | "
            f"[{aggregate['cuda_event']['bootstrap_95']['lower_95']:.3f}%, "
            f"{aggregate['cuda_event']['bootstrap_95']['upper_95']:.3f}%] |"
        ),
        "",
        "## Per-cell CUPTI result",
        "",
        "| layer | split | N | FlashInfer ms | R13a ms | reduction | 95% CI |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for cell in payload["cells"]:
        comparison = cell["comparison_vs_flashinfer"]["cupti"]
        lines.append(
            f"| {cell['layer']} | {cell['split']} | {cell['context_tokens']} | "
            f"{comparison['baseline_ms']:.6f} | {comparison['candidate_ms']:.6f} | "
            f"{comparison['reduction_percent']:.3f}% | "
            f"[{comparison['paired_bootstrap_95']['lower_95']:.3f}%, "
            f"{comparison['paired_bootstrap_95']['upper_95']:.3f}%] |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            f"- Audit/correctness/provenance: {gates['audit_correctness_provenance']}",
            f"- CUPTI contribution gate: {gates['cupti_contribution_gate']}",
            f"- CUDA Event guardrail: {gates['cuda_event_guardrail']}",
            f"- Block stability gate: {gates['block_stability_gate']}",
            f"- Current-upstream merge-ready: {gates['current_upstream_merge_ready']}",
            "",
            "The last gate remains false until the kernel is ported to and retested "
            "against current DeepGEMM main; this report qualifies the pinned frozen stack.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    runs: dict[tuple[int, str, int, str, str], dict[str, Any]] = {}
    input_hashes: dict[tuple[int, str, int], set[str]] = defaultdict(set)
    graph_hashes: dict[tuple[int, str, int, str], set[str]] = defaultdict(set)
    source_hashes: dict[str, set[str]] = defaultdict(set)
    reference_source_hashes: set[str] = set()
    framework_hashes: set[str] = set()
    layer_runtime_hashes: dict[int, set[str]] = defaultdict(set)
    runtime_classes: set[str] = set()
    layer_gpu_uuids: dict[int, set[str]] = defaultdict(set)
    manifest_hashes: dict[int, set[str]] = defaultdict(set)
    source_summaries: list[str] = []

    expected_keys = {
        (split, context, variant)
        for split in SPLITS
        for context in CONTEXTS
        for variant in VARIANTS
    }
    for layer in LAYERS:
        layer_name = f"layer{layer:03d}"
        for block_name, expected_order in BLOCKS:
            summary_path = root / layer_name / block_name / "summary.json"
            block = load_json(summary_path)
            source_summaries.append(str(summary_path))
            if block.get("protocol") != PROTOCOL:
                raise ValueError(f"{summary_path} does not use the release protocol")
            _require_passed(
                block.get("initial_gpu_exclusivity"),
                f"{summary_path} initial GPU exclusivity",
            )
            manifest_hashes[layer].add(str(block["manifest_sha256"]))
            runtime_hash = str(block["runtime_identity"]["sha256"])
            layer_runtime_hashes[layer].add(runtime_hash)
            runtime_classes.add(_runtime_class(block["runtime_identity"]))
            gpu_uuid = str(
                block["runtime_identity"]["payload"]["device"]["physical_uuid"]
            )
            layer_gpu_uuids[layer].add(gpu_uuid)

            rows = block.get("rows", [])
            actual_keys = {
                (row["split"], int(row["context_tokens"]), row["variant"])
                for row in rows
            }
            if actual_keys != expected_keys or len(rows) != len(expected_keys):
                raise ValueError(f"{summary_path} has an incomplete or duplicate matrix")
            if {row["split"] for row in rows} != set(SPLITS):
                raise ValueError(f"{summary_path} contains a non-release split")
            for split in SPLITS:
                for context in CONTEXTS:
                    actual_order = tuple(
                        row["variant"]
                        for row in rows
                        if row["split"] == split
                        and int(row["context_tokens"]) == context
                    )
                    if actual_order != expected_order:
                        raise ValueError(
                            f"{summary_path} has order {actual_order}, expected "
                            f"{expected_order} at {split}/N={context}"
                        )
            for row in rows:
                if any(
                    row.get(field) != "passed"
                    for field in ("exactness", "kineto_topology", "gpu_exclusivity")
                ):
                    raise ValueError(f"failed summary gate in {summary_path}: {row}")
                split = str(row["split"])
                context = int(row["context_tokens"])
                variant = str(row["variant"])
                result_path = summary_path.parent / row["result"]
                result = load_json(result_path)
                label = f"layer={layer}/{block_name}/{split}/N={context}/{variant}"
                _validate_result(result, row, block, label=label)
                key = (layer, split, context, block_name, variant)
                if key in runs:
                    raise ValueError(f"duplicate run: {key}")
                runs[key] = result
                input_hashes[(layer, split, context)].add(
                    str(result["input_content_hash"])
                )
                graph_hashes[(layer, split, context, variant)].add(
                    str(result["graph_fingerprint"])
                )
                source = result["source_identity"]
                source_hashes[variant].add(str(source["candidate"]["sha256"]))
                reference_source_hashes.add(str(source["reference"]["sha256"]))
                framework_hashes.add(str(source["framework_sha256"]))

    if any(len(values) != 1 for values in manifest_hashes.values()):
        raise ValueError("a layer changed replay manifest between blocks")
    if any(len(values) != 1 for values in layer_runtime_hashes.values()):
        raise ValueError("a layer changed GPU/runtime between blocks")
    if any(len(values) != 1 for values in layer_gpu_uuids.values()):
        raise ValueError("a layer moved between physical GPUs")
    if len(runtime_classes) != 1:
        raise ValueError("workers do not share one hardware/software runtime class")
    if any(len(values) != 1 for values in input_hashes.values()):
        raise ValueError("a cell did not use byte-identical inputs for every variant/block")
    if any(len(values) != 1 for values in graph_hashes.values()):
        raise ValueError("a variant graph changed between blocks")
    if any(len(values) != 1 for values in source_hashes.values()):
        raise ValueError("a variant source identity changed during the campaign")
    if len(reference_source_hashes) != 1 or len(framework_hashes) != 1:
        raise ValueError("reference or framework source changed during the campaign")

    cells: list[dict[str, Any]] = []
    for layer in LAYERS:
        for split in SPLITS:
            for context in CONTEXTS:
                variants: dict[str, Any] = {}
                for variant in VARIANTS:
                    variants[variant] = {
                        "cupti": _variant_latency(
                            runs,
                            layer=layer,
                            split=split,
                            context=context,
                            variant=variant,
                            pass_name="formal_kernel_sum",
                        ),
                        "cuda_event": _variant_latency(
                            runs,
                            layer=layer,
                            split=split,
                            context=context,
                            variant=variant,
                            pass_name="cuda_event_total",
                        ),
                    }
                comparison: dict[str, Any] = {}
                torch_comparison: dict[str, Any] = {}
                for metric, pass_name in (
                    ("cupti", "formal_kernel_sum"),
                    ("cuda_event", "cuda_event_total"),
                ):
                    baseline_ms = variants[BASELINE][metric][
                        "median_of_block_medians_ms"
                    ]
                    candidate_ms = variants[CANDIDATE][metric][
                        "median_of_block_medians_ms"
                    ]
                    paired = _paired_reductions(
                        runs,
                        layer=layer,
                        split=split,
                        context=context,
                        pass_name=pass_name,
                    )
                    seed = 20260905 + layer * 100_000 + context + (
                        0 if metric == "cupti" else 1
                    )
                    comparison[metric] = {
                        "baseline_ms": baseline_ms,
                        "candidate_ms": candidate_ms,
                        "speedup": baseline_ms / candidate_ms,
                        "reduction_percent": 100.0
                        * (baseline_ms - candidate_ms)
                        / baseline_ms,
                        "paired_block_fixture_reductions_percent": paired,
                        "paired_reduction_statistics": _statistics(paired),
                        "paired_bootstrap_95": _bootstrap_median_ci(
                            paired,
                            seed=seed,
                        ),
                    }
                    torch_ms = variants["torch"][metric][
                        "median_of_block_medians_ms"
                    ]
                    torch_paired = _paired_reductions(
                        runs,
                        layer=layer,
                        split=split,
                        context=context,
                        pass_name=pass_name,
                        baseline_variant="torch",
                    )
                    torch_comparison[metric] = {
                        "baseline_ms": torch_ms,
                        "candidate_ms": candidate_ms,
                        "speedup": torch_ms / candidate_ms,
                        "reduction_percent": 100.0
                        * (torch_ms - candidate_ms)
                        / torch_ms,
                        "paired_block_fixture_reductions_percent": torch_paired,
                        "paired_reduction_statistics": _statistics(torch_paired),
                        "paired_bootstrap_95": _bootstrap_median_ci(
                            torch_paired,
                            seed=seed + 10_000_000,
                        ),
                    }
                cells.append(
                    {
                        "layer": layer,
                        "split": split,
                        "context_tokens": context,
                        "variants": variants,
                        "comparison_vs_flashinfer": comparison,
                        "comparison_vs_torch": torch_comparison,
                    }
                )

    aggregate: dict[str, Any] = {"cell_count": len(cells)}
    for metric_index, metric in enumerate(("cupti", "cuda_event")):
        reductions = [
            cell["comparison_vs_flashinfer"][metric]["reduction_percent"]
            for cell in cells
        ]
        aggregate[metric] = {
            "winning_cells": sum(value > 0.0 for value in reductions),
            "losing_cells": sum(value < 0.0 for value in reductions),
            "cell_reductions_percent": _statistics(reductions),
            "bootstrap_95": _bootstrap_median_ci(
                reductions,
                seed=2026090500 + metric_index,
            ),
        }

    aggregate_torch: dict[str, Any] = {"cell_count": len(cells)}
    for metric_index, metric in enumerate(("cupti", "cuda_event")):
        reductions = [
            cell["comparison_vs_torch"][metric]["reduction_percent"]
            for cell in cells
        ]
        aggregate_torch[metric] = {
            "winning_cells": sum(value > 0.0 for value in reductions),
            "losing_cells": sum(value < 0.0 for value in reductions),
            "cell_reductions_percent": _statistics(reductions),
            "bootstrap_95": _bootstrap_median_ci(
                reductions,
                seed=2026090510 + metric_index,
            ),
        }

    max_block_range = max(
        cell["variants"][variant][metric]["block_range_percent_of_median"]
        for cell in cells
        for variant in (BASELINE, CANDIDATE)
        for metric in ("cupti", "cuda_event")
    )
    cup = aggregate["cupti"]
    event = aggregate["cuda_event"]
    audit_gate = True
    cup_gate = (
        cup["winning_cells"] >= 38
        and cup["cell_reductions_percent"]["min"] >= -2.0
        and cup["bootstrap_95"]["lower_95"] > 0.0
    )
    event_gate = (
        event["winning_cells"] >= 36
        and event["cell_reductions_percent"]["median"] > 0.0
    )
    stability_gate = max_block_range <= 5.0
    frozen_stack_pass = audit_gate and cup_gate and event_gate and stability_gate
    gold = (
        frozen_stack_pass
        and cup["winning_cells"] == len(cells)
        and event["winning_cells"] == len(cells)
    )
    gates = {
        "audit_correctness_provenance": "passed" if audit_gate else "failed",
        "cupti_contribution_gate": "passed" if cup_gate else "failed",
        "cuda_event_guardrail": "passed" if event_gate else "failed",
        "block_stability_gate": "passed" if stability_gate else "failed",
        "maximum_observed_block_range_percent": max_block_range,
        "minimum_release_rule": (
            "CUPTI wins >=38/40, no cell regresses >2%, aggregate median "
            "reduction bootstrap lower bound >0; Event wins >=36/40 with "
            "positive median; all baseline/candidate block ranges <=5%"
        ),
        "gold_rule": "minimum rule plus 40/40 wins in both CUPTI and CUDA Event",
        "frozen_stack_status": (
            "gold" if gold else "passed" if frozen_stack_pass else "failed"
        ),
        "current_upstream_merge_ready": False,
        "current_upstream_reason": (
            "R13a is qualified against pinned DeepGEMM 7c95b14; porting and "
            "rerunning current-main attention tests remains a separate gate"
        ),
    }
    payload = {
        "schema_version": 1,
        "title": "R13a frozen-stack release qualification",
        "method": (
            "three direct balanced-order blocks; median of block medians; "
            "cell-level deterministic bootstrap"
        ),
        "matrix": {
            "layers": list(LAYERS),
            "splits": list(SPLITS),
            "contexts": list(CONTEXTS),
            "variants": list(VARIANTS),
            "blocks": [
                {"name": name, "variant_order": list(order)}
                for name, order in BLOCKS
            ],
            "cells": len(cells),
            "timed_result_files": len(runs),
            "tuning_split_used": False,
        },
        "protocol": PROTOCOL,
        "audit": {
            "all_result_gates_passed": True,
            "same_input_bytes_within_each_cell": True,
            "stable_graph_within_each_variant_cell": True,
            "stable_variant_sources": True,
            "stable_framework_source": True,
            "one_runtime_per_layer": True,
            "same_runtime_class_across_layers": True,
            "physical_gpu_by_layer": {
                str(layer): next(iter(layer_gpu_uuids[layer])) for layer in LAYERS
            },
            "manifest_sha256_by_layer": {
                str(layer): next(iter(manifest_hashes[layer])) for layer in LAYERS
            },
            "framework_sha256": next(iter(framework_hashes)),
            "variant_source_sha256": {
                variant: next(iter(source_hashes[variant])) for variant in VARIANTS
            },
            "reference_source_sha256": next(iter(reference_source_hashes)),
            "source_summaries": source_summaries,
        },
        "aggregate_vs_flashinfer": aggregate,
        "aggregate_vs_torch": aggregate_torch,
        "acceptance_gates": gates,
        "cells": cells,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "qualification.json", payload)
    _write_csv(output_dir / "qualification.csv", cells)
    _write_markdown(output_dir / "qualification.md", payload)
    print(output_dir / "qualification.json")
    print(f"frozen_stack_status={gates['frozen_stack_status']}")


if __name__ == "__main__":
    main()
