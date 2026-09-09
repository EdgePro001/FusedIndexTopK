#!/usr/bin/env python3
"""Validate R16a fast guards and hierarchical repair on one replay fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from index_topk_perflab.api import PrefillCase, RunMode
from index_topk_perflab.artifacts import write_json_atomic
from index_topk_perflab.experimental.fused_r16a.plugin import create_variant
from index_topk_perflab.inputs import make_prefill_inputs
from index_topk_perflab.replay import ReplayInputFactory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="generate deterministic standard inputs instead of loading replay data",
    )
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--split", default="test_normal")
    parser.add_argument("--fixture", choices=("A", "B"), default="A")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--force-underflow-start", type=int, default=0)
    parser.add_argument("--force-underflow-rows", type=int, default=0)
    parser.add_argument(
        "--zero-query",
        action="store_true",
        help="replace Q with zero to stress exact selection under all-equal scores",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import deep_gemm
    import torch

    if args.synthetic:
        case = PrefillCase(
            case_id=f"r16a-synthetic-n{args.context}",
            query_tokens=4096,
            context_tokens=args.context,
            top_k=2048,
            seed=args.seed,
        )
        inputs = make_prefill_inputs(case, torch.device("cuda"), fixture_id=args.fixture)
        input_source = "synthetic-standard-v2"
    else:
        if args.manifest is None:
            parser.error("--manifest is required unless --synthetic is set")
        factory = ReplayInputFactory(
            args.manifest.resolve(), split=args.split, verify_sha256=True
        )
        case = next(
            item for item in factory.cases(seed=args.seed) if item.context_tokens == args.context
        )
        inputs = factory(case, torch.device("cuda"), args.fixture)
        input_source = "frozen-replay"
    if args.zero_query:
        inputs.q.zero_()
    plugin = create_variant({"verbose_build": False})
    graph = plugin.prepare(case, inputs, options=plugin.options, mode=RunMode.CORRECTNESS)

    physical = deep_gemm.fp8_mqa_logits(
        inputs.q,
        (inputs.kv, inputs.kv_scales),
        inputs.weights,
        inputs.k_start,
        inputs.k_end,
        clean_logits=True,
    )
    reference_values, reference_ids = torch.topk(physical, case.top_k, dim=1, sorted=False)
    reference_values = torch.sort(reference_values, dim=1, descending=True).values

    artifacts: dict[str, Any] = dict(graph.initial_artifacts)
    fast_failure_flags = None
    for node in graph.nodes:
        if node.spec.stage_id == "candidate_producer" and args.force_underflow_rows:
            force_end = args.force_underflow_start + args.force_underflow_rows
            if not 0 <= args.force_underflow_start < force_end <= case.query_tokens:
                raise ValueError("forced underflow range lies outside query rows")
            artifacts["thresholds"][args.force_underflow_start : force_end].fill_(torch.inf)
        node.run(None, artifacts)
        if node.spec.stage_id == "candidate_reducer":
            torch.cuda.synchronize()
            fast_failure_flags = artifacts["fast_failure_flags"].clone()

    torch.cuda.synchronize()
    output_ids = artifacts["indices"].squeeze(1)
    safe_ids = output_ids.to(torch.int64).clamp(0, physical.shape[1] - 1)
    output_values = physical.gather(1, safe_ids)
    output_values.masked_fill_(output_ids < 0, -torch.inf)
    output_values = torch.sort(output_values, dim=1, descending=True).values
    mismatch_rows = torch.any(output_values != reference_values, dim=1)
    duplicate_rows = torch.any(
        torch.sort(output_ids, dim=1).values[:, 1:] == torch.sort(output_ids, dim=1).values[:, :-1],
        dim=1,
    )
    unresolved = artifacts["failure_flags"]
    repair_errors = artifacts.get("repair_error_flags", torch.zeros_like(unresolved))
    first_mismatch = torch.nonzero(mismatch_rows).flatten()[:8]
    details = []
    for row_tensor in first_mismatch:
        row = int(row_tensor.item())
        expected = reference_ids[row]
        actual = output_ids[row]
        details.append(
            {
                "row": row,
                "fast_failed": int(fast_failure_flags[row].item()),
                "unresolved": int(unresolved[row].item()),
                "repair_error": int(repair_errors[row].item()),
                "expected_id_minmax": [int(expected.min().item()), int(expected.max().item())],
                "actual_id_minmax": [int(actual.min().item()), int(actual.max().item())],
                "value_mismatches": int(
                    torch.count_nonzero(output_values[row] != reference_values[row]).item()
                ),
            }
        )
    result = {
        "context": case.context_tokens,
        "input_source": input_source,
        "split": args.split,
        "fixture": args.fixture,
        "sample_elements": graph.metadata["sample_elements"],
        "repair_chunks": graph.metadata.get("repair_chunks", 0),
        "forced_underflow_rows": args.force_underflow_rows,
        "forced_underflow_start": args.force_underflow_start,
        "zero_query": args.zero_query,
        "fast_failure_rows": int(torch.count_nonzero(fast_failure_flags).item()),
        "unresolved_failure_rows": int(torch.count_nonzero(unresolved).item()),
        "repair_error_rows": int(torch.count_nonzero(repair_errors & fast_failure_flags).item()),
        "value_mismatch_rows": int(torch.count_nonzero(mismatch_rows).item()),
        "duplicate_output_rows": int(torch.count_nonzero(duplicate_rows).item()),
        "first_mismatches": details,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        write_json_atomic(args.output.resolve(), result)
    if result["unresolved_failure_rows"] or result["value_mismatch_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
