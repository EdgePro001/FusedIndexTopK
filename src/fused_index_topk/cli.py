"""Command-line entry point for correctness, formal timing, and comparison."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .campaign import build_campaign_plan, seal_campaign, write_campaign_plan
from .comparison import write_comparison
from .config import load_config
from .contract import resolved_problem_contract
from .provenance import variant_identity
from .registry import available_variants, load_variant
from .runner import default_run_id, run_benchmark, run_correctness

DEFAULT_CONFIG = Path("configs/fused_index_topk_h20.json")


def _default_artifact(run_id: str, variant: str, filename: str) -> Path:
    root = Path(os.environ.get("ITK_ARTIFACT_ROOT", "results"))
    return root / "raw" / run_id / variant / filename


def _variants(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    rows = []
    for variant_id, item in sorted(config.variants.items()):
        plugin = load_variant(item.factory, options=item.options)
        if plugin.descriptor.plugin_id != variant_id:
            raise ValueError(
                f"configured ID {variant_id!r} does not match plugin ID "
                f"{plugin.descriptor.plugin_id!r}"
            )
        rows.append(
            {
                "configured_id": variant_id,
                "factory": item.factory,
                "options": dict(item.options),
                "descriptor": asdict(plugin.descriptor),
                "identity": variant_identity(
                    config, variant_id, plugin, options=item.options
                ),
                "exact_reference": variant_id == config.exact_reference_variant,
                "baseline": variant_id == config.baseline_variant,
            }
        )
    payload = {
        "configured": rows,
        "installed_or_builtin": available_variants(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _contract(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(
        json.dumps(
            resolved_problem_contract(config),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def _check(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_id = args.run_id or default_run_id("correctness")
    output = args.output or _default_artifact(run_id, args.variant, "correctness.json")
    result = run_correctness(
        config,
        config_path=args.config,
        variant_id=args.variant,
        run_id=run_id,
        output_path=output,
    )
    print(
        f"correctness={result['status']} variant={result['variant']['plugin_id']} "
        f"cases={len(result['cases'])} artifact={output}"
    )


def _bench(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    run_id = args.run_id or default_run_id("benchmark")
    output = args.output or _default_artifact(run_id, args.variant, "benchmark.json")
    result = run_benchmark(
        config,
        config_path=args.config,
        variant_id=args.variant,
        run_id=run_id,
        correctness_path=args.correctness,
        output_path=output,
    )
    formal = {
        (row["query_tokens"], row["context_tokens"]): row
        for row in result["summary"]
        if row["pass"] == "formal_kernel_sum"
    }
    event = {
        (row["query_tokens"], row["context_tokens"]): row
        for row in result["summary"]
        if row["pass"] == "cuda_event_total"
    }
    print(f"benchmark={result['status']} artifact={output}")
    for key in sorted(formal):
        row = formal[key]
        event_row = event[key]
        print(
            f"  Q={row['query_tokens']:>4} N={row['context_tokens']:>7} "
            f"CUPTI-kernel-sum median={row['median_ms']:.6f} ms "
            f"p95={row['p95_ms']:.6f} ms n={row['count']} | "
            f"CUDA-Event median={event_row['median_ms']:.6f} ms "
            f"p95={event_row['p95_ms']:.6f} ms n={event_row['count']}"
        )


def _compare(args: argparse.Namespace) -> None:
    result = write_comparison(args.output, args.baseline, args.candidate)
    print(f"comparison={result['status']} artifact={args.output}")
    for case in result.get("cases", []):
        print(
            f"  Q={case['query_tokens']:>4} N={case['context_tokens']:>7} "
            f"speedup={case['speedup']:.4f}x "
            f"CI95=[{case['bootstrap_95pct_speedup_ci'][0]:.4f}, "
            f"{case['bootstrap_95pct_speedup_ci'][1]:.4f}] "
            f"event={case['cuda_event_speedup']:.4f}x {case['status']}"
        )


def _campaign_plan(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output = args.output or Path("results") / "campaigns" / args.run_id / "plan.json"
    plan = build_campaign_plan(
        config,
        config_path=args.config,
        candidate_variant=args.candidate,
        run_id=args.run_id,
        output_root=args.output_root,
    )
    write_campaign_plan(output, plan)
    print(f"campaign_plan=created runs={len(plan['runs'])} artifact={output}")


def _campaign_seal(args: argparse.Namespace) -> None:
    result = seal_campaign(args.plan, args.output)
    print(f"campaign={result['status']} artifact={args.output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    variants = subparsers.add_parser("variants", help="list configured and installed variants")
    variants.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    variants.set_defaults(handler=_variants)

    contract = subparsers.add_parser(
        "contract", help="print the resolved input/score/TopK/measurement contract"
    )
    contract.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    contract.set_defaults(handler=_contract)

    check = subparsers.add_parser("check", help="run the correctness gate on H20")
    check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    check.add_argument("--variant", required=True)
    check.add_argument("--run-id")
    check.add_argument("--output", type=Path)
    check.set_defaults(handler=_check)

    bench = subparsers.add_parser(
        "bench",
        help="run formal Kineto/CUPTI timing plus the CUDA-event guardrail on H20",
    )
    bench.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    bench.add_argument("--variant", required=True)
    bench.add_argument("--run-id")
    bench.add_argument("--correctness", type=Path, required=True)
    bench.add_argument("--output", type=Path)
    bench.set_defaults(handler=_bench)

    compare = subparsers.add_parser("compare", help="compare two formal benchmark artifacts")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=_compare)

    campaign_plan = subparsers.add_parser(
        "campaign-plan", help="freeze an independent ABBA/BAAB run schedule"
    )
    campaign_plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    campaign_plan.add_argument("--candidate", required=True)
    campaign_plan.add_argument("--run-id", required=True)
    campaign_plan.add_argument("--output-root", type=Path, default=Path("results/campaigns"))
    campaign_plan.add_argument("--output", type=Path)
    campaign_plan.set_defaults(handler=_campaign_plan)

    campaign_seal = subparsers.add_parser(
        "campaign-seal", help="validate and seal independent campaign results"
    )
    campaign_seal.add_argument("--plan", type=Path, required=True)
    campaign_seal.add_argument("--output", type=Path, required=True)
    campaign_seal.set_defaults(handler=_campaign_seal)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
