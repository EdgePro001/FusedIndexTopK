#!/usr/bin/env python3
"""Resolve the exact NVTX label used by a configured variant/case."""

from __future__ import annotations

import argparse
import json

from index_topk_perflab.config import load_config
from index_topk_perflab.nvtx import (
    ncu_push_pop_filter,
    pipeline_label,
    stage_label,
)
from index_topk_perflab.provenance import variant_identity
from index_topk_perflab.registry import load_variant


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fused_index_topk_h20.json")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument(
        "--stage",
        default="pipeline",
        help="physical stage ID, or 'pipeline' for the complete graph",
    )
    parser.add_argument(
        "--format",
        choices=("label", "ncu", "json"),
        default="label",
        help="plain label, NCU push/pop filter, or a JSON description",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    case = config.make_case(args.target_length, purpose="profile")
    plugin = load_variant(
        config.variant_factory(args.variant),
        options=config.variant_options(args.variant),
    )
    variant_id = plugin.descriptor.plugin_id
    if variant_id != args.variant:
        raise ValueError(
            f"configured variant ID {args.variant!r} does not match plugin ID {variant_id!r}"
        )
    identity = variant_identity(
        config,
        args.variant,
        plugin,
        options=config.variant_options(args.variant),
    )
    fingerprint = identity["fingerprint"]
    cache_key = f"{variant_id}-{fingerprint[:16]}"
    if args.stage == "pipeline":
        label = pipeline_label(variant_id, case.case_id)
    else:
        label = stage_label(variant_id, case.case_id, args.stage)
    ncu_filter = ncu_push_pop_filter(label)

    if args.format == "label":
        print(label)
    elif args.format == "ncu":
        print(ncu_filter)
    else:
        print(
            json.dumps(
                {
                    "case_id": case.case_id,
                    "cache_key": cache_key,
                    "descriptor_fingerprint": fingerprint,
                    "label": label,
                    "ncu_filter": ncu_filter,
                    "stage": args.stage,
                    "target_length": args.target_length,
                    "variant": variant_id,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
