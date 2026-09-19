"""Independent ABBA/BAAB benchmark campaigns and campaign-level statistics."""

from __future__ import annotations

import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import canonical_hash, load_json, write_json_atomic
from .comparison import benchmark_validation_errors
from .contract import FORMAL_DECISION_POLICY, plan_identity, protocol_identity
from .runner import file_sha256, utc_now
from .summary import validate_measurements

_SAFE_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
CaseKey = tuple[int, int]


def _case_key(value: Mapping[str, Any]) -> CaseKey:
    return int(value["query_tokens"]), int(value["context_tokens"])


def _case_label(key: CaseKey) -> str:
    return f"Q={key[0]}, N={key[1]}"


def build_campaign_plan(
    config: Any,
    *,
    config_path: str | Path,
    candidate_variant: str,
    run_id: str,
    output_root: str | Path = "results/campaigns",
) -> dict[str, Any]:
    """Build the frozen run order for at least three independent campaigns."""

    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("run_id must be one safe path component")
    config.variant_factory(candidate_variant)
    baseline_variant = config.baseline_variant
    if candidate_variant == baseline_variant:
        raise ValueError("campaign candidate must differ from the configured baseline")
    orders = tuple(config.timing.campaign_orders)
    if config.timing.campaign_repetitions < 3 or len(orders) < 3:
        raise ValueError("formal comparison requires at least three campaigns")

    root = Path(output_root) / run_id
    runs: list[dict[str, Any]] = []
    for campaign_index, order in enumerate(orders, start=1):
        if order not in {"ABBA", "BAAB"}:
            raise ValueError(f"unsupported campaign order {order!r}")
        for position, arm in enumerate(order, start=1):
            variant = baseline_variant if arm == "A" else candidate_variant
            slot = f"c{campaign_index:02d}-p{position}-{arm.lower()}"
            slot_run_id = f"{run_id}-{slot}"
            artifact_path = root / "runs" / slot / variant / "benchmark.json"
            runs.append(
                {
                    "campaign": campaign_index,
                    "position": position,
                    "order": order,
                    "arm": arm,
                    "variant": variant,
                    "run_id": slot_run_id,
                    "artifact_path": str(artifact_path),
                }
            )
    payload = {
        "schema_version": 1,
        "artifact_type": "indextopk_campaign_plan",
        "created_at_utc": utc_now(),
        "run_id": run_id,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "protocol_identity": protocol_identity(config),
        "formal_plan_identity": plan_identity(config),
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "campaign_repetitions": config.timing.campaign_repetitions,
        "campaign_orders": list(orders),
        "independent_process_per_run": True,
        "decision_policy": FORMAL_DECISION_POLICY,
        "runs": runs,
    }
    payload["plan_hash"] = canonical_hash(payload)
    return payload


def write_campaign_plan(path: str | Path, plan: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite campaign plan: {destination}")
    return write_json_atomic(destination, dict(plan))


def _formal_run_summary(payload: Mapping[str, Any]) -> dict[CaseKey, dict[str, float]]:
    rows = validate_measurements(payload.get("measurements", []))
    primary_samples: dict[CaseKey, list[float]] = defaultdict(list)
    event_samples: dict[CaseKey, list[float]] = defaultdict(list)
    for row in rows:
        case_key = _case_key(row)
        if row["pass"] == "formal_kernel_sum" and row["scope_id"] == "operator_total":
            primary_samples[case_key].append(float(row["latency_ms"]))
        elif row["pass"] == "cuda_event_total" and row["scope_id"] == "operator_total":
            event_samples[case_key].append(float(row["latency_ms"]))
    result: dict[CaseKey, dict[str, float]] = {}
    if set(primary_samples) != set(event_samples):
        raise ValueError("formal and supplemental case matrices differ")
    for case_key, values in primary_samples.items():
        ordered = sorted(values)
        median = float(statistics.median(ordered))
        mad = float(statistics.median(abs(value - median) for value in ordered))
        event_ordered = sorted(event_samples[case_key])
        event_median = float(statistics.median(event_ordered))
        event_mad = float(
            statistics.median(abs(value - event_median) for value in event_ordered)
        )
        result[case_key] = {
            "count": len(ordered),
            "median_ms": median,
            "p90_ms": _quantile(ordered, 0.90),
            "p95_ms": _quantile(ordered, 0.95),
            "normalized_mad": mad / median if median else float("inf"),
            "p90_to_median_ratio": _quantile(ordered, 0.90) / median,
            "event_count": len(event_ordered),
            "event_median_ms": event_median,
            "event_p90_ms": _quantile(event_ordered, 0.90),
            "event_p95_ms": _quantile(event_ordered, 0.95),
            "event_normalized_mad": (
                event_mad / event_median if event_median else float("inf")
            ),
            "event_p90_to_median_ratio": (
                _quantile(event_ordered, 0.90) / event_median
            ),
        }
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    if len(values) == 1:
        return float(values[0])
    position = fraction * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def _campaign_bootstrap(
    values: Sequence[float],
    *,
    repetitions: int = 10_000,
    seed: int,
) -> tuple[float, float]:
    if len(values) < 3:
        raise ValueError("campaign bootstrap requires at least three independent values")
    generator = random.Random(seed)
    draws = sorted(
        statistics.median(generator.choice(values) for _ in values)
        for _ in range(repetitions)
    )
    return _quantile(draws, 0.025), _quantile(draws, 0.975)


def _atomic_case_errors(payload: Mapping[str, Any]) -> list[str]:
    """Verify every per-Q/N seal against the aggregate benchmark artifact."""

    errors: list[str] = []
    cases = {
        _case_key(item["case"]): item
        for item in payload.get("cases", [])
        if isinstance(item, Mapping) and isinstance(item.get("case"), Mapping)
    }
    records = payload.get("case_artifacts")
    if not isinstance(records, list):
        return ["per-case atomic artifact manifest is missing"]
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("per-case atomic artifact record is malformed")
            continue
        case_key = (
            int(record.get("query_tokens", -1)),
            int(record.get("context_tokens", -1)),
        )
        label = _case_label(case_key)
        path = Path(str(record.get("path", "")))
        if not path.is_file():
            errors.append(f"{label} atomic case artifact is missing: {path}")
            continue
        if path.stat().st_size != record.get("bytes") or file_sha256(path) != record.get(
            "sha256"
        ):
            errors.append(f"{label} atomic case artifact hash/size mismatch")
            continue
        try:
            sealed = load_json(path)
        except ValueError as error:
            errors.append(f"{label} atomic case artifact is invalid: {error}")
            continue
        if (
            not isinstance(sealed, Mapping)
            or sealed.get("artifact_type") != "indextopk_benchmark_case"
            or sealed.get("status") != "complete"
        ):
            errors.append(f"{label} atomic case artifact is not complete")
            continue
        if sealed.get("run_id") != payload.get("run_id"):
            errors.append(f"{label} atomic case run_id mismatch")
        if sealed.get("config_hash") != payload.get("config_hash") or sealed.get(
            "measurement_contract_hash"
        ) != payload.get("measurement_contract_hash"):
            errors.append(f"{label} atomic case contract mismatch")
        identity = payload.get("variant", {}).get("identity", {})
        if sealed.get("variant_fingerprint") != identity.get("fingerprint"):
            errors.append(f"{label} atomic case variant mismatch")
        aggregate_case = cases.get(case_key)
        if sealed.get("case") != aggregate_case:
            errors.append(f"{label} atomic case payload differs from aggregate")
        aggregate_rows = [
            row
            for row in payload.get("measurements", [])
            if _case_key(row) == case_key
        ]
        if sealed.get("measurements") != aggregate_rows:
            errors.append(f"{label} atomic measurements differ from aggregate")
        aggregate_summary = [
            row
            for row in payload.get("summary", [])
            if _case_key(row) == case_key
        ]
        if sealed.get("summary") != aggregate_summary:
            errors.append(f"{label} atomic summary differs from aggregate")
    return errors


def _input_signature(payload: Mapping[str, Any]) -> dict[CaseKey, Any]:
    return {
        _case_key(item["case"]): item.get("input_content_hash")
        or item.get("input_fingerprint")
        for item in payload.get("cases", [])
    }


def seal_campaign(plan_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Validate every slot and estimate uncertainty across campaign blocks."""

    plan = load_json(plan_path)
    if plan.get("artifact_type") != "indextopk_campaign_plan":
        raise ValueError("not an IndexTopK campaign plan")
    recorded_plan_hash = plan.get("plan_hash")
    unhashed_plan = {key: value for key, value in plan.items() if key != "plan_hash"}
    if canonical_hash(unhashed_plan) != recorded_plan_hash:
        raise ValueError("campaign plan hash does not match its contents")
    if int(plan.get("campaign_repetitions", 0)) < 3:
        raise ValueError("formal comparison requires at least three campaigns")
    policy = plan.get("decision_policy")
    if policy != FORMAL_DECISION_POLICY:
        raise ValueError("campaign plan decision policy is missing or unsupported")
    for name in ("protocol_identity", "formal_plan_identity"):
        record = plan.get(name)
        record_payload = record.get("payload") if isinstance(record, Mapping) else None
        if (
            not isinstance(record, Mapping)
            or not isinstance(record_payload, Mapping)
            or canonical_hash(record_payload) != record.get("sha256")
        ):
            raise ValueError(f"campaign plan has an invalid {name}")

    loaded: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    failures: list[str] = []
    common_contract_hash: str | None = None
    common_runtime_hash: str | None = None
    common_inputs: dict[CaseKey, Any] | None = None
    for slot in plan.get("runs", []):
        path = Path(slot["artifact_path"])
        try:
            payload = load_json(path)
        except (FileNotFoundError, ValueError) as error:
            failures.append(f"{slot.get('run_id')}: {error}")
            continue
        errors = benchmark_validation_errors(payload)
        failures.extend(f"{slot['run_id']}: {error}" for error in errors)
        if payload.get("schema_version") != 3:
            failures.append(f"{slot['run_id']}: formal campaign requires benchmark schema v3")
        failures.extend(
            f"{slot['run_id']}: {error}" for error in _atomic_case_errors(payload)
        )
        if payload.get("run_id") != slot.get("run_id"):
            failures.append(f"{slot['run_id']}: run_id mismatch")
        if payload.get("variant", {}).get("plugin_id") != slot.get("variant"):
            failures.append(f"{slot['run_id']}: variant mismatch")
        if payload.get("protocol_identity") != plan["protocol_identity"]:
            failures.append(f"{slot['run_id']}: protocol differs from campaign plan")
        if payload.get("plan_identity") != plan["formal_plan_identity"]:
            failures.append(f"{slot['run_id']}: formal plan differs from campaign plan")
        contract_hash = payload.get("measurement_contract_hash")
        runtime_hash = (payload.get("runtime_identity") or {}).get("sha256")
        input_signature = _input_signature(payload)
        common_contract_hash = common_contract_hash or contract_hash
        common_runtime_hash = common_runtime_hash or runtime_hash
        common_inputs = common_inputs or input_signature
        if contract_hash != common_contract_hash:
            failures.append(f"{slot['run_id']}: measurement contract mismatch")
        if runtime_hash != common_runtime_hash:
            failures.append(f"{slot['run_id']}: runtime identity mismatch")
        if input_signature != common_inputs:
            failures.append(f"{slot['run_id']}: input content mismatch")
        loaded.append((slot, payload))
    if len(loaded) != len(plan.get("runs", [])):
        failures.append("one or more planned runs are missing")
    if failures:
        raise ValueError("campaign cannot be sealed:\n- " + "\n- ".join(dict.fromkeys(failures)))

    by_campaign: dict[int, list[tuple[Mapping[str, Any], dict[str, Any]]]] = defaultdict(list)
    for slot, payload in loaded:
        by_campaign[int(slot["campaign"])].append((slot, _formal_run_summary(payload)))
    expected_campaigns = set(range(1, int(plan["campaign_repetitions"]) + 1))
    if set(by_campaign) != expected_campaigns:
        raise ValueError("campaign IDs do not match the plan")

    campaign_cases: dict[CaseKey, list[dict[str, Any]]] = defaultdict(list)
    for campaign, runs in sorted(by_campaign.items()):
        ordered_slots = sorted(runs, key=lambda item: int(item[0]["position"]))
        actual_order = "".join(str(slot["arm"]) for slot, _ in ordered_slots)
        expected_order = str(ordered_slots[0][0]["order"])
        if actual_order != expected_order or actual_order not in {"ABBA", "BAAB"}:
            raise ValueError(f"campaign {campaign} does not follow its frozen order")
        case_keys = set.intersection(*(set(summary) for _, summary in ordered_slots))
        for case_key in sorted(case_keys):
            arm_medians: dict[str, list[float]] = defaultdict(list)
            arm_p95: dict[str, list[float]] = defaultdict(list)
            arm_event_medians: dict[str, list[float]] = defaultdict(list)
            arm_event_p95: dict[str, list[float]] = defaultdict(list)
            for slot, summary in ordered_slots:
                arm = str(slot["arm"])
                arm_medians[arm].append(summary[case_key]["median_ms"])
                arm_p95[arm].append(summary[case_key]["p95_ms"])
                arm_event_medians[arm].append(
                    summary[case_key]["event_median_ms"]
                )
                arm_event_p95[arm].append(summary[case_key]["event_p95_ms"])
            if len(arm_medians["A"]) != 2 or len(arm_medians["B"]) != 2:
                raise ValueError(f"campaign {campaign} must contain two runs per arm")
            baseline_ms = float(statistics.median(arm_medians["A"]))
            candidate_ms = float(statistics.median(arm_medians["B"]))
            baseline_event_ms = float(statistics.median(arm_event_medians["A"]))
            candidate_event_ms = float(statistics.median(arm_event_medians["B"]))
            stability_records = [
                {
                    "arm": str(slot["arm"]),
                    "position": int(slot["position"]),
                    "count": int(summary[case_key]["count"]),
                    "normalized_mad": float(summary[case_key]["normalized_mad"]),
                    "p90_to_median_ratio": float(
                        summary[case_key]["p90_to_median_ratio"]
                    ),
                    "event_count": int(summary[case_key]["event_count"]),
                    "event_normalized_mad": float(
                        summary[case_key]["event_normalized_mad"]
                    ),
                    "event_p90_to_median_ratio": float(
                        summary[case_key]["event_p90_to_median_ratio"]
                    ),
                }
                for slot, summary in ordered_slots
            ]
            stable = all(
                record["count"] >= int(policy["minimum_trials_per_run"])
                and record["normalized_mad"] <= float(policy["maximum_normalized_mad"])
                and record["p90_to_median_ratio"]
                <= float(policy["maximum_p90_to_median_ratio"])
                and record["event_count"] >= int(policy["minimum_trials_per_run"])
                and record["event_normalized_mad"]
                <= float(policy["maximum_normalized_mad"])
                and record["event_p90_to_median_ratio"]
                <= float(policy["maximum_p90_to_median_ratio"])
                for record in stability_records
            )
            campaign_cases[case_key].append(
                {
                    "campaign": campaign,
                    "order": actual_order,
                    "baseline_median_ms": baseline_ms,
                    "candidate_median_ms": candidate_ms,
                    "speedup": baseline_ms / candidate_ms,
                    "candidate_to_baseline_p95_ratio": statistics.median(arm_p95["B"])
                    / statistics.median(arm_p95["A"]),
                    "baseline_cuda_event_median_ms": baseline_event_ms,
                    "candidate_cuda_event_median_ms": candidate_event_ms,
                    "cuda_event_speedup": baseline_event_ms / candidate_event_ms,
                    "candidate_to_baseline_cuda_event_p95_ratio": statistics.median(
                        arm_event_p95["B"]
                    )
                    / statistics.median(arm_event_p95["A"]),
                    "within_run_stability": "passed" if stable else "failed",
                    "run_stability": stability_records,
                }
            )

    cases: list[dict[str, Any]] = []
    statuses: list[str] = []
    for case_key, blocks in sorted(campaign_cases.items()):
        query_tokens, context_tokens = case_key
        speedups = [float(block["speedup"]) for block in blocks]
        p95_ratios = [float(block["candidate_to_baseline_p95_ratio"]) for block in blocks]
        event_speedups = [float(block["cuda_event_speedup"]) for block in blocks]
        event_p95_ratios = [
            float(block["candidate_to_baseline_cuda_event_p95_ratio"])
            for block in blocks
        ]
        low, high = _campaign_bootstrap(
            speedups,
            repetitions=int(policy["campaign_bootstrap_repetitions"]),
            seed=20260815 + query_tokens * 31 + context_tokens,
        )
        median_speedup = float(statistics.median(speedups))
        stable = all(block["within_run_stability"] == "passed" for block in blocks)
        if not stable:
            status = "unstable"
        elif (
            median_speedup >= float(policy["improved_minimum_speedup"])
            and low > 1.0
            and max(p95_ratios)
            <= float(policy["maximum_candidate_to_baseline_p95_ratio"])
            and max(event_p95_ratios)
            <= float(policy["maximum_candidate_to_baseline_p95_ratio"])
        ):
            status = "improved"
        elif median_speedup <= float(policy["regressed_maximum_speedup"]) and high < 1.0:
            status = "regressed"
        else:
            status = "inconclusive"
        statuses.append(status)
        cases.append(
            {
                "query_tokens": query_tokens,
                "context_tokens": context_tokens,
                "campaigns": blocks,
                "campaign_count": len(blocks),
                "median_speedup": median_speedup,
                "campaign_speedup_range": [min(speedups), max(speedups)],
                "campaign_bootstrap_95pct_ci": [low, high],
                "worst_candidate_to_baseline_p95_ratio": max(p95_ratios),
                "median_cuda_event_speedup": float(statistics.median(event_speedups)),
                "campaign_cuda_event_speedup_range": [
                    min(event_speedups),
                    max(event_speedups),
                ],
                "worst_candidate_to_baseline_cuda_event_p95_ratio": max(
                    event_p95_ratios
                ),
                "within_run_stability": "passed" if stable else "failed",
                "status": status,
            }
        )
    overall = (
        "unstable"
        if any(status == "unstable" for status in statuses)
        else "improved"
        if statuses and all(status == "improved" for status in statuses)
        else "regressed"
        if any(status == "regressed" for status in statuses)
        else "inconclusive"
    )
    result = {
        "schema_version": 2,
        "artifact_type": "indextopk_campaign",
        "status": overall,
        "sealed_at_utc": utc_now(),
        "plan": {
            "path": str(plan_path),
            "sha256": file_sha256(plan_path),
            "plan_hash": recorded_plan_hash,
        },
        "baseline_variant": plan["baseline_variant"],
        "candidate_variant": plan["candidate_variant"],
        "campaign_repetitions": plan["campaign_repetitions"],
        "decision_policy": policy,
        "uncertainty_unit": "independent_campaign_block",
        "formal_metric": "formal_kernel_sum/operator_total/kineto_cupti",
        "guardrail_metric": "cuda_event_total/operator_total/direct_cuda_event",
        "within_run_trials_used_for": [
            "primary_median",
            "primary_p95",
            "cuda_event_median",
            "cuda_event_p95",
        ],
        "within_run_bootstrap_used_for_formal_ci": False,
        "cases": cases,
    }
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite campaign artifact: {destination}")
    write_json_atomic(destination, result)
    return result
