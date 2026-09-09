from __future__ import annotations

from pathlib import Path

import pytest

from index_topk_perflab.config import load_config
from index_topk_perflab.runtime import (
    runtime_identity,
    validate_gpu_exclusivity,
    validate_runtime,
)

ROOT = Path(__file__).resolve().parents[1]


def _runtime() -> dict:
    return {
        "visible_device_count": 1,
        "selected_device": {
            "physical_uuid": "GPU-selected",
            "name": "NVIDIA H20-3e",
            "capability": [9, 0],
            "multiprocessor_count": 78,
            "total_memory_bytes": 150109880320,
        },
        "torch": "2.10.0+cu130",
        "torch_cuda_build": "13.0",
        "sources": {
            "deep_gemm": {
                "commit": "7c95b14aa4a66edd7b682e5acdde62351ca81197",
                "status_short": [],
            }
        },
        "imports": {
            "deep_gemm": {
                "package_root": "/venv/deep_gemm",
                "distribution_package_root": "/venv/deep_gemm",
                "distribution_version": "2.0.0+7c95b14",
                "wheel_record_sha256": "record",
                "extension_sha256": "binary",
            }
        },
        "compute_processes": [],
    }


def test_frozen_h20_runtime_contract_accepts_matching_installation() -> None:
    validate_runtime(
        _runtime(), load_config(ROOT / "configs" / "r13a_h20_release.json")
    )


def test_runtime_rejects_unbound_deepgemm_binary() -> None:
    runtime = _runtime()
    runtime["imports"]["deep_gemm"]["distribution_version"] = "2.0.0+deadbee"
    with pytest.raises(RuntimeError, match="not built from the frozen checkout"):
        validate_runtime(
            runtime, load_config(ROOT / "configs" / "r13a_h20_release.json")
        )


def test_runtime_ignores_process_on_nonvisible_gpu() -> None:
    runtime = _runtime()
    runtime["compute_processes"] = [
        {
            "gpu_uuid": "GPU-other",
            "pid": "12345",
            "process_name": "unrelated-work",
            "used_gpu_memory": "614",
        }
    ]
    validate_runtime(
        runtime, load_config(ROOT / "configs" / "r13a_h20_release.json")
    )


def test_runtime_rejects_process_on_selected_gpu() -> None:
    runtime = _runtime()
    runtime["compute_processes"] = [
        {
            "gpu_uuid": "GPU-selected",
            "pid": "12345",
            "process_name": "competing-work",
            "used_gpu_memory": "614",
        }
    ]
    with pytest.raises(RuntimeError, match="other compute processes"):
        validate_runtime(
            runtime, load_config(ROOT / "configs" / "r13a_h20_release.json")
        )


def test_runtime_identity_excludes_diagnostic_tool_versions() -> None:
    first = _runtime()
    first["tools"] = {"nsys": "2025.1", "ncu": "2025.1"}
    second = _runtime()
    second["tools"] = {"nsys": "2026.2", "ncu": "2026.2"}
    assert runtime_identity(first) == runtime_identity(second)


def test_per_case_gpu_exclusivity_rejects_mid_run_competitor() -> None:
    runtime = _runtime()
    runtime["process_id"] = 101
    state = {
        "compute_processes": [
            {
                "gpu_uuid": "GPU-selected",
                "pid": "202",
                "process_name": "mid-run-competitor",
                "used_gpu_memory": "30",
            }
        ]
    }
    with pytest.raises(RuntimeError, match="exclusivity gate failed"):
        validate_gpu_exclusivity(state, runtime, phase="unit:after_candidate")


def test_per_case_gpu_exclusivity_ignores_other_gpu() -> None:
    runtime = _runtime()
    runtime["process_id"] = 101
    state = {
        "compute_processes": [
            {
                "gpu_uuid": "GPU-other",
                "pid": "202",
                "process_name": "other-gpu-work",
                "used_gpu_memory": "30",
            }
        ]
    }
    result = validate_gpu_exclusivity(state, runtime, phase="unit:before_candidate")
    assert result["status"] == "passed"
