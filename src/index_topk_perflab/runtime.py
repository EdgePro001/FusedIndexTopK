"""Runtime inventory and frozen-target gates for formal GPU measurements."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .artifacts import canonical_hash


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deepgemm_installation(deep_gemm: Any, deep_gemm_cpp: Any) -> dict[str, Any]:
    """Bind the imported Python package, extension, and wheel manifest together."""

    distribution = importlib.metadata.distribution("deep_gemm")
    record = distribution.read_text("RECORD")
    module_path = Path(deep_gemm.__file__).resolve()
    extension_path = Path(deep_gemm_cpp.__file__).resolve()
    located_package = Path(distribution.locate_file("deep_gemm")).resolve()
    return {
        "module_path": str(module_path),
        "package_root": str(module_path.parent),
        "distribution_package_root": str(located_package),
        "distribution_version": distribution.version,
        "wheel_record_sha256": (
            hashlib.sha256(record.encode("utf-8")).hexdigest()
            if record is not None
            else None
        ),
        "extension_path": str(extension_path),
        "extension_bytes": extension_path.stat().st_size,
        "extension_sha256": _sha256_file(extension_path),
    }


def _run(command: Iterable[str], *, check: bool = False) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if check:
            raise RuntimeError(f"command failed: {' '.join(command)}") from error
        return ""
    return completed.stdout.strip()


def _tool_version(command: str, *arguments: str) -> str | None:
    output = _run((command, *arguments))
    return output.splitlines()[0] if output else None


def _git_state(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    repository = Path(path).resolve()
    if not repository.is_dir():
        return {"path": str(repository), "error": "directory does not exist"}
    commit = _run(("git", "-C", str(repository), "rev-parse", "HEAD"))
    status = _run(("git", "-C", str(repository), "status", "--short"))
    return {
        "path": str(repository),
        "commit": commit or None,
        "status_short": status.splitlines() if status else [],
    }


def _nvidia_smi() -> list[dict[str, str]]:
    fields = (
        "index",
        "uuid",
        "name",
        "compute_cap",
        "memory.total",
        "driver_version",
        "pstate",
        "persistence_mode",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks.current.memory",
        "temperature.gpu",
    )
    output = _run(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values, strict=True)))
    return rows


def _compute_processes() -> list[dict[str, str]]:
    fields = ("gpu_uuid", "pid", "process_name", "used_gpu_memory")
    output = _run(
        (
            "nvidia-smi",
            f"--query-compute-apps={','.join(fields)}",
            "--format=csv,noheader,nounits",
        )
    )
    if not output or "No running processes" in output:
        return []
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",", maxsplit=len(fields) - 1)]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values, strict=True)))
    return rows


def _selected_physical_gpu_uuid(
    gpu_inventory: Iterable[dict[str, str]],
    compute_processes: Iterable[dict[str, str]],
) -> str | None:
    """Resolve the one CUDA-visible device to its physical GPU UUID.

    ``nvidia-smi --query-compute-apps`` is host-wide even when CUDA exposes a
    single device.  Once PyTorch has initialized CUDA, the current PID is the
    strongest mapping from the logical CUDA device to its physical UUID.  The
    CUDA_VISIBLE_DEVICES token is retained as a conservative fallback for the
    short interval in which the driver has not yet published the process row.
    """

    current_pid = str(os.getpid())
    for process in compute_processes:
        if process.get("pid") == current_pid and process.get("gpu_uuid"):
            return process["gpu_uuid"]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if len(tokens) != 1:
        return None
    token = tokens[0]
    if token.startswith(("GPU-", "MIG-")):
        return token
    if token.isdigit():
        for gpu in gpu_inventory:
            if gpu.get("index") == token and gpu.get("uuid"):
                return gpu["uuid"]
    return None


def collect_gpu_state() -> dict[str, Any]:
    """Collect lightweight clock/power/temperature state outside measurements."""

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "nvidia_smi": _nvidia_smi(),
        "compute_processes": _compute_processes(),
    }


def validate_gpu_exclusivity(
    state: dict[str, Any],
    runtime: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """Reject a per-case snapshot containing another process on the selected GPU."""

    selected_gpu_uuid = (runtime.get("selected_device") or {}).get("physical_uuid")
    if not selected_gpu_uuid:
        raise RuntimeError(f"GPU exclusivity gate at {phase} has no selected GPU UUID")
    expected_pid = str(runtime.get("process_id", os.getpid()))
    selected_processes = [
        process
        for process in state.get("compute_processes", [])
        if process.get("gpu_uuid") == selected_gpu_uuid
    ]
    other_processes = [
        process
        for process in selected_processes
        if str(process.get("pid")) != expected_pid
    ]
    if other_processes:
        raise RuntimeError(
            f"GPU exclusivity gate failed at {phase}: other compute processes "
            f"are using the selected GPU: {other_processes}"
        )
    return {
        "status": "passed",
        "phase": phase,
        "selected_gpu_uuid": selected_gpu_uuid,
        "expected_pid": expected_pid,
        "selected_gpu_processes": selected_processes,
        "other_processes": [],
    }


def collect_runtime(*, deepgemm_source: str | Path | None = None) -> dict[str, Any]:
    """Collect the execution environment; this function intentionally initializes CUDA."""

    import deep_gemm
    import deep_gemm_cpp
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; H20 experiments must run through scripts/run_h20.sh"
        )
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    source = deepgemm_source or os.environ.get("DEEPGEMM_SOURCE")
    gpu_inventory = _nvidia_smi()
    compute_processes = _compute_processes()
    physical_uuid = _selected_physical_gpu_uuid(gpu_inventory, compute_processes)
    return {
        "process_id": os.getpid(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(),
        "selected_device": {
            "logical_index": device,
            "physical_uuid": physical_uuid,
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        },
        "nvidia_smi": gpu_inventory,
        "compute_processes": compute_processes,
        "imports": {
            "deep_gemm": _deepgemm_installation(deep_gemm, deep_gemm_cpp),
        },
        "tools": {
            "nvcc": _tool_version(os.environ.get("CUDACXX", "nvcc"), "--version"),
            "nsys": _tool_version("nsys", "--version"),
            "ncu": _tool_version("ncu", "--version"),
        },
        "sources": {"deep_gemm": _git_state(source)},
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "CUDACXX",
                "DEEPGEMM_SOURCE",
                "DG_JIT_CACHE_DIR",
                "DG_JIT_NVCC_COMPILER",
                "DG_JIT_USE_NVRTC",
                "TORCH_EXTENSIONS_DIR",
                "TRITON_CACHE_DIR",
                "PYTORCH_ALLOC_CONF",
                "ITK_VARIANT_FINGERPRINT",
            )
        },
    }


def runtime_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    """Return the compute identity without telemetry or diagnostic tool versions."""

    device = runtime.get("selected_device") or {}
    imports = runtime.get("imports") or {}
    deep_gemm = imports.get("deep_gemm") or {}
    sources = runtime.get("sources") or {}
    deep_gemm_source = sources.get("deep_gemm") or {}
    inventory = runtime.get("nvidia_smi") or []
    selected_smi = next(
        (
            row
            for row in inventory
            if row.get("uuid") == device.get("physical_uuid")
        ),
        inventory[0] if inventory else {},
    )
    payload = {
        "schema_version": 1,
        "python": runtime.get("python"),
        "torch": runtime.get("torch"),
        "torch_cuda_build": runtime.get("torch_cuda_build"),
        "device": {
            key: device.get(key)
            for key in (
                "physical_uuid",
                "name",
                "capability",
                "total_memory_bytes",
                "multiprocessor_count",
            )
        },
        "driver_version": selected_smi.get("driver_version"),
        "deep_gemm": {
            "commit": deep_gemm_source.get("commit"),
            "distribution_version": deep_gemm.get("distribution_version"),
            "wheel_record_sha256": deep_gemm.get("wheel_record_sha256"),
            "extension_sha256": deep_gemm.get("extension_sha256"),
        },
    }
    return {"sha256": canonical_hash(payload), "payload": payload}


def validate_runtime(runtime: dict[str, Any], config: Any) -> None:
    """Reject measurements that do not match the target frozen by the config."""

    errors: list[str] = []
    target = config.target
    device = runtime["selected_device"]
    if target.require_single_visible_gpu and runtime["visible_device_count"] != 1:
        errors.append(
            f"expected exactly one visible GPU, got {runtime['visible_device_count']}"
        )
    if target.gpu_name != device["name"]:
        errors.append(
            f"expected GPU name {target.gpu_name!r}, got {device['name']!r}"
        )
    if list(target.compute_capability) != device["capability"]:
        errors.append(
            f"expected compute capability {list(target.compute_capability)}, "
            f"got {device['capability']}"
        )
    if target.multiprocessor_count != device["multiprocessor_count"]:
        errors.append(
            f"expected {target.multiprocessor_count} SMs, "
            f"got {device['multiprocessor_count']}"
        )
    if target.total_memory_bytes != device["total_memory_bytes"]:
        errors.append(
            f"expected {target.total_memory_bytes} bytes of GPU memory, "
            f"got {device['total_memory_bytes']}"
        )
    try:
        cuda_build = tuple(int(part) for part in runtime["torch_cuda_build"].split(".")[:2])
    except (AttributeError, TypeError, ValueError):
        errors.append(f"could not parse torch CUDA build {runtime.get('torch_cuda_build')!r}")
    else:
        if cuda_build < (12, 8):
            errors.append(f"expected a CUDA >=12.8 PyTorch build, got {cuda_build}")
    if runtime.get("torch") != config.sources.torch_version:
        errors.append(
            f"PyTorch version mismatch: expected {config.sources.torch_version!r}, "
            f"got {runtime.get('torch')!r}"
        )
    source = runtime.get("sources", {}).get("deep_gemm") or {}
    expected_commit = config.sources.deep_gemm_commit
    if source.get("commit") != expected_commit:
        errors.append(
            f"DeepGEMM commit mismatch: expected {expected_commit}, got {source.get('commit')}"
        )
    if source.get("status_short"):
        errors.append("DeepGEMM source checkout is dirty")
    installation = runtime.get("imports", {}).get("deep_gemm") or {}
    package_root = installation.get("package_root")
    distribution_root = installation.get("distribution_package_root")
    if package_root != distribution_root:
        errors.append(
            "imported deep_gemm package does not match its installed distribution: "
            f"module={package_root!r}, distribution={distribution_root!r}"
        )
    installed_version = str(installation.get("distribution_version") or "")
    local_revision = installed_version.partition("+")[2]
    if len(local_revision) < 7 or not expected_commit.startswith(local_revision):
        errors.append(
            "installed DeepGEMM distribution is not built from the frozen checkout: "
            f"expected commit {expected_commit}, distribution version={installed_version!r}"
        )
    if not installation.get("wheel_record_sha256"):
        errors.append("installed DeepGEMM distribution has no RECORD fingerprint")
    if not installation.get("extension_sha256"):
        errors.append("imported deep_gemm_cpp extension has no binary fingerprint")
    other_processes = []
    selected_gpu_uuid = device.get("physical_uuid")
    for process in runtime.get("compute_processes", []):
        if (
            selected_gpu_uuid is not None
            and process.get("gpu_uuid") != selected_gpu_uuid
        ):
            continue
        try:
            if int(process["pid"]) != os.getpid():
                other_processes.append(process)
        except (KeyError, TypeError, ValueError):
            other_processes.append(process)
    if other_processes:
        errors.append(f"other compute processes are using the visible GPU: {other_processes}")
    if errors:
        raise RuntimeError("runtime gate failed:\n- " + "\n- ".join(errors))
