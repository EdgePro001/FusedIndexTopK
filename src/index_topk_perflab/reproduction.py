"""One-command H20 smoke reproduction orchestration.

This module deliberately uses only the Python standard library so that the
preflight gate can run before the project environment has been created.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifacts import canonical_hash, load_json, write_json_atomic

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
NVCC_RELEASE = re.compile(r"\brelease\s+(\d+\.\d+)")
DEFAULT_CONFIG = Path("configs/fused_index_topk_h20.json")
STAGE_NAMES = (
    "preflight",
    "environment_setup",
    "lint",
    "unit_tests",
    "variants",
    "baseline_correctness",
    "candidate_correctness",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_capture(command: Sequence[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "status": "passed" if passed else "failed", "detail": detail})


def _visible_selector(environment: dict[str, str]) -> str:
    value = environment.get("CUDA_VISIBLE_DEVICES", "0").strip()
    return value


def parse_nvidia_smi_rows(output: str) -> list[dict[str, Any]]:
    """Parse the stable CSV query emitted by the preflight nvidia-smi call."""

    rows: list[dict[str, Any]] = []
    for fields in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(fields) != 5:
            continue
        index, name, uuid, memory_mib, driver = (field.strip() for field in fields)
        try:
            memory_bytes = int(memory_mib) * 1024 * 1024
        except ValueError:
            memory_bytes = 0
        rows.append(
            {
                "index": index,
                "name": name,
                "uuid": uuid,
                "memory_total_mib": int(memory_mib) if memory_bytes else None,
                "memory_total_bytes": memory_bytes,
                "driver_version": driver,
            }
        )
    return rows


def select_visible_gpu(rows: Sequence[dict[str, Any]], selector: str) -> dict[str, Any] | None:
    if "," in selector or not selector:
        return None
    for row in rows:
        if selector in (str(row.get("index")), str(row.get("uuid"))):
            return dict(row)
    return None


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def collect_preflight(
    *,
    project_root: Path,
    config_path: Path,
    runtime_root: Path,
    cuda_home: Path,
    min_free_gib: float,
    environment: dict[str, str] | None = None,
    capture: Callable[..., dict[str, Any]] = _run_capture,
) -> dict[str, Any]:
    """Collect a machine-readable preflight record without importing CUDA packages."""

    environment = dict(os.environ if environment is None else environment)
    config = load_json(config_path)
    target = config["target"]
    checks: list[dict[str, Any]] = []

    python_ok = sys.version_info[:2] == (3, 12)
    _check(checks, "python_3_12", python_ok, platform.python_version())

    tools: dict[str, str | None] = {}
    for tool in ("git", "curl", "apt-get", "dpkg-deb", "nvidia-smi"):
        tools[tool] = shutil.which(tool, path=environment.get("PATH"))
        _check(checks, f"tool_{tool}", tools[tool] is not None, tools[tool] or "not found")

    nvcc = cuda_home / "bin" / "nvcc"
    nvcc_result = capture([str(nvcc), "--version"], env=environment)
    nvcc_text = f"{nvcc_result.get('stdout', '')}\n{nvcc_result.get('stderr', '')}"
    match = NVCC_RELEASE.search(nvcc_text)
    nvcc_release = match.group(1) if match else None
    _check(
        checks,
        "cuda_toolkit_13_0",
        nvcc_result.get("returncode") == 0 and nvcc_release == "13.0",
        nvcc_release or nvcc_text.strip(),
    )

    selector = _visible_selector(environment)
    _check(
        checks,
        "single_visible_gpu_selector",
        bool(selector) and "," not in selector,
        selector or "empty",
    )
    smi_command = [
        tools.get("nvidia-smi") or "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    smi_result = capture(smi_command, env=environment)
    gpu_rows = parse_nvidia_smi_rows(smi_result.get("stdout", ""))
    selected_gpu = select_visible_gpu(gpu_rows, selector)
    _check(
        checks,
        "selected_gpu_resolved",
        smi_result.get("returncode") == 0 and selected_gpu is not None,
        selected_gpu or smi_result.get("stderr", "unresolved"),
    )
    if selected_gpu is not None:
        _check(
            checks,
            "gpu_name",
            selected_gpu["name"] == target["gpu_name"],
            {"expected": target["gpu_name"], "actual": selected_gpu["name"]},
        )
        minimum_memory = int(target["total_memory_bytes"] * 0.99)
        _check(
            checks,
            "gpu_memory",
            selected_gpu["memory_total_bytes"] >= minimum_memory,
            {
                "expected_minimum_bytes": minimum_memory,
                "actual_bytes": selected_gpu["memory_total_bytes"],
            },
        )
        _check(
            checks,
            "driver_detected",
            bool(selected_gpu["driver_version"]),
            selected_gpu["driver_version"],
        )
    else:
        _check(checks, "gpu_name", False, {"expected": target["gpu_name"], "actual": None})
        _check(
            checks,
            "gpu_memory",
            False,
            {"expected_bytes": target["total_memory_bytes"], "actual_bytes": None},
        )
        _check(checks, "driver_detected", False, None)

    storage_parent = _nearest_existing_parent(runtime_root)
    writable = storage_parent.is_dir() and os.access(storage_parent, os.W_OK)
    _check(checks, "runtime_root_writable", writable, str(storage_parent))
    try:
        free_bytes = shutil.disk_usage(storage_parent).free
    except OSError:
        free_bytes = 0
    required_bytes = int(min_free_gib * 1024**3)
    _check(
        checks,
        "runtime_root_free_space",
        free_bytes >= required_bytes,
        {"required_bytes": required_bytes, "free_bytes": free_bytes},
    )

    git_result = capture(["git", "status", "--short"], env=environment)
    revision_result = capture(["git", "rev-parse", "HEAD"], env=environment)
    git_state = {
        "revision": revision_result.get("stdout", "").strip() or None,
        "dirty": bool(git_result.get("stdout", "").strip()),
        "status_short": git_result.get("stdout", "").splitlines(),
    }
    return {
        "schema_version": 1,
        "artifact_type": "fused_index_topk_preflight",
        "created_at_utc": utc_now(),
        "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
        "project_root": str(project_root),
        "config": str(config_path),
        "config_sha256": canonical_hash(config),
        "runtime_root": str(runtime_root),
        "cuda_home": str(cuda_home),
        "cuda_visible_devices": selector,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "tools": tools,
        "nvcc_release": nvcc_release,
        "gpus": gpu_rows,
        "selected_gpu": selected_gpu,
        "git": git_state,
        "checks": checks,
    }


def build_smoke_config(config: dict[str, Any], *, context_tokens: int = 6144) -> dict[str, Any]:
    """Narrow a frozen release config to one correctness case.

    ``parse_config`` requires every profiling case to remain a subset of
    ``workload.benchmark_cases``, so the profiling matrix is narrowed with the
    workload rather than left pointing at the dropped context lengths.
    """

    smoke = json.loads(json.dumps(config))
    cases = [
        case
        for case in smoke["workload"]["correctness_cases"]
        if int(case["context_tokens"]) == context_tokens
    ]
    if len(cases) != 1:
        raise ValueError(f"config must contain exactly one N={context_tokens} correctness case")
    smoke["name"] = f"{smoke['name']}_smoke_n{context_tokens}"
    smoke["workload"]["correctness_cases"] = cases
    smoke["workload"]["benchmark_cases"] = [dict(cases[0])]
    profiling = smoke.get("profiling")
    if isinstance(profiling, dict):
        for field in ("nsys_cases", "ncu_cases"):
            declared = profiling.get(field)
            if not isinstance(declared, list):
                continue
            retained = [
                case
                for case in declared
                if isinstance(case, dict) and int(case.get("context_tokens", -1)) == context_tokens
            ]
            profiling[field] = retained or [dict(cases[0])]
    return smoke


def render_report(run: dict[str, Any]) -> str:
    lines = [
        "# FusedIndexTopK smoke reproduction",
        "",
        f"- Status: **{run['status']}**",
        f"- Run ID: `{run['run_id']}`",
        f"- Started: `{run['started_at_utc']}`",
        f"- Completed: `{run.get('completed_at_utc', 'incomplete')}`",
        f"- Reproduction level: `{run['level']}`",
        "",
        "## Stages",
        "",
        "| Stage | Status | Duration (s) | Log |",
        "|---|---|---:|---|",
    ]
    for stage in run["stages"]:
        duration = stage.get("duration_seconds")
        duration_text = f"{duration:.3f}" if isinstance(duration, (float, int)) else "-"
        log = stage.get("log")
        log_text = f"`{log}`" if log else "-"
        lines.append(f"| {stage['name']} | {stage['status']} | {duration_text} | {log_text} |")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This smoke run validates environment setup, CPU control-plane checks, "
            "variant discovery, and exact correctness at Q=4096/N=6144 for both "
            "the FlashInfer baseline and FusedIndexTopK candidate. It does not "
            "reproduce the published held-out replay performance numbers.",
            "",
        ]
    )
    git = run.get("source", {})
    if git:
        lines.extend(
            [
                "## Source",
                "",
                f"- Git revision: `{git.get('revision')}`",
                f"- Dirty worktree: `{git.get('dirty')}`",
                "",
            ]
        )
    if run.get("failure"):
        lines.extend(["## Failure", "", f"`{run['failure']}`", ""])
    return "\n".join(lines)


class Reproducer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = (
            (self.project_root / args.config).resolve()
            if not args.config.is_absolute()
            else args.config.resolve()
        )
        self.runtime_root = args.runtime_root.resolve()
        self.output_root = args.artifact_root.resolve() / args.run_id
        self.logs = self.output_root / "logs"
        self.run_path = self.output_root / "report.json"
        self.run: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "fused_index_topk_reproduction",
            "run_id": args.run_id,
            "level": args.level,
            "status": "running",
            "started_at_utc": utc_now(),
            "project_root": str(self.project_root),
            "runtime_root": str(self.runtime_root),
            "config": str(self.config_path),
            "stages": [],
        }

    def _persist(self) -> None:
        write_json_atomic(self.run_path, self.run)
        (self.output_root / "REPORT.md").write_text(render_report(self.run), encoding="utf-8")

    def _stage(self, name: str, command: Sequence[str], *, artifact: Path | None = None) -> bool:
        log_path = self.logs / f"{name}.log"
        stage = {
            "name": name,
            "status": "running",
            "started_at_utc": utc_now(),
            "command": list(command),
            "log": str(log_path.relative_to(self.output_root)),
        }
        if artifact is not None:
            stage["artifact"] = str(artifact.relative_to(self.output_root))
        self.run["stages"].append(stage)
        self._persist()
        print(f"==> {name}", flush=True)
        started = time.monotonic()
        environment = dict(os.environ)
        environment["ITK_RUNTIME_ROOT"] = str(self.runtime_root)
        environment["CUDA_HOME"] = str(self.args.cuda_home)
        environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
        with log_path.open("w", encoding="utf-8") as log:
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=self.project_root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as error:
                log.write(f"could not start command: {error}\n")
                returncode = 127
            else:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                returncode = process.wait()
        stage["completed_at_utc"] = utc_now()
        stage["duration_seconds"] = time.monotonic() - started
        stage["returncode"] = returncode
        stage["status"] = "passed" if returncode == 0 else "failed"
        self._persist()
        return returncode == 0

    def execute(self) -> int:
        if self.output_root.exists():
            raise FileExistsError(f"refusing to overwrite reproduction run: {self.output_root}")
        self.logs.mkdir(parents=True)
        try:
            preflight_path = self.output_root / "preflight.json"
            started = time.monotonic()
            preflight = collect_preflight(
                project_root=self.project_root,
                config_path=self.config_path,
                runtime_root=self.runtime_root,
                cuda_home=self.args.cuda_home,
                min_free_gib=self.args.min_free_gib,
            )
            write_json_atomic(preflight_path, preflight)
            self.run["source"] = preflight["git"]
            self.run["stages"].append(
                {
                    "name": "preflight",
                    "status": preflight["status"],
                    "started_at_utc": preflight["created_at_utc"],
                    "completed_at_utc": utc_now(),
                    "duration_seconds": time.monotonic() - started,
                    "artifact": "preflight.json",
                }
            )
            self._persist()
            if preflight["status"] != "passed":
                failed = [
                    item["name"]
                    for item in preflight["checks"]
                    if item["status"] == "failed"
                ]
                raise RuntimeError("preflight failed: " + ", ".join(failed))

            smoke_config_path = self.output_root / "smoke-config.json"
            write_json_atomic(smoke_config_path, build_smoke_config(load_json(self.config_path)))
            runner = self.project_root / "scripts" / "run_h20.sh"
            if not self.args.skip_setup and not self._stage(
                "environment_setup", [str(self.project_root / "scripts" / "setup_h20.sh")]
            ):
                raise RuntimeError("environment setup failed")
            if self.args.skip_setup:
                self.run["stages"].append(
                    {
                        "name": "environment_setup",
                        "status": "skipped",
                        "reason": "--skip-setup",
                    }
                )
                self._persist()

            commands = [
                ("lint", [str(runner), "python", "-m", "ruff", "check", "."], None),
                ("unit_tests", [str(runner), "python", "-m", "pytest", "-q"], None),
                (
                    "variants",
                    [
                        str(runner),
                        "python",
                        "-m",
                        "index_topk_perflab.cli",
                        "variants",
                        "--config",
                        str(smoke_config_path),
                    ],
                    None,
                ),
            ]
            for name, command, artifact in commands:
                if not self._stage(name, command, artifact=artifact):
                    raise RuntimeError(f"{name} failed")

            baseline = load_json(smoke_config_path)["baseline_variant"]
            candidate = "fused_index_topk"
            for stage_name, variant in (
                ("baseline_correctness", baseline),
                ("candidate_correctness", candidate),
            ):
                artifact = self.output_root / "correctness" / variant / "correctness.json"
                command = [
                    str(runner), "python", "-m", "index_topk_perflab.cli", "check",
                    "--config", str(smoke_config_path), "--variant", variant,
                    "--run-id", f"{self.args.run_id}-{variant}", "--output", str(artifact),
                ]
                if not self._stage(stage_name, command, artifact=artifact):
                    raise RuntimeError(f"{stage_name} failed")

            self.run["status"] = "passed"
            returncode = 0
        except KeyboardInterrupt:
            self.run["status"] = "interrupted"
            self.run["failure"] = "KeyboardInterrupt"
            self.run["completed_at_utc"] = utc_now()
            self._persist()
            raise
        except Exception as error:
            self.run["status"] = "failed"
            self.run["failure"] = f"{type(error).__name__}: {error}"
            returncode = 1
        self.run["completed_at_utc"] = utc_now()
        self._persist()
        print(
            f"reproduction={self.run['status']} "
            f"report={self.output_root / 'REPORT.md'}",
            flush=True,
        )
        return returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("smoke",), default="smoke")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(
            os.environ.get("ITK_RUNTIME_ROOT", f"/data/{os.environ.get('USER', 'unknown')}")
        ),
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0")),
    )
    parser.add_argument(
        "--run-id",
        default=f"smoke-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="reuse an already-created frozen H20 runtime",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if not SAFE_ID.fullmatch(args.run_id):
        parser.error("--run-id must be one safe path component")
    if args.min_free_gib < 0:
        parser.error("--min-free-gib must be non-negative")
    if args.artifact_root is None:
        args.artifact_root = args.runtime_root / "artifacts" / "reproductions"
    raise SystemExit(Reproducer(args).execute())


if __name__ == "__main__":
    main()
