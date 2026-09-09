#!/usr/bin/env python3
"""Run one lab command while reporting artifact-backed progress."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _load_cases(config_path: Path) -> list[tuple[int, int]]:
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload["workload"]["benchmark_cases"]
    cases = [(int(row["query_tokens"]), int(row["context_tokens"])) for row in rows]
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("benchmark matrix must contain unique Q/N cases")
    return cases


def _case_path(artifact_path: Path, query_tokens: int, context_tokens: int) -> Path:
    directory = artifact_path.with_name(f"{artifact_path.stem}.cases")
    return directory / f"Q{query_tokens}-N{context_tokens}.json"


def _case_metrics(path: Path) -> dict[str, float | int]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "complete":
        raise ValueError(f"case artifact is not complete: {path}")
    selected: dict[str, dict[str, Any]] = {}
    for row in payload.get("summary", []):
        pass_id = row.get("pass")
        if (
            pass_id in {"formal_kernel_sum", "cuda_event_total"}
            and row.get("scope_id") == "operator_total"
            and row.get("stage_id") == "operator_total"
        ):
            selected[str(pass_id)] = row
    if set(selected) != {"formal_kernel_sum", "cuda_event_total"}:
        raise ValueError(f"case artifact has no complete operator summary: {path}")
    formal = selected["formal_kernel_sum"]
    event = selected["cuda_event_total"]
    return {
        "formal_count": int(formal["count"]),
        "formal_median_ms": float(formal["median_ms"]),
        "formal_p95_ms": float(formal["p95_ms"]),
        "event_count": int(event["count"]),
        "event_median_ms": float(event["median_ms"]),
        "event_p95_ms": float(event["p95_ms"]),
    }


class _Terminal:
    def __init__(self) -> None:
        self.interactive = sys.stdout.isatty()
        self._active_width = 0
        self._last_noninteractive = 0.0

    def status(self, message: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if self.interactive:
            padding = " " * max(0, self._active_width - len(message))
            print(f"\r{message}{padding}", end="", flush=True)
            self._active_width = len(message)
        elif force or now - self._last_noninteractive >= 30.0:
            print(message, flush=True)
            self._last_noninteractive = now

    def line(self, message: str) -> None:
        if self.interactive and self._active_width:
            print("\r" + " " * self._active_width + "\r", end="")
        self._active_width = 0
        print(message, flush=True)


def _tail(path: Path, line_count: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def _artifact_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return "unreadable"
    return str(payload.get("status", "missing-status"))


def _command_from_args(values: list[str]) -> list[str]:
    command = values[1:] if values[:1] == ["--"] else values
    if not command:
        raise ValueError("a command is required after --")
    return command


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)


def _run(args: argparse.Namespace) -> int:
    command = _command_from_args(args.command)
    artifact = args.artifact.resolve()
    config = args.config.resolve() if args.config is not None else None
    if args.kind == "benchmark" and config is None:
        raise ValueError("--config is required for benchmark progress")
    if artifact.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {artifact}")

    cases = _load_cases(config) if config is not None and args.kind == "benchmark" else []
    log_path = (
        args.log.resolve()
        if args.log is not None
        else artifact.with_name(f"{artifact.stem}.run.log")
    )
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite run log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    terminal = _Terminal()
    terminal.line(f"[{args.label}] started | artifact={artifact} | log={log_path}")
    started = time.monotonic()
    seen: set[tuple[int, int]] = set()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if cases:
                    for index, (query_tokens, context_tokens) in enumerate(cases, start=1):
                        key = (query_tokens, context_tokens)
                        path = _case_path(artifact, *key)
                        if key in seen or not path.is_file():
                            continue
                        metrics = _case_metrics(path)
                        seen.add(key)
                        terminal.line(
                            f"[{args.label}] case {index}/{len(cases)} complete | "
                            f"Q={query_tokens} N={context_tokens} | "
                            f"CUPTI median={metrics['formal_median_ms']:.6f} ms "
                            f"P95={metrics['formal_p95_ms']:.6f} ms "
                            f"n={metrics['formal_count']} | "
                            f"Event median={metrics['event_median_ms']:.6f} ms "
                            f"P95={metrics['event_p95_ms']:.6f} ms "
                            f"n={metrics['event_count']}"
                        )
                    remaining = [case for case in cases if case not in seen]
                    current = (
                        f"Q={remaining[0][0]} N={remaining[0][1]}"
                        if remaining
                        else "sealing aggregate artifact"
                    )
                    percent = 100.0 * len(seen) / len(cases)
                    message = (
                        f"[{args.label}] {len(seen)}/{len(cases)} "
                        f"({percent:5.1f}%) | current={current} | "
                        f"elapsed={time.monotonic() - started:.0f}s"
                    )
                else:
                    message = f"[{args.label}] running | elapsed={time.monotonic() - started:.0f}s"
                terminal.status(message)
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            terminal.line(f"[{args.label}] interrupted; stopping child process")
            _terminate(process)
            return 130

        if cases:
            for index, (query_tokens, context_tokens) in enumerate(cases, start=1):
                key = (query_tokens, context_tokens)
                path = _case_path(artifact, *key)
                if key in seen or not path.is_file():
                    continue
                metrics = _case_metrics(path)
                seen.add(key)
                terminal.line(
                    f"[{args.label}] case {index}/{len(cases)} complete | "
                    f"Q={query_tokens} N={context_tokens} | "
                    f"CUPTI median={metrics['formal_median_ms']:.6f} ms "
                    f"P95={metrics['formal_p95_ms']:.6f} ms | "
                    f"Event median={metrics['event_median_ms']:.6f} ms "
                    f"P95={metrics['event_p95_ms']:.6f} ms"
                )
        return_code = int(process.returncode or 0)

    status = _artifact_status(artifact)
    elapsed = time.monotonic() - started
    if return_code == 0 and status in {"passed", "complete"}:
        terminal.line(f"[{args.label}] {status} | elapsed={elapsed:.1f}s | artifact={artifact}")
        return 0

    terminal.line(
        f"[{args.label}] failed | exit={return_code} artifact_status={status} | log={log_path}"
    )
    excerpt = _tail(log_path)
    if excerpt:
        terminal.line("--- last log lines ---")
        terminal.line(excerpt)
    return return_code or 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("stage", "benchmark"), default="benchmark")
    parser.add_argument("--label", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if not 0.2 <= args.poll_seconds <= 60.0:
        raise SystemExit("--poll-seconds must be in [0.2, 60]")
    try:
        return_code = _run(args)
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(f"progress runner error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
