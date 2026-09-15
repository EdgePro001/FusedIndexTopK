# One-command smoke reproduction

The P0 path validates that the frozen H20 environment can be created, the
control plane is healthy, and both the FlashInfer baseline and FusedIndexTopK
candidate are exact on one deterministic short-context case.

On an H20 host, run:

```bash
ITK_RUNTIME_ROOT=/data/$USER ./scripts/reproduce.sh --level smoke
```

The command performs a hardware/toolchain preflight, runs
`scripts/setup_h20.sh`, executes Ruff and the unit tests, discovers configured
variants, and runs correctness at `Q=4096, N=6144`. It stops at the first failed
gate and never starts correctness after a failed setup or test stage.

Artifacts are written beneath:

```text
$ITK_RUNTIME_ROOT/artifacts/reproductions/<run-id>/
  preflight.json
  smoke-config.json
  correctness/<variant>/correctness.json
  logs/
  report.json
  REPORT.md
```

To reuse an already-created runtime (for example, on an offline machine), add
`--skip-setup`. Every run ID is immutable; choose a new `--run-id` to rerun.

This is a code/smoke reproduction gate, not a reproduction of the published
held-out replay performance numbers. Those numbers require the private replay
manifest and tensors described in [METHODOLOGY.md](METHODOLOGY.md).

On a CPU-only development machine the command is expected to fail at
preflight, but it still writes a diagnostic `preflight.json` and `REPORT.md`.
Use `make lint` and `make test` for the CPU-only control-plane checks.
