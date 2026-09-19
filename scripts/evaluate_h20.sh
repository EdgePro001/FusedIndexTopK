#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

config="configs/fused_index_topk_h20.json"
candidate=""
baseline=""
run_id="h20-eval-$(date -u +%Y%m%dT%H%M%SZ)"
mode="screening"
runtime_root="${ITK_RUNTIME_ROOT:-${XDG_CACHE_HOME:-${HOME:?HOME is not set}/.cache}/fused-index-topk}"
artifact_root="${ITK_ARTIFACT_ROOT:-${runtime_root}/artifacts}"

usage() {
    cat >&2 <<'EOF'
usage: scripts/evaluate_h20.sh --candidate ID [options]

Required:
  --candidate ID        candidate variant to evaluate

Options:
  --mode MODE           screening (default) or formal
  --run-id ID           safe, unique artifact prefix
  --config PATH         default: configs/fused_index_topk_h20.json
  --baseline ID         default: baseline_variant from config
  --artifact-root PATH  default: the per-user FusedIndexTopK cache

screening runs two independent correctness gates, one baseline matrix, one
candidate matrix, and a diagnostic comparison. formal runs the correctness
gates followed by the configured ABBA/BAAB campaign.
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?missing value for --config}"; shift 2 ;;
        --candidate) candidate="${2:?missing value for --candidate}"; shift 2 ;;
        --baseline) baseline="${2:?missing value for --baseline}"; shift 2 ;;
        --run-id) run_id="${2:?missing value for --run-id}"; shift 2 ;;
        --mode) mode="${2:?missing value for --mode}"; shift 2 ;;
        --artifact-root) artifact_root="${2:?missing value for --artifact-root}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${candidate}" ]]; then
    usage
    exit 2
fi
if [[ ! "${run_id}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$ ]]; then
    echo "--run-id must be one safe path component" >&2
    exit 2
fi
if [[ "${mode}" != "screening" && "${mode}" != "formal" ]]; then
    echo "--mode must be screening or formal" >&2
    exit 2
fi
if [[ ! -f "${config}" ]]; then
    echo "missing config: ${config}" >&2
    exit 1
fi
if [[ -z "${baseline}" ]]; then
    baseline="$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["baseline_variant"])' \
        "${config}")"
fi
if [[ "${candidate}" == "${baseline}" ]]; then
    echo "candidate and baseline must differ" >&2
    exit 2
fi

export ITK_ARTIFACT_ROOT="${artifact_root}"
baseline_check_run="${run_id}-check-a"
candidate_check_run="${run_id}-check-b"
baseline_correctness="${artifact_root}/raw/${baseline_check_run}/${baseline}/correctness.json"
candidate_correctness="${artifact_root}/raw/${candidate_check_run}/${candidate}/correctness.json"

echo "H20 evaluation"
echo "  mode=${mode}"
echo "  run_id=${run_id}"
echo "  config=${config}"
echo "  baseline=${baseline}"
echo "  candidate=${candidate}"
echo "  artifact_root=${artifact_root}"

python3 scripts/live_progress.py \
    --kind stage \
    --label "correctness baseline" \
    --artifact "${baseline_correctness}" \
    -- \
    scripts/run_h20.sh python -m fused_index_topk.cli check \
        --config "${config}" \
        --variant "${baseline}" \
        --run-id "${baseline_check_run}" \
        --output "${baseline_correctness}"

python3 scripts/live_progress.py \
    --kind stage \
    --label "correctness candidate" \
    --artifact "${candidate_correctness}" \
    -- \
    scripts/run_h20.sh python -m fused_index_topk.cli check \
        --config "${config}" \
        --variant "${candidate}" \
        --run-id "${candidate_check_run}" \
        --output "${candidate_correctness}"

if [[ "${mode}" == "formal" ]]; then
    scripts/run_campaign.sh \
        --config "${config}" \
        --candidate "${candidate}" \
        --run-id "${run_id}" \
        --baseline-correctness "${baseline_correctness}" \
        --candidate-correctness "${candidate_correctness}" \
        --output-root "${artifact_root}/campaigns"
    echo "formal evaluation complete: ${artifact_root}/campaigns/${run_id}/campaign.json"
    exit 0
fi

baseline_bench_run="${run_id}-screen-a"
candidate_bench_run="${run_id}-screen-b"
baseline_benchmark="${artifact_root}/raw/${baseline_bench_run}/${baseline}/benchmark.json"
candidate_benchmark="${artifact_root}/raw/${candidate_bench_run}/${candidate}/benchmark.json"
comparison="${artifact_root}/comparisons/${run_id}.json"

python3 scripts/live_progress.py \
    --kind benchmark \
    --label "benchmark baseline" \
    --config "${config}" \
    --artifact "${baseline_benchmark}" \
    -- \
    scripts/run_h20.sh python -m fused_index_topk.cli bench \
        --config "${config}" \
        --variant "${baseline}" \
        --run-id "${baseline_bench_run}" \
        --correctness "${baseline_correctness}" \
        --output "${baseline_benchmark}"

python3 scripts/live_progress.py \
    --kind benchmark \
    --label "benchmark candidate" \
    --config "${config}" \
    --artifact "${candidate_benchmark}" \
    -- \
    scripts/run_h20.sh python -m fused_index_topk.cli bench \
        --config "${config}" \
        --variant "${candidate}" \
        --run-id "${candidate_bench_run}" \
        --correctness "${candidate_correctness}" \
        --output "${candidate_benchmark}"

scripts/run_h20.sh python -m fused_index_topk.cli compare \
    --baseline "${baseline_benchmark}" \
    --candidate "${candidate_benchmark}" \
    --output "${comparison}"

echo "screening complete"
echo "  baseline=${baseline_benchmark}"
echo "  candidate=${candidate_benchmark}"
echo "  comparison=${comparison}"
echo "  note=single-run screening is diagnostic; use --mode formal for a decision"
