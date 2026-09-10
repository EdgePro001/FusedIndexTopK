#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

config="configs/fused_index_topk_h20.json"
candidate=""
run_id=""
baseline_correctness=""
candidate_correctness=""
output_root="${ITK_ARTIFACT_ROOT:-/data/${USER:?USER is not set}/artifacts}/campaigns"

usage() {
    echo "usage: scripts/run_campaign.sh --candidate ID --run-id ID --baseline-correctness PATH --candidate-correctness PATH [--config PATH] [--output-root PATH]" >&2
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?missing value for --config}"; shift 2 ;;
        --candidate) candidate="${2:?missing value for --candidate}"; shift 2 ;;
        --run-id) run_id="${2:?missing value for --run-id}"; shift 2 ;;
        --baseline-correctness) baseline_correctness="${2:?missing value}"; shift 2 ;;
        --candidate-correctness) candidate_correctness="${2:?missing value}"; shift 2 ;;
        --output-root) output_root="${2:?missing value for --output-root}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${candidate}" || -z "${run_id}" || -z "${baseline_correctness}" || -z "${candidate_correctness}" ]]; then
    usage
    exit 2
fi
if [[ ! "${run_id}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$ ]]; then
    echo "--run-id must be one safe path component" >&2
    exit 2
fi
for path in "${config}" "${baseline_correctness}" "${candidate_correctness}"; do
    if [[ ! -f "${path}" ]]; then
        echo "missing required file: ${path}" >&2
        exit 1
    fi
done

campaign_root="${output_root}/${run_id}"
plan="${campaign_root}/plan.json"
sealed="${campaign_root}/campaign.json"
PYTHONPATH=src python3 -m index_topk_perflab.cli campaign-plan \
    --config "${config}" \
    --candidate "${candidate}" \
    --run-id "${run_id}" \
    --output-root "${output_root}" \
    --output "${plan}"

while IFS=$'\t' read -r run_index run_count arm variant slot_run_id artifact_path; do
    if [[ "${arm}" == "A" ]]; then
        correctness="${baseline_correctness}"
    else
        correctness="${candidate_correctness}"
    fi
    echo "campaign run ${run_index}/${run_count}: arm=${arm} variant=${variant}"
    python3 scripts/live_progress.py \
        --kind benchmark \
        --label "campaign ${run_index}/${run_count} ${arm}" \
        --config "${config}" \
        --artifact "${artifact_path}" \
        -- \
        scripts/run_h20.sh python -m index_topk_perflab.cli bench \
            --config "${config}" \
            --variant "${variant}" \
            --run-id "${slot_run_id}" \
            --correctness "${correctness}" \
            --output "${artifact_path}"
done < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); [print(i, len(p["runs"]), r["arm"], r["variant"], r["run_id"], r["artifact_path"], sep="\t") for i,r in enumerate(p["runs"], 1)]' "${plan}")

PYTHONPATH=src python3 -m index_topk_perflab.cli campaign-seal \
    --plan "${plan}" \
    --output "${sealed}"

echo "campaign sealed artifact=${sealed}"
