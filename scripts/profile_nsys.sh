#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

config="configs/r13a_h20_release.json"
variant=""
target_length=""
run_id=""
replicate="1"
correctness=""
output_root="${ITK_ARTIFACT_ROOT:-/data/${USER:?USER is not set}/artifacts}/profiles"

usage() {
    echo "usage: scripts/profile_nsys.sh --variant ID --target-length N --run-id ID --correctness PATH [--config PATH] [--replicate N] [--output-root PATH]" >&2
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?missing value for --config}"; shift 2 ;;
        --variant) variant="${2:?missing value for --variant}"; shift 2 ;;
        --target-length) target_length="${2:?missing value for --target-length}"; shift 2 ;;
        --run-id) run_id="${2:?missing value for --run-id}"; shift 2 ;;
        --replicate) replicate="${2:?missing value for --replicate}"; shift 2 ;;
        --correctness) correctness="${2:?missing value for --correctness}"; shift 2 ;;
        --output-root) output_root="${2:?missing value for --output-root}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${variant}" || -z "${target_length}" || -z "${run_id}" || -z "${correctness}" ]]; then
    usage
    exit 2
fi
if [[ ! "${target_length}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--target-length must be a positive integer" >&2
    exit 2
fi
if [[ ! "${replicate}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--replicate must be a positive integer" >&2
    exit 2
fi
if [[ ! "${run_id}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$ ]]; then
    echo "--run-id must be one safe path component" >&2
    exit 2
fi
if [[ ! -f "${config}" ]]; then
    echo "missing config: ${config}" >&2
    exit 1
fi
if [[ ! -f "${correctness}" ]]; then
    echo "missing correctness artifact: ${correctness}" >&2
    exit 1
fi

label_json="$(scripts/run_h20.sh python scripts/nvtx_label.py \
    --config "${config}" \
    --variant "${variant}" \
    --target-length "${target_length}" \
    --stage pipeline \
    --format json)"
resolved_variant="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["variant"])' "${label_json}")"
variant_cache_key="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cache_key"])' "${label_json}")"
nvtx_label="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["label"])' "${label_json}")"
if [[ ! "${resolved_variant}" =~ ^[a-z][a-z0-9_.-]{0,63}$ ]]; then
    echo "resolved variant is not a safe path component: ${resolved_variant}" >&2
    exit 2
fi

profile_dir="${output_root}/${run_id}/${resolved_variant}"
tag="nsys-prefill-target-L${target_length}-pipeline-r${replicate}"
output_base="${profile_dir}/${tag}"
report="${output_base}.nsys-rep"
sqlite="${output_base}.sqlite"
metadata="${output_base}.metadata.json"
stats_csv="${output_base}.nvtx-kern-sum.csv"
verification_json="${output_base}.verification.json"
for artifact in "${report}" "${sqlite}" "${metadata}" "${stats_csv}" "${verification_json}"; do
    if [[ -e "${artifact}" ]]; then
        echo "refusing to overwrite existing profile artifact: ${artifact}" >&2
        echo "choose a new --run-id or --replicate" >&2
        exit 1
    fi
done
mkdir -p "${profile_dir}"

target_command=(
    python -m index_topk_perflab.profile capture
    --config "${config}"
    --variant "${variant}"
    --target-length "${target_length}"
    --mode nsys
    --stage pipeline
    --run-id "${run_id}"
    --correctness "${correctness}"
    --metadata-output "${metadata}"
)
profiler_command=(
    nsys profile
    --trace=cuda,nvtx
    --sample=none
    --cpuctxsw=none
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --wait=all
    --force-overwrite=true
    --output="${output_base}"
)
profiler_command_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \
    "${profiler_command[@]}" "${target_command[@]}")"

ITK_VARIANT_CACHE_KEY="${variant_cache_key}" \
scripts/run_h20.sh \
    "${profiler_command[@]}" \
    env \
    ITK_EXPECTED_NVTX_LABEL="${nvtx_label}" \
    ITK_PROFILER_COMMAND_JSON="${profiler_command_json}" \
    "${target_command[@]}"

if [[ ! -s "${report}" || ! -s "${metadata}" ]]; then
    echo "Nsight Systems did not produce a report and metadata pair: ${output_base}" >&2
    exit 1
fi

ITK_VARIANT_CACHE_KEY="${variant_cache_key}" \
scripts/run_h20.sh nsys export \
    --type sqlite \
    --output "${sqlite}" \
    "${report}"
if [[ ! -s "${sqlite}" ]]; then
    echo "Nsight Systems SQLite export is missing or empty: ${sqlite}" >&2
    exit 1
fi
stats_tmp="${stats_csv}.tmp-$$"
cleanup() {
    if [[ -e "${stats_tmp}" ]]; then
        rm -f -- "${stats_tmp}"
    fi
}
trap cleanup EXIT
if ! ITK_VARIANT_CACHE_KEY="${variant_cache_key}" \
    scripts/run_h20.sh nsys stats --format csv --report nvtx_kern_sum "${sqlite}" \
    > "${stats_tmp}"; then
    echo "failed to read NVTX/kernel summary from ${sqlite}" >&2
    exit 1
fi
mv -- "${stats_tmp}" "${stats_csv}"
trap - EXIT
PYTHONPATH=src python3 -m index_topk_perflab.profile_validation nsys \
    --input "${stats_csv}" \
    --expected-nvtx-label "${nvtx_label}" \
    --stage pipeline \
    --output "${verification_json}"

ITK_VARIANT_CACHE_KEY="${variant_cache_key}" \
scripts/run_h20.sh python -m index_topk_perflab.profile finalize \
    --metadata "${metadata}" \
    --native-report "${report}" \
    --export "${sqlite}" \
    --verification-json "${verification_json}"

echo "nsys profile verified report=${report} metadata=${metadata}"
