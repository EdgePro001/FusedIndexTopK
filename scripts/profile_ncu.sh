#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

config="configs/fused_index_topk_h20.json"
variant=""
target_length=""
stage=""
run_id=""
replicate="1"
correctness=""
runtime_root="${ITK_RUNTIME_ROOT:-${XDG_CACHE_HOME:-${HOME:?HOME is not set}/.cache}/fused-index-topk}"
output_root="${ITK_ARTIFACT_ROOT:-${runtime_root}/artifacts}/profiles"
replay_manifest=""
replay_split=""
replay_seed="20260825"

usage() {
    echo "usage: scripts/profile_ncu.sh --variant ID --target-length N --stage ID --run-id ID --correctness PATH [--config PATH] [--replicate N] [--output-root PATH] [--replay-manifest PATH --replay-split SPLIT [--replay-seed N]]" >&2
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) config="${2:?missing value for --config}"; shift 2 ;;
        --variant) variant="${2:?missing value for --variant}"; shift 2 ;;
        --target-length) target_length="${2:?missing value for --target-length}"; shift 2 ;;
        --stage) stage="${2:?missing value for --stage}"; shift 2 ;;
        --run-id) run_id="${2:?missing value for --run-id}"; shift 2 ;;
        --replicate) replicate="${2:?missing value for --replicate}"; shift 2 ;;
        --correctness) correctness="${2:?missing value for --correctness}"; shift 2 ;;
        --output-root) output_root="${2:?missing value for --output-root}"; shift 2 ;;
        --replay-manifest) replay_manifest="${2:?missing value for --replay-manifest}"; shift 2 ;;
        --replay-split) replay_split="${2:?missing value for --replay-split}"; shift 2 ;;
        --replay-seed) replay_seed="${2:?missing value for --replay-seed}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${variant}" || -z "${target_length}" || -z "${stage}" || -z "${run_id}" || -z "${correctness}" ]]; then
    usage
    exit 2
fi
if [[ "${stage}" == "pipeline" ]]; then
    echo "NCU requires one physical --stage, not pipeline" >&2
    exit 2
fi
if [[ ! "${stage}" =~ ^[a-z][a-z0-9_.-]{0,63}$ ]]; then
    echo "--stage must be a safe physical stage ID" >&2
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
if [[ ! "${replay_seed}" =~ ^[0-9]+$ ]]; then
    echo "--replay-seed must be a non-negative integer" >&2
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
if [[ -n "${replay_manifest}" || -n "${replay_split}" ]]; then
    if [[ -z "${replay_manifest}" || -z "${replay_split}" ]]; then
        echo "--replay-manifest and --replay-split must be provided together" >&2
        exit 2
    fi
    if [[ ! -f "${replay_manifest}" ]]; then
        echo "missing replay manifest: ${replay_manifest}" >&2
        exit 1
    fi
    case "${replay_split}" in
        tuning|test_normal|test_hard) ;;
        *) echo "invalid --replay-split: ${replay_split}" >&2; exit 2 ;;
    esac
fi

label_json="$(scripts/run_h20.sh python scripts/nvtx_label.py \
    --config "${config}" \
    --variant "${variant}" \
    --target-length "${target_length}" \
    --stage "${stage}" \
    --format json)"
resolved_variant="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["variant"])' "${label_json}")"
variant_cache_key="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["cache_key"])' "${label_json}")"
nvtx_label="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["label"])' "${label_json}")"
ncu_filter="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["ncu_filter"])' "${label_json}")"
if [[ ! "${resolved_variant}" =~ ^[a-z][a-z0-9_.-]{0,63}$ ]]; then
    echo "resolved variant is not a safe path component: ${resolved_variant}" >&2
    exit 2
fi
if [[ "${ncu_filter}" != "${nvtx_label}/" ]]; then
    echo "invalid NCU push/pop filter generated for ${nvtx_label}" >&2
    exit 2
fi

metrics="$(python3 -c 'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print(",".join(data["profiling"]["ncu_metrics"]))' "${config}")"
if [[ -z "${metrics}" ]]; then
    echo "config contains no explicit NCU metrics" >&2
    exit 2
fi
section_args=()
while IFS= read -r section; do
    if [[ -n "${section}" ]]; then
        section_args+=(--section "${section}")
    fi
done < <(python3 -c 'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); print("\n".join(data["profiling"]["ncu_sections"]))' "${config}")
if [[ "${#section_args[@]}" -eq 0 ]]; then
    echo "config contains no NCU sections" >&2
    exit 2
fi

profile_dir="${output_root}/${run_id}/${resolved_variant}"
tag="ncu-prefill-target-L${target_length}-${stage}-r${replicate}"
output_base="${profile_dir}/${tag}"
report="${output_base}.ncu-rep"
raw_csv="${output_base}.raw.csv"
metadata="${output_base}.metadata.json"
verification_json="${output_base}.verification.json"
for artifact in "${report}" "${raw_csv}" "${metadata}" "${verification_json}"; do
    if [[ -e "${artifact}" ]]; then
        echo "refusing to overwrite existing profile artifact: ${artifact}" >&2
        echo "choose a new --run-id or --replicate" >&2
        exit 1
    fi
done
mkdir -p "${profile_dir}"

target_command=(
    python -m fused_index_topk.profile capture
    --config "${config}"
    --variant "${variant}"
    --target-length "${target_length}"
    --mode ncu
    --stage "${stage}"
    --run-id "${run_id}"
    --correctness "${correctness}"
    --metadata-output "${metadata}"
)
if [[ -n "${replay_manifest}" ]]; then
    target_command+=(
        --replay-manifest "${replay_manifest}"
        --replay-split "${replay_split}"
        --replay-seed "${replay_seed}"
    )
fi
ncu_args=(
    --target-processes application-only
    --profile-from-start off
    --replay-mode application
    --app-replay-mode strict
    --cache-control none
    --clock-control base
    --pipeline-boost-state stable
    --nvtx
    --nvtx-include "${ncu_filter}"
    --metrics "${metrics}"
    "${section_args[@]}"
    --export "${output_base}"
)
profiler_command=(ncu "${ncu_args[@]}")
profiler_command_json="$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' \
    "${profiler_command[@]}" "${target_command[@]}")"

ITK_VARIANT_CACHE_KEY="${variant_cache_key}" \
scripts/run_h20.sh env \
    ITK_EXPECTED_NVTX_LABEL="${nvtx_label}" \
    ITK_NCU_METRICS="${metrics}" \
    ITK_PROFILER_COMMAND_JSON="${profiler_command_json}" \
    ncu \
    "${ncu_args[@]}" \
    "${target_command[@]}"

if [[ ! -s "${report}" || ! -s "${metadata}" ]]; then
    echo "Nsight Compute did not produce a report and metadata pair: ${output_base}" >&2
    exit 1
fi

raw_tmp="${raw_csv}.tmp-$$"
cleanup() {
    if [[ -e "${raw_tmp}" ]]; then
        rm -f -- "${raw_tmp}"
    fi
}
trap cleanup EXIT
if ! ITK_VARIANT_CACHE_KEY="${variant_cache_key}" \
    scripts/run_h20.sh ncu --import "${report}" --page raw --csv > "${raw_tmp}"; then
    echo "failed to export raw NCU CSV from ${report}" >&2
    exit 1
fi
mv -- "${raw_tmp}" "${raw_csv}"
trap - EXIT
PYTHONPATH=src python3 -m fused_index_topk.profile_validation ncu \
    --input "${raw_csv}" \
    --metrics "${metrics}" \
    --expected-nvtx-label "${nvtx_label}" \
    --stage "${stage}" \
    --output "${verification_json}"

ITK_VARIANT_CACHE_KEY="${variant_cache_key}" \
scripts/run_h20.sh python -m fused_index_topk.profile finalize \
    --metadata "${metadata}" \
    --native-report "${report}" \
    --export "${raw_csv}" \
    --verification-json "${verification_json}"

echo "ncu profile verified report=${report} metadata=${metadata}"
