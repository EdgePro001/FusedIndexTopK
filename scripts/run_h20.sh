#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${ITK_RUNTIME_ROOT:-${XDG_CACHE_HOME:-${HOME:?HOME is not set}/.cache}/fused-index-topk}"
venv="${runtime_root}/runtime/venv"
torch_lib="${venv}/lib/python3.12/site-packages/torch/lib"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.0}"
deepgemm_source="${runtime_root}/src/DeepGEMM-exact"
deepselect_source="${runtime_root}/src/DeepSelect-exact"
python_dev_include="${runtime_root}/runtime/python3.12-dev/root/usr/include"

if [[ "$#" -eq 0 ]]; then
    echo "usage: scripts/run_h20.sh <command> [args...]" >&2
    exit 2
fi
if [[ ! -x "${venv}/bin/python" ]]; then
    echo "missing H20 virtual environment: ${venv}; run scripts/setup_h20.sh" >&2
    exit 1
fi
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "missing CUDA toolkit: ${cuda_home}" >&2
    exit 1
fi
if [[ ! -d "${deepgemm_source}/.git" ]]; then
    echo "missing exact DeepGEMM checkout: ${deepgemm_source}" >&2
    exit 1
fi
if [[ ! -d "${deepselect_source}/.git" ]]; then
    echo "missing exact DeepSelect checkout: ${deepselect_source}" >&2
    exit 1
fi
if [[ ! -f "${python_dev_include}/python3.12/Python.h" ]]; then
    echo "missing private Python development headers; run scripts/setup_h20.sh" >&2
    exit 1
fi

cache_key="${ITK_VARIANT_CACHE_KEY:-}"
if [[ -z "${cache_key}" ]]; then
    requested_variant=""
    requested_config="configs/fused_index_topk_h20.json"
    requested_length=""
    arguments=("$@")
    for ((index = 0; index < ${#arguments[@]}; index++)); do
        case "${arguments[index]}" in
            --variant) requested_variant="${arguments[index + 1]:-}" ;;
            --config) requested_config="${arguments[index + 1]:-}" ;;
            --target-length) requested_length="${arguments[index + 1]:-}" ;;
        esac
    done
    if [[ -n "${requested_variant}" ]]; then
        if [[ -z "${requested_length}" ]]; then
            requested_length="$("${venv}/bin/python" -c \
                'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8"))["profiling"]; print(p["nsys_cases"][0]["context_tokens"])' \
                "${requested_config}")"
        fi
        variant_info="$(
            cd "${project_root}"
            env \
                PYTHONPATH="${project_root}/src" \
                DEEPSELECT_SOURCE="${deepselect_source}" \
                "${venv}/bin/python" scripts/nvtx_label.py \
                    --config "${requested_config}" \
                    --variant "${requested_variant}" \
                    --target-length "${requested_length}" \
                    --format json
        )"
        cache_key="$("${venv}/bin/python" -c \
            'import json,sys; print(json.loads(sys.argv[1])["cache_key"])' \
            "${variant_info}")"
    else
        cache_key="shared"
    fi
fi
if [[ ! "${cache_key}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$ ]]; then
    echo "ITK_VARIANT_CACHE_KEY must be one safe path component" >&2
    exit 2
fi

cache_root="${runtime_root}/runtime/cache/${cache_key}"
mkdir -p \
    "${cache_root}/deep-gemm" \
    "${cache_root}/torch-extensions" \
    "${cache_root}/triton" \
    "${runtime_root}/artifacts"

cd "${project_root}"
exec env \
    PATH="${venv}/bin:${cuda_home}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONPATH="${project_root}/src" \
    LD_LIBRARY_PATH="${torch_lib}:${cuda_home}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    CUDA_HOME="${cuda_home}" \
    CUDACXX="${cuda_home}/bin/nvcc" \
    CPATH="${python_dev_include}/python3.12:${python_dev_include}" \
    TORCH_CUDA_ARCH_LIST="9.0a" \
    DEEPGEMM_SOURCE="${deepgemm_source}" \
    DEEPSELECT_SOURCE="${deepselect_source}" \
    DG_JIT_NVCC_COMPILER="${cuda_home}/bin/nvcc" \
    DG_JIT_USE_NVRTC=0 \
    DG_JIT_CACHE_DIR="${cache_root}/deep-gemm" \
    TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions" \
    TRITON_CACHE_DIR="${cache_root}/triton" \
    ITK_VARIANT_FINGERPRINT="${cache_key}" \
    ITK_ARTIFACT_ROOT="${runtime_root}/artifacts" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    "$@"
