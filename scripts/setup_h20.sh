#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="${ITK_RUNTIME_ROOT:-/data/${USER:?USER is not set}}"
venv="${runtime_root}/runtime/venv"
torch_lib="${venv}/lib/python3.12/site-packages/torch/lib"
cuda_home="${CUDA_HOME:-/usr/local/cuda-13.0}"
deepgemm_source="${runtime_root}/src/DeepGEMM-exact"
flashinfer_source="${runtime_root}/src/flashinfer-v0.6.17"
config="${project_root}/configs/r13a_h20_release.json"
flashinfer_lock="${project_root}/src/index_topk_perflab/variants/deepgemm_flashinfer/SOURCE_LOCK.json"

if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
    echo "missing CUDA toolkit: ${cuda_home}" >&2
    exit 1
fi
mkdir -p "${runtime_root}/src" "${runtime_root}/runtime" "${runtime_root}/artifacts"

expected_deepgemm_commit="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["sources"]["deep_gemm_commit"])' \
    "${config}")"
if [[ ! -d "${deepgemm_source}/.git" ]]; then
    git clone https://github.com/deepseek-ai/DeepGEMM.git "${deepgemm_source}"
    git -C "${deepgemm_source}" checkout --detach "${expected_deepgemm_commit}"
fi
actual_deepgemm_commit="$(git -C "${deepgemm_source}" rev-parse HEAD)"
if [[ "${actual_deepgemm_commit}" != "${expected_deepgemm_commit}" ]]; then
    echo "DeepGEMM checkout mismatch: expected ${expected_deepgemm_commit}, got ${actual_deepgemm_commit}" >&2
    exit 1
fi
if [[ -n "$(git -C "${deepgemm_source}" status --short)" ]]; then
    echo "DeepGEMM checkout must be clean" >&2
    exit 1
fi
git -C "${deepgemm_source}" submodule update --init --recursive

flashinfer_repository="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["repository"])' \
    "${flashinfer_lock}")"
flashinfer_tag="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["tag"])' \
    "${flashinfer_lock}")"
expected_flashinfer_commit="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["commit"])' \
    "${flashinfer_lock}")"
if [[ ! -d "${flashinfer_source}/.git" ]]; then
    git clone --branch "${flashinfer_tag}" --depth 1 \
        "${flashinfer_repository}" "${flashinfer_source}"
fi
actual_flashinfer_commit="$(git -C "${flashinfer_source}" rev-parse HEAD)"
if [[ "${actual_flashinfer_commit}" != "${expected_flashinfer_commit}" ]]; then
    echo "FlashInfer checkout mismatch: expected ${expected_flashinfer_commit}, got ${actual_flashinfer_commit}" >&2
    exit 1
fi
if [[ -n "$(git -C "${flashinfer_source}" status --short)" ]]; then
    echo "FlashInfer checkout must be clean" >&2
    exit 1
fi

if [[ ! -x "${venv}/bin/python" ]]; then
    # Some minimal Ubuntu images omit the distro ensurepip package.  The
    # interpreter and isolation still work, so create the venv without pip and
    # bootstrap pip below without requiring sudo.
    python3 -m venv --without-pip "${venv}"
fi
if ! "${venv}/bin/python" -m pip --version >/dev/null 2>&1; then
    get_pip="${runtime_root}/runtime/get-pip.py"
    curl --fail --location --silent --show-error \
        https://bootstrap.pypa.io/get-pip.py --output "${get_pip}"
    "${venv}/bin/python" "${get_pip}"
fi
"${venv}/bin/python" -m pip install --upgrade \
    pip setuptools wheel ninja packaging pytest ruff "numpy==2.3.3"
"${venv}/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.10.0+cu130"

# The host image has the Python runtime but no system-wide development headers,
# and this account intentionally has no sudo.  Extract the matching Ubuntu
# header package into the private runtime tree; no system package is installed.
python_dev_root="${runtime_root}/runtime/python3.12-dev"
python_include="${python_dev_root}/root/usr/include/python3.12"
if [[ ! -f "${python_include}/Python.h" ]]; then
    mkdir -p "${python_dev_root}/packages" "${python_dev_root}/root"
    (
        cd "${python_dev_root}/packages"
        apt-get download libpython3.12-dev
        for package in libpython3.12-dev_*.deb; do
            dpkg-deb --extract "${package}" "${python_dev_root}/root"
        done
    )
fi

common_env=(
    env
    "PATH=${venv}/bin:${cuda_home}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    "CUDA_HOME=${cuda_home}"
    "CUDACXX=${cuda_home}/bin/nvcc"
    "TORCH_CUDA_ARCH_LIST=9.0a"
    "CPATH=${python_include}:${python_dev_root}/root/usr/include"
    "LD_LIBRARY_PATH=${torch_lib}:${cuda_home}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
)
# PyTorch is part of the extension ABI.  Remove only setuptools-generated
# objects so a runtime-version change cannot silently reuse a stale binary.
(
    cd "${deepgemm_source}"
    "${common_env[@]}" "${venv}/bin/python" setup.py clean --all
)
"${common_env[@]}" "${venv}/bin/python" -m pip install \
    --no-build-isolation --no-cache-dir --no-deps --force-reinstall "${deepgemm_source}"
"${venv}/bin/python" -m pip install --no-deps -e "${project_root}"

"${common_env[@]}" \
    DEEPGEMM_SOURCE="${deepgemm_source}" \
    FLASHINFER_SOURCE="${flashinfer_source}" \
    "${venv}/bin/python" -c \
    'import importlib.metadata as m, deep_gemm, deep_gemm_cpp, torch; from pathlib import Path; assert Path(__import__("os").environ["FLASHINFER_SOURCE"], "include/flashinfer/topk.cuh").is_file(); print("torch", torch.__version__, "cuda", torch.version.cuda); print("deep_gemm", m.version("deep_gemm"), deep_gemm.__file__); print("deep_gemm_cpp", deep_gemm_cpp.__file__); print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
