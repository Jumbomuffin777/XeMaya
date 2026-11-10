#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# XeMaya's engine for Windows with the CUDA target, cross-built on Linux in a container that stays between runs
# (tools/package/windows.sh builds the whole 7z in one go from a commit; this one is for working on the Windows and
# NVIDIA code: the toolchains are built once, the working tree is the source, a changed file is one compile).
#
#   win_build.sh image                 the container image: Ubuntu 26.04 with the cross tools and NVIDIA's CUDA toolkit
#   win_build.sh toolchain             intel/llvm (the contrib toolchain, then the SYCL runtime for Windows), xwin's
#                                      MSVC CRT and Windows SDK (Microsoft's license is accepted here), NVIDIA's
#                                      Windows import libraries
#   win_build.sh configure [-D...]     CMake for x86_64-pc-windows-msvc, the contrib-llvm mode
#   win_build.sh build [TARGET..]      ninja (default: strata)
#   win_build.sh worker DIR            the folder a hybrid segment worker runs from (strata.exe and the DLLs it loads)
#   win_build.sh run COMMAND..         a command in the container (/src: the working tree, /work: the work folder)
#
# Environment: XEMAYA_WIN_WORK (the work folder: downloads, the build tree; required), XEMAYA_WIN_ARCHS (the NVIDIA
# architectures, default the package's list), XEMAYA_WIN_MEMORY (default 40g), XEMAYA_WIN_CPUS (default 16),
# XEMAYA_CONTAINER (docker or podman; default: whichever is there).
# What is built does not run here: it needs a Windows PC with NVIDIA's driver.
set -euo pipefail
src=$(cd "$(dirname "$0")/../.." && pwd)
work=${XEMAYA_WIN_WORK:?set XEMAYA_WIN_WORK to a folder for the downloads and the build tree}
engine=${XEMAYA_CONTAINER:-$(command -v docker || command -v podman)}
image=xemaya-win-build:1
archs=${XEMAYA_WIN_ARCHS:-sm_60;sm_61;sm_70;sm_75;sm_80;sm_86;sm_89;sm_90}
mkdir -p "$work/home" "$work/cache" "$work/build"

inside() {
    "$engine" run --rm -i --memory "${XEMAYA_WIN_MEMORY:-40g}" --cpus "${XEMAYA_WIN_CPUS:-16}" \
        -u "$(id -u):$(id -g)" -e HOME=/work/home -e CCACHE_DIR=/work/ccache -e CCACHE_COMPILERCHECK=content \
        -e XEMAYA_WIN_ARCHS="$archs" -v "$src":/src -v "$work":/work -w /src "$image" "$@"
}

case ${1:-} in
image)
    "$engine" build -t "$image" - <<'DOCKER'
FROM docker.io/library/ubuntu@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7
ENV DEBIAN_FRONTEND=noninteractive
# as tools/package/windows.sh, without what the engine's `strata` target does not take (.NET, ROCm, 7z)
RUN apt-get update -q && apt-get install -y -q --no-install-recommends git cmake ninja-build g++ libhwloc-dev \
    libzstd-dev python3 python3-pip python3-venv curl ca-certificates unzip ccache lld lld-21 llvm-21 clang-21 \
    nvidia-cuda-toolkit opencl-headers patch xz-utils && rm -rf /var/lib/apt/lists/*
DOCKER
    ;;
toolchain)
    inside sh -eu <<'SH'
git config --global --add safe.directory '*'
python3 tools/intel_llvm_build.py --contrib --yes
xwin_ver=0.10.0
xwin_tar=xwin-$xwin_ver-x86_64-unknown-linux-musl.tar.gz
if [ ! -d /work/xwin ]; then
    curl -fsSL -o "/work/cache/$xwin_tar" "https://github.com/Jake-Shadle/xwin/releases/download/$xwin_ver/$xwin_tar"
    curl -fsSL -o "/work/cache/$xwin_tar.sha256" "https://github.com/Jake-Shadle/xwin/releases/download/$xwin_ver/$xwin_tar.sha256"
    echo "$(cat "/work/cache/$xwin_tar.sha256")  /work/cache/$xwin_tar" | sha256sum -c
    tar -xzf "/work/cache/$xwin_tar" -C /work/cache
    "/work/cache/xwin-$xwin_ver-x86_64-unknown-linux-musl/xwin" --accept-license --cache-dir /work/cache/xwin \
        --arch x86_64 splat --copy --output /work/xwin.new
    mv /work/xwin.new /work/xwin
fi
python3 tools/intel_llvm_build.py --windows --xwin /work/xwin
# Windows' import libraries for cuBLAS, the CUDA runtime and the CUDA driver, and the toolkit's headers (the same on
# Windows): tools/package/windows.sh's, its archives and their SHA-256
mkdir -p /work/cudalib/include
cuda_lib() {
    f=/work/cache/$(basename "$1")
    [ -f "$f" ] || curl -fsSL -o "$f" "https://developer.download.nvidia.com/compute/cuda/redist/$1"
    echo "$2  $f" | sha256sum -c -
    unzip -q -j -o "$f" '*/lib/x64/cublas.lib' '*/lib/x64/cudart.lib' '*/lib/x64/cuda.lib' -d /work/cudalib || true
}
cuda_lib cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.4.127-archive.zip \
    6a1c32e68ee1a95ca17334691ff9ad1ffe7f352c24a083d55e4c96b8063b2bcb
cuda_lib libcublas/windows-x86_64/libcublas-windows-x86_64-12.4.5.8-archive.zip \
    698140f12da055a3709eee2e022fcfe7bc8edf31f30115e3f7a5c877a9491de5
for l in cublas cudart cuda; do [ -f "/work/cudalib/$l.lib" ] || { echo "no $l.lib" >&2; exit 1; }; done
dpkg -L nvidia-cuda-dev libcu++-dev | sed -n 's|^/usr/include/||p' |
    while read -r h; do [ -f "/usr/include/$h" ] && echo "$h"; done | tar -C /usr/include -cf - -T - |
    tar -C /work/cudalib/include -xf -
echo "toolchain ready"
SH
    ;;
configure)
    shift
    inside sh -eu -c 'cmake -G Ninja -S /src -B /work/build/engine -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_TOOLCHAIN_FILE=/src/cmake/toolchain-windows-msvc.cmake \
        -DCMAKE_CXX_COMPILER=/src/.tools/intel-llvm-contrib/install/bin/clang++ -DSTRATA_LICENSE=contrib \
        -DSTRATA_XWIN=/work/xwin -DSTRATA_SYCL_WINDOWS=/src/.tools/intel-llvm-win/install \
        -DSTRATA_CUDA_ARCHS="$XEMAYA_WIN_ARCHS" -DSTRATA_CUDA_PATH=/usr/lib/cuda -DSTRATA_CUDA_PTX= \
        -DSTRATA_ONEMKL=OFF -DSTRATA_HIP_ARCHS= -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON \
        -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON -DSTRATA_GGML_DIR=/src/third_party/main/llama.cpp \
        -DSTRATA_CUDA_WINDOWS_LIBS=/work/cudalib -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@"' sh "$@"
    ;;
build)
    shift
    if [ $# -eq 0 ]; then set -- strata strata-device; fi
    inside ninja -C /work/build/engine "$@"
    ;;
worker)
    # the folder a segment worker runs from: the engine, intel/llvm's SYCL runtime and oneMath with its cuBLAS backend
    # (tools/package/windows_bundle.py's lists, the NVIDIA part).  NVIDIA's driver and cuBLAS, and Microsoft's Visual
    # C++ runtime, are the PC's own.
    out=${2:?the folder to write}
    mkdir -p "$out"
    cp "$work/build/engine/strata.exe" "$work/build/engine/strata-device.exe" "$out/"
    for d in sycl9.dll ur_win_proxy_loader.dll ur_adapter_cuda.dll xptifw.dll; do
        cp "$src/.tools/intel-llvm-win/install/bin/$d" "$out/"
    done
    cp "$work/build/engine/bin/onemath.dll" "$work/build/engine/bin/onemath_blas_cublas.dll" "$out/"
    git -C "$src" rev-parse HEAD > "$out/COMMIT"
    (cd "$out" && rm -f SHA256SUMS && sha256sum -- * > .sums && mv .sums SHA256SUMS)
    ls -l "$out"
    ;;
run)
    shift
    inside "$@"
    ;;
*)
    sed -n '5,22p' "$0" >&2
    exit 2
    ;;
esac
