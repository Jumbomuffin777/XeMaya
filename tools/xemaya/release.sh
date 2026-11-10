#!/bin/bash
# SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# tools/xemaya/release.sh - XeMaya's Linux release folder for Intel GPUs, from a commit alone.
#
#   tools/xemaya/release.sh [COMMIT]          (default HEAD; then the work tree must be clean)
#
# The input is the commit (git archive): nothing of the work tree or of an earlier build goes in.  It is built with
# Intel oneAPI's icpx (the compiler XeMaya's recorded results were made with; ONEAPI_ROOT, default /opt/intel/oneapi),
# Intel GPUs only (no CUDA, no HIP), ggml-cpu for any AVX2 CPU.  The host tests and the kernels' parity tests on the
# OpenCL CPU device run before anything is packed: no GPU is opened.  Out comes, in XEMAYA_RELEASE_OUT (default
# build/release):
#
#   xemaya-<version>-linux-x86_64.tar.xz    the commit's source with engine/ (the built programs and oneMath), the
#                                           `xemaya.sh` command and BUILDINFO
#   xemaya-<version>-source.tar.gz          the commit's source alone
#   SHA256SUMS                              of the two
#
# Intel's run-time libraries (SYCL, oneMKL) are not in the folder: `./xemaya.sh setup` installs them from PyPI, or an
# installed oneAPI is used.  Needs cmake, ninja, patchelf and xz.  XEMAYA_CMAKE_ARGS adds CMake arguments (a local
# source of a fetched dependency, for example).  XEMAYA_JOBS is the number of files compiled at once (default 4); XEMAYA_GGML_DIR a
# llama.cpp checkout at the pinned commit (default: CMake fetches it).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
commit=${1:-HEAD}
if [ "$commit" = HEAD ] && [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "the work tree has changes that are not committed: a release is built from a commit" >&2; exit 1
fi
rev=$(git rev-parse --verify "$commit^{commit}")
version=$(git show "$rev:CMakeLists.txt" | sed -n 's/^set(XEMAYA_VERSION "\(.*\)")$/\1/p')
[ -n "$version" ] || { echo "no XEMAYA_VERSION in CMakeLists.txt at $rev" >&2; exit 1; }
out=$(realpath -m "${XEMAYA_RELEASE_OUT:-build/release}")
name=xemaya-$version
work=$out/work
jobs=${XEMAYA_JOBS:-4}
rm -rf "$work" "$out/$name-linux-x86_64.tar.xz" "$out/$name-source.tar.gz" "$out/SHA256SUMS"
mkdir -p "$work/$name" "$work/build"

echo "== $name from $rev"
git archive --format=tar "$rev" | tar -x -C "$work/$name"
# the source archive: the commit as it is, the same bytes on any machine
git archive --format=tar.gz --prefix="$name/" -o "$out/$name-source.tar.gz" "$rev"

set +eu
# shellcheck disable=SC1091
source "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null 2>&1
set -eu
command -v patchelf >/dev/null || { echo "patchelf is needed (your distribution's package, or pip install patchelf)" >&2; exit 1; }
command -v icpx >/dev/null || { echo "icpx is not on PATH after setvars.sh (ONEAPI_ROOT)" >&2; exit 1; }

ggml=()
[ -n "${XEMAYA_GGML_DIR:-}" ] && ggml=(-DSTRATA_GGML_DIR="$XEMAYA_GGML_DIR")
# XEMAYA_CMAKE_ARGS is several arguments: split on purpose
# shellcheck disable=SC2086
cmake -S "$work/$name" -B "$work/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
  -DSTRATA_LICENSE=contrib -DXEMAYA_ALLOW_ICPX=ON -DSTRATA_CUDA_ARCHS= -DSTRATA_HIP_ARCHS= \
  -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=ON "${ggml[@]}" \
  ${XEMAYA_CMAKE_ARGS:-} \
  > "$out/configure.log" 2>&1 || { tail -30 "$out/configure.log"; exit 1; }

host_tests=(gguf_split_test gguf_reader_test model_spec_test segment_link_test share_rules_test split_rules_test
            suffix_drafter_test controller_test draft_policy_test conv_cache_test conversation_cache_test
            logit_bias_test batch_groups_test)
parity=(glm5_parity ds41_parity minimax_parity)
ninja -C "$work/build" -j"$jobs" strata strata-device "${parity[@]}" "${host_tests[@]}" > "$out/build.log" 2>&1 \
  || { tail -40 "$out/build.log"; exit 1; }

echo "== tests (no GPU is opened: SYCL sees the OpenCL CPU device alone)"
rx="^($(IFS='|'; echo "${host_tests[*]}|${parity[*]}"))\$"
ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 ctest --test-dir "$work/build" -R "$rx" -j2 --output-on-failure \
  > "$out/tests.log" 2>&1 || { tail -40 "$out/tests.log"; exit 1; }
tail -3 "$out/tests.log"
ran=$(grep -c "Passed" "$out/tests.log")
[ "$ran" -eq $((${#host_tests[@]} + ${#parity[@]})) ] || { echo "ran $ran tests, not all of the list" >&2; exit 1; }

echo "== the folder"
dst=$work/$name
mkdir -p "$dst/engine/lib" "$dst/engine/selftest"
cp "$work/build/strata" "$work/build/strata-device" "$dst/engine/"
# MiniMax-M3.1 is held back from the release: its parity test runs above and is not shipped
cp "$work/build/glm5_parity" "$work/build/ds41_parity" "$dst/engine/selftest/"
# oneMath and its oneMKL backend (Apache-2.0, built here from the fork CMake fetches): the real files, by soname
for lib in "$work"/build/lib/libonemath*.so.[0-9]*; do cp "$lib" "$dst/engine/lib/"; done
# oneMath opens its backend by the name without a version (libonemath_blas_mklgpu.so): without it the engine's
# matrix products fall back to its own kernels
for lib in "$dst"/engine/lib/*.so.[0-9]*; do ln -s "$(basename "$lib")" "${lib%.so.*}.so"; done
strip --strip-unneeded "$dst"/engine/strata "$dst"/engine/strata-device "$dst"/engine/selftest/* "$dst"/engine/lib/*.so.[0-9]*
# the programs find oneMath beside them, wherever the folder is put; `xemaya.sh` adds Intel's libraries to the path
# shellcheck disable=SC2016
{
patchelf --set-rpath '$ORIGIN/lib' "$dst/engine/strata" "$dst/engine/strata-device"
patchelf --set-rpath '$ORIGIN/../lib' "$dst"/engine/selftest/*
patchelf --set-rpath '$ORIGIN' "$dst"/engine/lib/*.so.[0-9]*
}
cp "$dst/tools/xemaya/xemaya.sh" "$dst/xemaya.sh"
chmod +x "$dst/xemaya.sh"
{
  echo "XeMaya $version"
  echo "commit $rev"
  echo "built $(git show -s --format=%cI "$rev") (the commit's date)"
  echo "compiler $(icpx --version | head -1)"
  echo "mode contrib (Intel GPUs only: oneMKL through oneMath; no CUDA, no HIP)"
  echo "engine $("$dst/engine/strata" --version | tr '\n' ' ')"
  echo "run-time libraries: Intel oneAPI 2026.2 (SYCL, oneMKL), Level Zero and Intel's compute runtime; not included"
} > "$dst/BUILDINFO"
(cd "$dst/engine" && find . -type f | sort | xargs sha256sum) >> "$dst/BUILDINFO"

# the same archive from the same inputs: the files' order, owner and times are fixed (the commit's date)
stamp=$(git show -s --format=%ct "$rev")
tar -C "$work" --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$stamp" -cf - "$name" \
  | xz -T0 -6 > "$out/$name-linux-x86_64.tar.xz"
(cd "$out" && sha256sum "$name-linux-x86_64.tar.xz" "$name-source.tar.gz" > SHA256SUMS)
rm -rf "$work"
echo "== built in $out"
ls -l "$out"
cat "$out/SHA256SUMS"
