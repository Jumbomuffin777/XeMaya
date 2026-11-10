#!/bin/bash
# SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# xemaya.sh - the command of XeMaya's Linux release folder (tools/xemaya/release.sh puts it at the folder's top, beside
# engine/ with the built programs).  See INSTALL.md in the same folder.
#
#   ./xemaya.sh setup               a Python venv in the folder: the server's packages, and Intel's SYCL and oneMKL run-time
#                                libraries from PyPI (not needed where oneAPI 2026.2 is installed: XEMAYA_ONEAPI=system)
#   ./xemaya.sh doctor              what is here and what is missing: the libraries, the GPUs the engine sees
#   ./xemaya.sh selftest            the engine's device self-test on the first GPU (seconds, a few MiB)
#   ./xemaya.sh selftest --cpu      the kernels' parity tests on the OpenCL CPU device: no GPU is opened
#   ./xemaya.sh models DIR...       the models found under DIR, as JSON; writes each one's server config and tokenizer
#   ./xemaya.sh serve ID [OPTIONS]  the OpenAI- and Anthropic-compatible server for a model `models` listed
#   ./xemaya.sh version
#
# XEMAYA_HOME is where the configs and tokenizers go (default ~/.local/share/xemaya).  The GPUs are SYCL's:
# ONEAPI_DEVICE_SELECTOR=level_zero:0,1 names two of them (default: the engine's own choice).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
home=${XEMAYA_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/xemaya}
venv=$here/.venv
# Intel's run-time libraries of the compiler the engine was built with (BUILDINFO): the same minor version
rt_pins="dpcpp-cpp-rt==2026.2.0 onemkl-sycl-blas==2026.2.0"

die() { echo "xemaya.sh: $*" >&2; exit 1; }

runtime() {
  # the SYCL runtime: the venv's (setup), else an installed oneAPI's
  if [ -e "$venv/lib/libsycl.so.9" ]; then
    LD_LIBRARY_PATH=$here/engine/lib:$venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    # the OpenCL CPU device of the wheel, for `selftest --cpu` (OCL_ICD_FILENAMES is the ICD loader's own variable)
    if [ -e "$venv/lib/libintelocl.so" ]; then
      OCL_ICD_FILENAMES=$venv/lib/libintelocl.so${OCL_ICD_FILENAMES:+:$OCL_ICD_FILENAMES}
      export OCL_ICD_FILENAMES
    fi
  elif [ -r "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" ]; then
    set +eu
    # shellcheck disable=SC1091
    source "${ONEAPI_ROOT:-/opt/intel/oneapi}/setvars.sh" --force >/dev/null 2>&1
    set -eu
    LD_LIBRARY_PATH=$here/engine/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  else
    die "Intel's SYCL run-time libraries are not here: run ./xemaya.sh setup"
  fi
  export LD_LIBRARY_PATH
}

python() {
  [ -x "$venv/bin/python" ] || die "no Python environment: run ./xemaya.sh setup"
  "$venv/bin/python" "$@"
}

cmd=${1:-help}
[ $# -gt 0 ] && shift
case "$cmd" in
setup)
  command -v python3 >/dev/null || die "python3 is needed (3.10 or later)"
  python3 -m venv "$venv" || die "python3 -m venv failed: install your distribution's python3-venv"
  "$venv/bin/pip" install --disable-pip-version-check -q -r "$here/requirements.txt"
  if [ "${XEMAYA_ONEAPI:-wheels}" = system ]; then
    echo "XEMAYA_ONEAPI=system: using the installed oneAPI (${ONEAPI_ROOT:-/opt/intel/oneapi})"
  else
    # shellcheck disable=SC2086
    "$venv/bin/pip" install --disable-pip-version-check -q $rt_pins
  fi
  echo "set up. Next: ./xemaya.sh doctor"
  ;;
doctor)
  runtime
  bad=0
  for p in strata strata-device; do
    miss=$(ldd "$here/engine/$p" 2>&1 | grep "not found" || true)
    if [ -n "$miss" ]; then echo "engine/$p lacks:"; echo "$miss"; bad=1; fi
  done
  [ $bad = 0 ] || die "libraries are missing: run ./xemaya.sh setup"
  "$here/engine/strata" --version
  if ! ls /dev/dri/renderD* >/dev/null 2>&1; then
    echo "no /dev/dri/renderD*: no GPU is visible here (a container without the device, or no driver)"
  elif ! [ -r "$(find /dev/dri -name 'renderD*' | sort | head -1)" ]; then
    echo "the GPU's device file is not readable: add this user to the render group (sudo usermod -aG render \$USER)"
  fi
  "$here/engine/strata-device" --list-devices || die "the engine found no GPU: Intel's compute runtime (Level Zero) \
is needed - Ubuntu: intel-opencl-icd and libze-intel-gpu1; Fedora: intel-compute-runtime and oneapi-level-zero"
  ;;
selftest)
  runtime
  if [ "${1:-}" = --cpu ]; then
    for t in glm5_parity ds41_parity minimax_parity; do
      echo "== $t (OpenCL CPU device)"
      ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 "$here/engine/selftest/$t" || die "$t failed"
    done
    echo "the CPU-device parity tests passed"
  else
    "$here/engine/strata-device" --selftest
  fi
  ;;
models)
  [ $# -gt 0 ] || die "usage: ./xemaya.sh models DIR..."
  roots=""
  for d in "$@"; do roots="$roots --root $d"; done
  mkdir -p "$home/configs"
  # shellcheck disable=SC2086
  python "$here/tools/xemaya/catalog.py" $roots --configs "$home/configs" --engine "$here/engine/strata" \
    --write --tokenizers --pretty
  ;;
serve)
  [ $# -gt 0 ] || die "usage: ./xemaya.sh serve ID [--port N] [--host ADDRESS] [--api-key KEY | --no-api-key]"
  id=$1; shift
  cfg=$home/configs/$id.json
  [ -r "$cfg" ] || die "no config for $id in $home/configs: run ./xemaya.sh models DIR first (ids are its \"id\" fields)"
  runtime
  export ZES_ENABLE_SYSMAN=1 SYCL_CACHE_PERSISTENT=1
  # the inherited update check asks for XeStrata's releases, which are not this program's: off unless asked for
  export XESTRATA_UPDATE_CHECK="${XESTRATA_UPDATE_CHECK:-0}"
  exec "$venv/bin/python" "$here/serve/server.py" --engine strata --config "$cfg" "$@"
  ;;
version)
  runtime
  "$here/engine/strata" --version
  [ -r "$here/BUILDINFO" ] && cat "$here/BUILDINFO"
  ;;
*)
  sed -n '6,15p' "$0" | sed 's/^# \{0,1\}//'
  ;;
esac
