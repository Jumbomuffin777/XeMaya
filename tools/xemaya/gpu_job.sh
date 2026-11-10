#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# One XeMaya GPU job under the host's rules: the whole-host lease, the safety guard, a memory cap and a time limit.
#   tools/xemaya/gpu_job.sh NOTE HOST_GIB SECONDS CMD...
# It refuses while the production model server is active (the owner stops it; this script never does), and when the
# kernel log of the last 20 minutes holds a NIC watchdog, a GPU engine reset or fault, an AER error or an OOM kill.
# XEMAYA_GPUS chooses the cards (default level_zero:0,1; the card behind the chipset only when it is named).
# XEMAYA_JOB_LOG is the guard's log (default <logs>/gpu_job/safety_guard.log).  The job ends by itself or by TERM.
set -u
T=$(cd "$(dirname "$0")" && pwd)
X=${XEMAYA_HOME:-$(cd "$T/../../.." && pwd)}
LEASE=${XEMAYA_LEASE:-/DATA/AppData/zability-gpu-lease/gpulease.py}
UNIT=${XEMAYA_PRODUCTION_UNIT:-zability-strata-qwen38.service}
[ $# -ge 4 ] || { echo "usage: gpu_job.sh NOTE HOST_GIB SECONDS CMD..." >&2; exit 2; }
note=$1 gib=$2 secs=$3; shift 3
log=${XEMAYA_JOB_LOG:-$X/logs/gpu_job/safety_guard.log}
mkdir -p "$(dirname "$log")"
if systemctl --user is-active -q "$UNIT" 2>/dev/null; then echo "gpu_job: $UNIT is active: no XeMaya GPU job" >&2; exit 3; fi
bad=$(journalctl -k --since "-20 min" --no-pager 2>/dev/null | grep -E "NETDEV WATCHDOG|Engine reset|Fault response|AER:|oom-kill" | tail -3)
if [ -n "$bad" ] && [ "${XEMAYA_IGNORE_KLOG:-0}" != 1 ]; then echo "gpu_job: kernel messages in the last 20 minutes:" >&2; echo "$bad" >&2; exit 4; fi
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export ONEAPI_DEVICE_SELECTOR=${XEMAYA_GPUS:-level_zero:0,1} ZES_ENABLE_SYSMAN=1 SYCL_CACHE_PERSISTENT=1
exec python3 "$LEASE" run --holder xemaya-dev --memory-budget-gib "$gib" --note "XeMaya: $note" -- \
  "$T/safety_guard.sh" --log "$log" -- \
  systemd-run --user --scope -q -p MemoryMax=$((gib + 10))G -p MemorySwapMax=0 nice -n 5 timeout -s TERM "$secs" "$@"
