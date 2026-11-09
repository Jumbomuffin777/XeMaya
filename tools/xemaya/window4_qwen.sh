#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# Window 4: one Qwen3.8 Flash-Next benchmark arm on the B70 + B65 (the production profile's arguments; the B60 is not
# opened: ONEAPI_DEVICE_SELECTOR=level_zero:0,1 from the profile).  Preflight, the whole-host lease, safety_guard.sh.
#   window4_qwen.sh ARM SPEC [extra qwen_bench.py args]      ARM: up (upstream Strata) | xe (XeStrata) | maya (XeMaya); not the engines names, which preflight.sh takes for a running engine
set -u
ARM=$1; SPEC=$2; shift 2
X=/DATA/Projects/xemaya; W=$X/logs/window4; T=$X/XeMaya/tools/xemaya; S=/DATA/AppData/strata-void-l3
mkdir -p $W
set +u; source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; set -u
# XeStrata and XeMaya do not have upstream SYCL port options: the experts come through the page cache instead
XE="--drop=--stream-experts --drop=--trim-stage-weights --drop=--vram-reserve-mib+ --add=--mmap-experts"
case $ARM in
  up)   ARM=strata; EXE=$S/upstream/releases/v0.2.12/engine/strata; EXTRA="--libdir $S/upstream/releases/v0.2.12/engine --cwd $S/upstream/Strata-v0.2.12";;
  xe)   ARM=xestrata; EXE=$X/build-xestrata/strata; EXTRA="--cwd $X/xestrata-0.2.2 $XE";;
  maya) ARM=xemaya; EXE=$X/build-engine/strata; EXTRA="--cwd $X/XeMaya $XE";;
  *) echo "arm?"; exit 2;;
esac
L=${LABEL:-$ARM-spec$SPEC}
bash $S/preflight.sh 40 28 28 | tail -1 | grep -q OK || { echo "PREFLIGHT FAILED"; bash $S/preflight.sh 40 28 28 | tail -3; exit 75; }
python3 /DATA/AppData/zability-gpu-lease/gpulease.py run --holder xemaya-dev --memory-budget-gib 60 --note "XeMaya window 4: Qwen bench $L" -- \
  $T/safety_guard.sh --log $W/safety_guard.log -- \
  systemd-run --user --scope -q -p MemoryMax=70G -p MemorySwapMax=0 nice -n 5 timeout -s TERM 1500 \
  python3 $T/qwen_bench.py --exe $EXE --profile $S/profiles/qwen38-ista-iq3s.json --prompts $X/prompts/qwen \
  --out $W/qwen-$L.json --label $L --spec $SPEC $EXTRA "$@" 2>&1 | tee $W/qwen-$L.log
echo "rc=${PIPESTATUS[0]}"
journalctl -k --since "-30 min" --no-pager | grep -E "NETDEV WATCHDOG|Engine reset|Fault response|AER:|oom-kill" | tail -5
