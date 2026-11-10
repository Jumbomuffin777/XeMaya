#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# One engine run for a GPU window's benchmarks, inside the whole-host lease:
#   gpulease.py run --holder xemaya-dev --memory-budget-gib 60 -- bash bench_run.sh NAME PROMPT_IDS [engine args...]
# Writes logs/window2/bench-NAME.{out,err,meta}: start/end times and the NVMe sectors read meanwhile.
set -u
X=${XEMAYA_WORK:?the work folder: LOCAL.md}; W=$X/logs/window2; NAME=$1; P=$2; shift 2
M=${XEMAYA_MODELS:?the model folder: LOCAL.md}/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export ZES_ENABLE_SYSMAN=1 SYCL_CACHE_PERSISTENT=1
s0=$(awk '$3=="nvme0n1"{print $6}' /proc/diskstats); t0=$(date +%s.%N)
systemd-run --user --scope -q -p MemoryMax=${MEMMAX:-60G} -p MemorySwapMax=0 timeout 1800 $X/build-engine/strata \
    --native "$M" --tokens-file "$P" --greedy "$@" > $W/bench-$NAME.out 2> $W/bench-$NAME.err
rc=$?; t1=$(date +%s.%N); s1=$(awk '$3=="nvme0n1"{print $6}' /proc/diskstats)
echo "name=$NAME rc=$rc start=$t0 end=$t1 nvme_read_GB=$(awk -v a=$s0 -v b=$s1 'BEGIN{printf "%.2f", (b-a)*512/1e9}') args=$*" \
    | tee $W/bench-$NAME.meta
grep -E "load .* s, prompt|below the VRAM|windows checked|layer split|VRAM slots|read in place|pinned" $W/bench-$NAME.err | cut -c1-260
exit $rc
