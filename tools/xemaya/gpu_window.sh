#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# XeMaya's GPU window: run inside the whole-host lease, with production stopped by the owner:
#   python3 /DATA/AppData/zability-gpu-lease/gpulease.py run --holder xemaya-dev --memory-budget-gib 60 -- \
#       bash tools/xemaya/gpu_window.sh
# 1. kernel parity on each card (glm5_parity with the F32 mixer checks, ds41_parity): a failure stops the window
# 2. GLM-5.3-Flash Maya-S v2 on B70 + B65 + B60 (--layer-split auto), forced along the saved llama.cpp reference's 16
#    tokens with every step's logits written, compared by tools/xemaya/compare_ref.py
# 3. a free greedy run of 128 tokens (its first 16 against the reference; decode speed without drafts)
# 4. iq_parity on its fixtures, each card, 5 minutes at most: reported, not a stop (on the CPU device its IQ4_NL step
#    took ~20 minutes in the OpenCL CPU compiler; it passed there)
# 5. the speed test: 128 tokens with the MTP layer's drafts (--spec 3)
# Any other failure stops the script (no step is retried or changed in the window). Logs: logs/window2/.
set -u
X=/DATA/Projects/xemaya
R=$X/XeMaya
W=$X/logs/window2
M=/mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
REF=$R/xemaya/reference/glm53-maya-s-v2-greedy16.json
E=$X/build-engine/strata
mkdir -p "$W"
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export ZES_ENABLE_SYSMAN=1
export SYCL_CACHE_PERSISTENT=1   # the engine is JIT-compiled (no -fsycl-targets): later runs reuse the first run's binaries
cap() { systemd-run --user --scope -q -p MemoryMax=$1 -p MemorySwapMax=0 nice -n 5 "${@:2}"; }
vram() { python3 /DATA/AppData/zability-gpu-lease/gpulease.py status | python3 -c 'import json,sys; t=json.load(sys.stdin)["telemetry"]["devices"]; print(" | ".join("%s %.1f GiB" % (d["name"].split()[-2], d["used"]/2**30) for d in t))'; }
stop() { echo "== $(date -u +%T) STOP: $*"; echo "VRAM: $(vram)"; exit 1; }
echo "== $(date -u +%T) start"
bash /DATA/AppData/strata-void-l3/preflight.sh 40 28 28 20 || stop "preflight"
python3 -c "import json; print(','.join(map(str, json.load(open('$REF'))['tokens'])))" > "$W/ref.ids" || stop "reference"
python3 -c "import json; print(','.join(map(str, json.load(open('$REF'))['prompt_ids'])))" > "$W/prompt.ids" || stop "prompt"
echo "prompt $(cat $W/prompt.ids); reference $(cat $W/ref.ids)"

if [ "${START:-1}" -le 1 ]; then
echo "== $(date -u +%T) 1. kernel parity, one card each, in parallel"
for d in 0 1 2; do
  ( for t in glm5_parity ds41_parity; do
      ONEAPI_DEVICE_SELECTOR=level_zero:$d cap 12G timeout 300 $X/build-icpx/$t > $W/$t-l0-$d.log 2>&1
      echo "$t level_zero:$d rc=$? last: $(tail -1 $W/$t-l0-$d.log)"
    done ) &
done
wait
grep -L "^PASS" $W/glm5_parity-l0-*.log $W/ds41_parity-l0-*.log | grep -q . && stop "a kernel test failed (above)"
fi

COMMON=(--native "$M" --tokens-file "$W/prompt.ids" --greedy --layer-split auto --max-context 4096)
echo "== $(date -u +%T) 2. Maya-S v2 on the three cards, forced along the reference, logits of every step"
cap 60G timeout 1200 $E "${COMMON[@]}" --max-new 16 --spec 0 --spec-follow "$W/ref.ids" --dump-logits "$W/forced.f32" \
    > "$W/forced.out" 2> "$W/forced.err"
rc=$?; tail -4 "$W/forced.err"; echo "VRAM after: $(vram)"
[ $rc = 0 ] || stop "forced run rc=$rc"
python3 $R/tools/xemaya/compare_ref.py "$REF" --forced "$W/forced.f32" | tee "$W/compare-forced.txt"

echo "== $(date -u +%T) 3. free greedy run, 128 tokens, no drafts"
cap 60G timeout 1200 $E "${COMMON[@]}" --max-new 128 --spec 0 > "$W/free.out" 2> "$W/free.err"
rc=$?; tail -3 "$W/free.err"
[ $rc = 0 ] || stop "free run rc=$rc"
python3 $R/tools/xemaya/compare_ref.py "$REF" --free "$W/free.out" | tee "$W/compare-free.txt"

echo "== $(date -u +%T) 4. iq_parity on the fixtures, one card each, in parallel (reported, not a stop)"
for d in 0 1 2; do
  ( ONEAPI_DEVICE_SELECTOR=level_zero:$d cap 12G timeout 300 stdbuf -oL $X/build-icpx/iq_parity $X/iq_fixture \
        > $W/iq_parity-l0-$d.log 2>&1
    echo "iq_parity level_zero:$d rc=$? last: $(tail -1 $W/iq_parity-l0-$d.log)" ) &
done
wait

echo "== $(date -u +%T) 5. speed with the MTP layer's drafts (--spec 3), 128 tokens"
cap 60G timeout 1200 $E "${COMMON[@]}" --max-new 128 --spec 3 > "$W/spec.out" 2> "$W/spec.err"
rc=$?; tail -3 "$W/spec.err"
[ $rc = 0 ] || stop "spec run rc=$rc"
cmp -s <(grep ^tokens: "$W/free.out") <(grep ^tokens: "$W/spec.out") && echo "spec run: the same 128 tokens as the free run" \
    || echo "spec run: tokens differ from the free run (greedy drafts should not change them; see spec.out)"
echo "== $(date -u +%T) done; VRAM: $(vram)"
