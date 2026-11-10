#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# XeMaya GPU window 5 (prepared, not yet run): DeepSeek-V4.1-Flash's first run on Arc, B70 + B65 only, bounded.
# Only in a window the owner has opened (production Qwen stopped, the router marking it unavailable):
#   tools/xemaya/window5_ds41.sh [FIRST_PHASE]
# Each phase takes the whole-host lease itself and runs under safety_guard.sh; a failure stops the script.
#   1 ds41_parity and glm5_parity on the B70 and the B65
#   2 DeepSeek Coder IQ2_XXS, forced along the llama.cpp CPU references (24- and 576-token prompts): logits compared
#   3 DeepSeek Coder, free greedy run of the 576-token prompt, 128 tokens: load, prompt, decode, VRAM
#   4 GLM-5.3 Maya-S v2 forced along chat-agent (the recurrent checkpoint loop was regrouped for DeepSeek's state)
# SAFETY: level_zero:0,1 (the B60 is never opened); no long-context run; engines end by themselves (no SIGKILL).
set -u
X=/DATA/Projects/xemaya; R=$X/XeMaya; W=${XEMAYA_WINDOW_DIR:-$X/logs/window5}; REF=$R/xemaya/reference; T=$R/tools/xemaya
E=$X/build-engine
DS=/mnt/ai/models/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf
GLM=/mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
FIRST=${1:-1}
mkdir -p $W
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ZES_ENABLE_SYSMAN=1 SYCL_CACHE_PERSISTENT=1
say() { echo "== $(date -u +%T) $*"; }
stop() { say "STOP: $*"; exit 1; }
# NOTE BUDGET_GIB SECONDS CMD...: the lease, the guard, a memory cap and a time limit around one job
job() {
  local note=$1 gib=$2 secs=$3; shift 3
  python3 /DATA/AppData/zability-gpu-lease/gpulease.py run --holder xemaya-dev --memory-budget-gib $gib --note "XeMaya window 5: $note" -- \
    $T/safety_guard.sh --log $W/safety_guard.log -- \
    systemd-run --user --scope -q -p MemoryMax=$((gib + 10))G -p MemorySwapMax=0 nice -n 5 timeout -s TERM $secs "$@"
}
kernel_log() { journalctl -k --since "-20 min" --no-pager | grep -E "NETDEV WATCHDOG|Engine reset|Fault response|AER:|oom-kill" && stop "a kernel message above"; return 0; }
NL=/home/dan/supervisor/NIGHT-LOG.md
open_line=$(grep -n "WINDOW [0-9]* OPEN" $NL | tail -1 | cut -d: -f1)
[ -n "$open_line" ] || stop "no open window in NIGHT-LOG"
tail -n +$((open_line + 1)) $NL | grep -q "XEMAYA DONE: router restored" && stop "the last window in NIGHT-LOG is closed: the owner opens a new one first"
bash /DATA/AppData/strata-void-l3/preflight.sh 40 28 28 | tail -1 | grep -q OK || stop "preflight"

if [ $FIRST -le 1 ]; then
  say "phase 1: kernel parity"
  for d in 0 1; do for t in ds41_parity glm5_parity; do
    ( export ONEAPI_DEVICE_SELECTOR=level_zero:$d; job "$t l0:$d" 8 900 $( [ $t = glm5_parity ] && echo $X/build-icpx || echo $E )/$t ) > $W/$t-l0-$d.log 2>&1
    tail -1 $W/$t-l0-$d.log | grep -q "status 0" || stop "$t on level_zero:$d (see $W/$t-l0-$d.log)"
  done; done
  kernel_log
fi
forced() {   # NAME MODEL REF PROMPT_IDS CONTEXT
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); open(sys.argv[2],'w').write(','.join(str(s['token']) for s in r['steps']))" $3 $W/$1.ref.ids
  job "forced $1" 60 2400 $E/strata --native $2 --tokens-file $4 --greedy --max-context $5 --layer-split auto --mmap-experts \
      --expert-cache auto --max-new $(tr ',' '\n' < $W/$1.ref.ids | grep -c .) --spec 0 --spec-follow $W/$1.ref.ids \
      --dump-logits $W/$1.forced.f32 > $W/$1.out 2> $W/$1.err
  grep -E "strata \(|load|prompt" $W/$1.err | tail -3
  python3 $T/compare_ref.py $3 --forced $W/$1.forced.f32 | tee $W/$1.compare.txt | tail -2
  tail -1 $W/$1.compare.txt | grep -q PASS || { say "forced $1: NOT PASS (continuing; see $W/$1.compare.txt)"; }
  kernel_log
}
if [ $FIRST -le 2 ]; then
  say "phase 2: DeepSeek Coder forced along llama.cpp"
  [ -f $W/ds41-short.compare.txt ] || forced ds41-short $DS $REF/ds41-coder-short.json $X/prompts/ds41/coder-short.ids 2048
  forced ds41-mid $DS $REF/ds41-coder-mid.json $X/prompts/ds41/coder-mid.ids 2048
fi
if [ $FIRST -le 3 ]; then
  say "phase 3: DeepSeek Coder, free run (576-token prompt, 128 tokens)"
  job "ds41 free" 60 2400 $E/strata --native $DS --tokens-file $X/prompts/ds41/coder-mid.ids --greedy --max-context 4096 \
      --layer-split auto --mmap-experts --expert-cache auto --max-new 128 --spec 0 > $W/ds41-free.out 2> $W/ds41-free.err
  grep -E "strata \(|strata plain" $W/ds41-free.err | tail -6
  python3 /DATA/AppData/zability-gpu-lease/gpulease.py status | grep -E '"name"|"used"' | paste - - | cut -c1-120
  kernel_log
fi
if [ $FIRST -le 4 ]; then
  say "phase 4: GLM-5.3 forced along chat-agent"
  forced glm-chat-agent $GLM $REF/glm53-maya-s-v2-chat-agent.json $X/prompts/chat-agent.ids 4096
fi
say "done: run /home/dan/supervisor/xemaya-window-end.sh, verify Qwen, then record the results in xemaya/STATUS.md"
