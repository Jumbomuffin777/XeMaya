#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# XeMaya GPU window 3: GLM-5.3-Flash Maya-S v2 attention, TTFT, long context and agent use. Run inside the whole-host
# lease with production stopped by the owner:
#   python3 /DATA/AppData/zability-gpu-lease/gpulease.py run --holder xemaya-dev --memory-budget-gib 60 -- \
#       bash tools/xemaya/window3.sh [FIRST_PHASE]
# Phases (each logs to logs/window3/; a failure stops the script, nothing is retried or changed in the window):
#   1 kernel parity on each card (glm5 with the tiled attention and the F32 mixers, ds41, iq) + the attention benchmark
#   2 engine, original attention, spec 0, prompt profile: ctx-2377, ctx-8k                     (the A of the A/B)
#   3 engine, tiled attention, the same                                                         (the B)
#   4 engine, the faster attention, spec 0: correctness (chat-agent, code-512, ctx-2377 against llama.cpp), uncached
#     prefill and decode at 64K, 32K, 8K; the prompt-end checkpoint
#   4c forced along each llama.cpp reference (chat-agent, code-512, ctx-2377), logits compared
#   4b the same at 32K with the DSA cells in VRAM (--kv-ram off)
#   5 engine, the faster attention, spec 3 (MTP drafts): ctx-2377, ctx-32k, the checkpoint with drafts
#   6 the HTTP server (OpenAI / Anthropic routes): a coding agent's session, the checkpoint on; then off for 3 turns
#   7 (optional, FIRST_PHASE=7) ctx-128k
set -u
X=/DATA/Projects/xemaya; R=$X/XeMaya; W=$X/logs/window3; P=$X/prompts; REF=$R/xemaya/reference
M=/mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
E=$X/build-engine/strata
FIRST=${1:-1}
mkdir -p $W
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export ZES_ENABLE_SYSMAN=1 SYCL_CACHE_PERSISTENT=1
cap() { systemd-run --user --scope -q -p MemoryMax=$1 -p MemorySwapMax=0 nice -n 5 "${@:2}"; }
stop() { echo "== $(date -u +%T) STOP: $*"; exit 1; }
say() { echo "== $(date -u +%T) $*"; }
plan() { python3 -c "import json,sys; print(json.dumps({'jobs': json.loads(sys.argv[1])}))" "$1"; }
serve() {   # NAME MLA SPEC CONTEXT PLAN_JSON [env...]
  local name=$1 mla=$2 spec=$3 ctx=$4 pj=$5; shift 5
  echo "$pj" > $W/plan-$name.json
  ( export XEMAYA_MLA_ATTN=$mla; for kv in "$@"; do export "$kv"; done   # (cap is a function: env cannot run it)
    cap 70G timeout 7200 python3 $R/tools/xemaya/engine_serve.py --exe $E --model $M \
      --out $W/$name.json --engine-arg=--layer-split --engine-arg=auto --engine-arg=--max-context --engine-arg=$ctx \
      --engine-arg=--spec --engine-arg=$spec $W/plan-$name.json ) 2>&1 | tee $W/$name.txt
  return ${PIPESTATUS[0]}
}
say "start (phase $FIRST)"
bash /DATA/AppData/strata-void-l3/preflight.sh 40 28 28 20 || stop "preflight"
setsid nohup $X/monitor3.sh > /dev/null 2>&1 < /dev/null &

if [ $FIRST -le 1 ]; then
  say "1. kernel parity, one card each, in parallel"
  for d in 0 1 2; do
    ( for t in glm5_parity ds41_parity "iq_parity $X/iq_fixture"; do
        n=${t%% *}
        ONEAPI_DEVICE_SELECTOR=level_zero:$d cap 12G timeout 300 stdbuf -oL $X/build-icpx/$t > $W/$n-l0-$d.log 2>&1
        echo "$n level_zero:$d rc=$? last: $(tail -1 $W/$n-l0-$d.log)"
      done ) &
    pids="${pids:-} $!"
  done
  wait $pids   # (not a bare wait: the telemetry recorder is a child too)
  grep -L -E "^PASS|iq_parity: 0 failures" $W/*_parity-l0-*.log | grep -q . && stop "a kernel test failed"
  say "1b. the attention benchmark (B70, then B60)"
  for d in 0 2; do for p in 2377 8192 32768 65536; do
    ONEAPI_DEVICE_SELECTOR=level_zero:$d cap 16G timeout 600 $X/build-icpx/glm5_parity --bench-mla $p 2>&1 | grep -v XEMAYA | tee -a $W/bench-mla-l0-$d.txt
  done; done
fi
J_AB=$(plan '[{"kind":"gen","name":"ctx-2377","prompt":"'$P'/ctx-2377.ids","max_new":128},
              {"kind":"gen","name":"ctx-8k","prompt":"'$P'/ctx-8k.ids","max_new":128}]')
if [ $FIRST -le 2 ]; then say "2. original attention (A)"; serve A-ref ref 0 16384 "$J_AB" STRATA_PROMPT_PROFILE=1 || stop "phase 2"; fi
if [ $FIRST -le 3 ]; then say "3. tiled attention (B)"; serve B-tiled tiled 0 16384 "$J_AB" STRATA_PROMPT_PROFILE=1 || stop "phase 3"; fi
# the faster of the two by the 2,377- and 8K-token prompts' read times (DONE's prompt_ms)
MLA=${MLA_FORCE:-}
[ -n "$MLA" ] || MLA=$(python3 - "$W" <<'EOF'
import json, sys
d = sys.argv[1]
def t(n):
    try: return sum(j["prompt_ms"] for j in json.load(open(f"{d}/{n}.json"))["jobs"])
    except Exception: return None
a, b = t("A-ref"), t("B-tiled")
print("tiled" if a is None or (b is not None and b <= a) else "ref")
EOF
)
say "the faster attention: $MLA"
if [ $FIRST -le 4 ]; then
  say "4. correctness, long context, the prompt-end checkpoint (spec 0, $MLA)"
  J=$(plan '[{"kind":"gen","name":"chat-agent","prompt":"'$P'/chat-agent.ids","max_new":128,"ref":"'$REF'/glm53-maya-s-v2-chat-agent.json"},
             {"kind":"gen","name":"code-512","prompt":"'$P'/code-512.ids","max_new":64,"ref":"'$REF'/glm53-maya-s-v2-code-512.json"},
             {"kind":"gen","name":"ctx-2377","prompt":"'$P'/ctx-2377.ids","max_new":64,"ref":"'$REF'/glm53-maya-s-v2-ctx-2377.json"},
             {"kind":"gen","name":"ctx-64k","prompt":"'$P'/ctx-64k.ids","max_new":128},
             {"kind":"gen","name":"ctx-32k","prompt":"'$P'/ctx-32k.ids","max_new":128},
             {"kind":"gen","name":"ctx-8k","prompt":"'$P'/ctx-8k.ids","max_new":128},
             {"kind":"checkpoint","name":"checkpoint","first":"'$P'/chat-agent.ids","then":"'$P'/code-512.ids","max_new":64,"reset":"'$P'/ctx-2377.ids"}]')
  serve C-main $MLA 0 69632 "$J" || say "phase 4 reported a failure (see C-main.txt); going on"
fi
if [ $FIRST -le 4 ]; then   # 4b: the DSA cells in VRAM (--kv-ram off): decode at 32K without the per-token gather
  say "4b. ctx-32k with --kv-ram off"
  J=$(plan '[{"kind":"gen","name":"ctx-32k-kvvram","prompt":"'$P'/ctx-32k.ids","max_new":128}]')
  echo "$J" > $W/plan-C2-kvvram.json
  ( export XEMAYA_MLA_ATTN=$MLA
    cap 70G timeout 3600 python3 $R/tools/xemaya/engine_serve.py --exe $E --model $M --out $W/C2-kvvram.json \
      --engine-arg=--layer-split --engine-arg=auto --engine-arg=--max-context --engine-arg=36864 --engine-arg=--spec \
      --engine-arg=0 --engine-arg=--kv-ram --engine-arg=off $W/plan-C2-kvvram.json ) 2>&1 | tee $W/C2-kvvram.txt
fi
if [ $FIRST -le 4 ]; then   # 4c: forced along each reference (its near-ties cannot hide a divergence), logits compared
  for r in ctx-2377 code-512 chat-agent; do
    say "4c. forced along the llama.cpp reference: $r"
    python3 -c "import json,sys; print(','.join(map(str, json.load(open(sys.argv[1]))['tokens'])))" \
        $REF/glm53-maya-s-v2-$r.json > $W/forced-$r.ref.ids
    n=$(tr ',' '\n' < $W/forced-$r.ref.ids | grep -c .)
    ( export XEMAYA_MLA_ATTN=$MLA
      cap 70G timeout 1800 $E --native $M --tokens-file $P/$r.ids --greedy --layer-split auto --max-context 8192 \
          --max-new $n --spec 0 --spec-follow $W/forced-$r.ref.ids --dump-logits $W/forced-$r.f32 \
          > $W/forced-$r.out 2> $W/forced-$r.err )
    python3 $R/tools/xemaya/compare_ref.py $REF/glm53-maya-s-v2-$r.json --forced $W/forced-$r.f32 | tee $W/forced-$r.txt | tail -2
  done
fi
if [ $FIRST -le 5 ]; then
  say "5. MTP drafts (spec 3, $MLA)"
  J=$(plan '[{"kind":"gen","name":"ctx-2377","prompt":"'$P'/ctx-2377.ids","max_new":128},
             {"kind":"gen","name":"ctx-32k","prompt":"'$P'/ctx-32k.ids","max_new":128},
             {"kind":"checkpoint","name":"checkpoint-spec","first":"'$P'/chat-agent.ids","then":"'$P'/code-512.ids","max_new":64,"reset":"'$P'/ctx-2377.ids"}]')
  serve D-spec $MLA 3 36864 "$J" || say "phase 5 reported a failure (see D-spec.txt); going on"
fi
if [ $FIRST -le 6 ]; then
  for ck in 1 0; do
    say "6. the HTTP server, an agent's session (checkpoint $ck)"
    cat > $W/server-$ck.json <<EOF
{"exe": "$E", "args": ["--native", "$M", "--max-context", "65536", "--layer-split", "auto"],
 "cwd": "$X/serve-glm", "tokenizer": "$X/serve-glm/tok/tokenizer", "model_name": "glm-5.3-flash-maya-s-v2",
 "log": "$W/server-$ck.engine.log", "host": "127.0.0.1", "port": 18990}
EOF
    ( cd $R && export XEMAYA_MLA_ATTN=$MLA XEMAYA_PROMPT_CHECKPOINT=$ck && cap 70G $X/venv/bin/python serve/server.py --engine strata \
        --config $W/server-$ck.json --host 127.0.0.1 --port 18990 --no-api-key > $W/server-$ck.log 2>&1 ) &
    for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:18990/health | grep -q '"loaded": true' && break; sleep 5; done
    curl -s -m 2 http://127.0.0.1:18990/health | head -c 200; echo
    files=6; [ $ck = 0 ] && files=3
    timeout 3600 $X/venv/bin/python $R/tools/xemaya/agent_session.py --url http://127.0.0.1:18990 --out $W/agent-$ck.json \
        --files $files 2>&1 | grep -v RuntimeWarning | tee $W/agent-$ck.txt
    pkill -INT -f "[s]erve/server.py --engine strata --config $W/server-$ck.json"; sleep 15
    pkill -9 -f "[s]erve/server.py --engine strata --config $W/server-$ck.json"; pkill -9 -f "[b]uild-engine/strata --serve"
    sleep 3
  done
fi
if [ $FIRST -eq 7 ]; then
  say "7. ctx-128k"
  serve E-128k $MLA 0 135168 "$(plan '[{"kind":"gen","name":"ctx-128k","prompt":"'$P'/ctx-128k.ids","max_new":64}]')" || say "phase 7 failed"
fi
pkill -f "[x]emaya/monitor3.sh"; pkill -f "[x]pu-smi dump -d"
python3 $R/tools/xemaya/window_report.py $W > $W/report.txt 2>&1; cat $W/report.txt
say "done"
