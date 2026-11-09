#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# XeMaya GPU window (2026-11-09): runs inside `gpulease.py run --holder xemaya-dev` (whole-host lease).
# Order: preflight; llama.cpp v0.8.0 SYCL reference (loopback llama-server: tokenize + 16 greedy tokens, then stopped);
# XeMaya GLM Maya-S v2 greedy 64 tokens on the same ids; kernel parity on each card.
set -u
W=/DATA/Projects/xemaya/logs/window
B=/DATA/Projects/xemaya
M=/mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
L=/DATA/AppData/maya-upstream/llama.cpp-master-20261029/build-sycl-v0.8.0/bin
PORT=18997
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
export ZES_ENABLE_SYSMAN=1
cap() { systemd-run --user --scope -q -p MemoryMax=$1 -p MemorySwapMax=0 nice -n 5 "${@:2}"; }
vram() { python3 /DATA/AppData/zability-gpu-lease/gpulease.py status | python3 -c 'import json,sys; t=json.load(sys.stdin)["telemetry"]["devices"]; print(" | ".join("%s %.1f GiB" % (d["name"].split()[-2], d["used"]/2**30) for d in t))'; }
echo "== $(date -u +%T) start"
bash /DATA/AppData/strata-void-l3/preflight.sh 40 28 28 20 || exit 75
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 2; }

if [ "${SKIP_REF:-0}" != 1 ]; then
echo "== $(date -u +%T) llama.cpp v0.8.0 SYCL reference server (127.0.0.1:$PORT)"
cap 60G $L/llama-server -m $M --host 127.0.0.1 --port $PORT -ngl 99 --n-cpu-moe 12 -ts 34,34,24 -c 4096 \
    --no-warmup -np 1 > $W/ref-server.log 2>&1 &
SP=$!
for i in $(seq 1 120); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; sleep 5; done
echo "server ready after ~$((i*5)) s: $(curl -s -m 2 http://127.0.0.1:$PORT/health)"
grep -E "model buffer size|KV buffer|n_cpu_moe|offloaded" $W/ref-server.log | tail -8
python3 - "$PORT" "$W" <<'EOF'
import json, sys, urllib.request
port, w = sys.argv[1], sys.argv[2]
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))
text = open(f"{w}/prompt.txt").read()
ids = post("/tokenize", {"content": text, "add_special": True})["tokens"]
open(f"{w}/prompt.ids", "w").write(",".join(map(str, ids)))
res = post("/completion", {"prompt": ids, "n_predict": 16, "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                           "cache_prompt": False, "return_tokens": True, "seed": 1})
json.dump(res, open(f"{w}/ref-completion.json", "w"), indent=1)
print("prompt ids:", ids)
print("ref tokens:", res.get("tokens"))
print("ref text:", repr(res.get("content")))
print("ref timings:", res.get("timings"))
EOF
kill $SP; wait $SP 2>/dev/null; sleep 3
echo "== $(date -u +%T) reference server stopped; VRAM: $(vram)"
fi

echo "== $(date -u +%T) XeMaya load + 64 greedy tokens (no MTP drafts)"
cap 60G timeout 480 $B/build-engine/strata --native $M --tokens-file $W/prompt.ids \
    --max-new 64 --greedy --spec 0 --max-context 4096 > $W/xemaya-gen.out 2> $W/xemaya-gen.err
echo "rc=$?"; tail -8 $W/xemaya-gen.err; tail -3 $W/xemaya-gen.out
echo "== $(date -u +%T) XeMaya done; VRAM: $(vram)"

echo "== $(date -u +%T) kernel parity on the three cards (in parallel, one card each)"
for d in 0 1 2; do
  ( for t in glm5_parity ds41_parity iq_parity elementwise_parity; do
      ONEAPI_DEVICE_SELECTOR=level_zero:$d cap 12G timeout 60 $B/build-icpx/$t > $W/$t-l0-$d.log 2>&1
      echo "$t level_zero:$d rc=$? FAIL-lines=$(grep -c FAIL $W/$t-l0-$d.log) last: $(tail -1 $W/$t-l0-$d.log)"
    done ) &
done
wait
echo "== $(date -u +%T) done; VRAM: $(vram)"
