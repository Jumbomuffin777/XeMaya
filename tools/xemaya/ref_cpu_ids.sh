#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# A llama.cpp reference for GLM-5.3-Flash Maya-S v2 on the CPU only (no GPU touched; production keeps its cards):
#   ref_cpu_ids.sh NAME PROMPT_IDS N_PREDICT
# llama.cpp v0.8.0 (8642541b2) built with GGML_SYCL=OFF, greedy, each step's top-5 log-probabilities, written to
# xemaya/reference/glm53-maya-s-v2-NAME.json (compare_ref.py's format). nice 19, 6 threads, a 40 GB cap and a
# watchdog that stops it if the host's MemAvailable falls under 14 GiB.
set -u
NAME=$1; IDS=$2; NP=$3
X=${XEMAYA_WORK:?the work folder: LOCAL.md}
OUT=$X/XeMaya/xemaya/reference
LOG=$X/logs/ref-$NAME
M=${XEMAYA_MODELS:?the model folder: LOCAL.md}/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
S=$X/ref/llama.cpp-v0.8.0/build-cpu/bin/llama-server
PORT=${PORT:-18996}
NT=$(tr ',' '\n' < "$IDS" | grep -c .)
CTX=$(( (NT + NP + 64 + 255) / 256 * 256 ))
mkdir -p "$OUT"
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 2; }
systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 nice -n 19 ionice -c3 \
    "$S" -m "$M" --host 127.0.0.1 --port $PORT -t 6 -tb 6 -c $CTX -np 1 --no-warmup > "$LOG.server.log" 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do
    a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$a" -lt 14 ] && { echo "WATCHDOG: MemAvailable ${a} GiB, stopping"; kill -9 $SP; }
    sleep 1; done ) &
for i in $(seq 1 360); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
echo "$(date -u +%T) $NAME: $NT prompt tokens, $NP to generate, context $CTX"
python3 - "$PORT" "$IDS" "$NP" "$OUT/glm53-maya-s-v2-$NAME.json" "$NAME" <<'EOF'
import json, sys, time, urllib.request
port, ids_path, n_predict, out, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
ids = [int(v) for v in open(ids_path).read().split(",") if v.strip()]
r = urllib.request.Request(f"http://127.0.0.1:{port}/completion", headers={"Content-Type": "application/json"},
    data=json.dumps({"prompt": ids, "n_predict": n_predict, "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                     "cache_prompt": False, "return_tokens": True, "n_probs": 5, "seed": 1}).encode())
t0 = time.time(); res = json.load(urllib.request.urlopen(r, timeout=6 * 3600)); dt = time.time() - t0
steps = [{"token": p.get("id"), "text": p.get("token"),
          "top5": [{"id": c.get("id"), "text": c.get("token"), "logprob": c.get("logprob")} for c in p.get("top_logprobs", [])]}
         for p in res.get("completion_probabilities", [])]
json.dump({"name": name, "model": "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS (peasantsmith/GLM-5.3-Flash-Maya-GGUF a1c2a554, sha256-checked)",
           "reference": "llama.cpp v0.8.0 (8642541b2), CPU backend only (GGML_SYCL=OFF), greedy (top_k 1, temperature 0)",
           "prompt_ids": ids, "tokens": res.get("tokens"), "text": res.get("content"), "steps": steps,
           "timings": res.get("timings"), "wall_s": round(dt, 1)}, open(out, "w"), indent=1, ensure_ascii=False)
ties = sum(1 for s in steps if len(s["top5"]) > 1 and s["top5"][0]["logprob"] - s["top5"][1]["logprob"] < 0.1)
print(f"{name}: {len(res.get('tokens') or [])} tokens in {dt:.0f} s, {ties} near-ties under 0.1: {res.get('content')!r:.120}")
EOF
echo "rc=$?"
kill $SP 2>/dev/null; sleep 3; kill -9 $SP 2>/dev/null; wait 2>/dev/null
echo "$(date -u +%T) $NAME done"
