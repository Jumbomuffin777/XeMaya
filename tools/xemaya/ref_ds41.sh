#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# llama.cpp references for DeepSeek-V4.1-Flash on the CPU only (no GPU is opened):
#   ref_ds41.sh MODEL_SHARD1 TAG N_PREDICT NAME=TEXTFILE [NAME=TEXTFILE ...]
# llama.cpp v0.8.0 (8642541b2) built with GGML_SYCL=OFF, one server load for every prompt, greedy, each step's top-5
# log-probabilities. Per prompt: /DATA/Projects/xemaya/prompts/ds41/TAG-NAME.ids (the tokens, BOS included) and
# xemaya/reference/ds41-TAG-NAME.json (compare_ref.py's format). nice 19, 8 threads, a 40 GB cap and a watchdog that
# stops the server when the host's MemAvailable falls under 14 GiB.
set -u
M=$1; TAG=$2; NP=$3; shift 3
X=/DATA/Projects/xemaya
OUT=$X/XeMaya/xemaya/reference
S=$X/ref/llama.cpp-v0.8.0/build-cpu/bin/llama-server
PORT=${PORT:-18996}
mkdir -p "$OUT" $X/prompts/ds41 $X/logs
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 2; }
systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 nice -n 19 ionice -c3 \
    "$S" -m "$M" --host 127.0.0.1 --port $PORT -t 8 -tb 8 -c 2048 -np 1 --no-warmup > "$X/logs/ref-ds41-$TAG.server.log" 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do
    a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    [ "$a" -lt 14 ] && { echo "WATCHDOG: MemAvailable ${a} GiB, stopping"; kill -TERM $SP; }
    sleep 1; done ) &
for i in $(seq 1 360); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
kill -0 $SP 2>/dev/null || { echo "the server exited (see $X/logs/ref-ds41-$TAG.server.log)"; exit 1; }
for spec in "$@"; do
  NAME=${spec%%=*}; TXT=${spec#*=}
  python3 - "$PORT" "$TXT" "$NP" "$OUT/ds41-$TAG-$NAME.json" "$TAG-$NAME" "$X/prompts/ds41/$TAG-$NAME.ids" "$M" <<'EOF'
import json, os, sys, time, urllib.request
port, txt, n_predict, out, name, ids_out, model = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
def post(path, body, timeout=6 * 3600):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Content-Type": "application/json"},
                               data=json.dumps(body).encode())
    return json.load(urllib.request.urlopen(r, timeout=timeout))
ids = post("/tokenize", {"content": open(txt).read(), "add_special": True, "parse_special": True})["tokens"]
open(ids_out, "w").write(",".join(map(str, ids)))
t0 = time.time()
res = post("/completion", {"prompt": ids, "n_predict": n_predict, "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                           "cache_prompt": False, "return_tokens": True, "n_probs": 5, "seed": 1})
dt = time.time() - t0
steps = [{"token": p.get("id"), "text": p.get("token"),
          "top5": [{"id": c.get("id"), "text": c.get("token"), "logprob": c.get("logprob")} for c in p.get("top_logprobs", [])]}
         for p in res.get("completion_probabilities", [])]
json.dump({"name": name, "model": os.path.basename(model),
           "reference": "llama.cpp v0.8.0 (8642541b2), CPU backend only (GGML_SYCL=OFF), greedy (top_k 1, temperature 0)",
           "prompt_ids": ids, "tokens": res.get("tokens"), "text": res.get("content"), "steps": steps,
           "timings": res.get("timings"), "wall_s": round(dt, 1)}, open(out, "w"), indent=1, ensure_ascii=False)
print(f"{name}: {len(ids)} prompt tokens, {len(res.get('tokens') or [])} generated in {dt:.0f} s: {res.get('content')!r:.160}", flush=True)
EOF
done
kill -TERM $SP 2>/dev/null; for i in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; sleep 1; done
echo "$(date -u +%T) ref_ds41 $TAG done"
