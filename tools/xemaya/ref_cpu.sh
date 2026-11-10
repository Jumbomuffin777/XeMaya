#!/bin/bash
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
# The llama.cpp reference for the GPU window, on the CPU only (no GPU is touched; production keeps its cards):
# llama.cpp v0.8.0 (8642541b2) built with GGML_SYCL=OFF, the saved prompt ids, 16 greedy tokens with the top-5
# probabilities of each step. Low priority, capped, and a watchdog stops it if the host runs short of memory.
set -u
X=${XEMAYA_WORK:?the work folder: LOCAL.md}
W=$X/logs/window
OUT=$X/XeMaya/xemaya/reference
M=${XEMAYA_MODELS:?the model folder: LOCAL.md}/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf
S=$X/ref/llama.cpp-v0.8.0/build-cpu/bin/llama-server
PORT=18996
mkdir -p "$OUT"
ss -ltn | grep -q ":$PORT " && { echo "port $PORT busy"; exit 2; }
systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 nice -n 19 ionice -c3 \
    "$S" -m "$M" --host 127.0.0.1 --port $PORT -t 8 -tb 8 -c 512 -np 1 --no-warmup > "$W/ref-cpu-server.log" 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do
    a=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
    if [ "$a" -lt 14 ]; then echo "WATCHDOG: MemAvailable ${a} GiB, stopping the reference"; pkill -9 -f "llama.cpp-v0.8.0/build-cpu/bin/llama-server"; fi
    sleep 1; done ) &
for i in $(seq 1 360); do curl -s -m 2 http://127.0.0.1:$PORT/health | grep -q ok && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
echo "$(date -u +%T) server: $(curl -s -m 2 http://127.0.0.1:$PORT/health)"
python3 - "$PORT" "$W" "$OUT" <<'EOF'
import json, sys, time, urllib.request
port, w, out = sys.argv[1:4]
def post(path, body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=7200))
text = open(f"{w}/prompt.txt").read()
saved = [int(v) for v in open(f"{w}/prompt.ids").read().split(",")]
ids = post("/tokenize", {"content": text, "add_special": True})["tokens"]
assert ids == saved, (ids, saved)
t0 = time.time()
res = post("/completion", {"prompt": ids, "n_predict": 16, "temperature": 0, "top_k": 1, "samplers": ["top_k"],
                           "cache_prompt": False, "return_tokens": True, "n_probs": 5, "seed": 1})
dt = time.time() - t0
probs = [{"token": p.get("id"), "text": p.get("token"),
          "top5": [{"id": c.get("id"), "text": c.get("token"), "logprob": c.get("logprob")} for c in p.get("top_logprobs", [])]}
         for p in res.get("completion_probabilities", [])]
rec = {"model": "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS (peasantsmith/GLM-5.3-Flash-Maya-GGUF a1c2a554, sha256-checked)",
       "reference": "llama.cpp v0.8.0 (8642541b2), CPU backend only (GGML_SYCL=OFF), greedy (top_k 1, temperature 0)",
       "prompt_text": text, "prompt_ids": ids, "tokens": res.get("tokens"), "text": res.get("content"),
       "steps": probs, "timings": res.get("timings"), "wall_s": round(dt, 1)}
json.dump(rec, open(f"{out}/glm53-maya-s-v2-greedy16.json", "w"), indent=1, ensure_ascii=False)
print("tokens:", rec["tokens"]); print("text:", repr(rec["text"])); print("wall:", rec["wall_s"], "s")
for k, p in enumerate(probs):
    t = p["top5"]
    gap = (t[0]["logprob"] - t[1]["logprob"]) if len(t) > 1 and t[0]["logprob"] is not None else None
    print(k, p["token"], repr(p["text"]), "margin(logprob top1-top2):", None if gap is None else round(gap, 3))
EOF
echo "rc=$?"
kill $SP 2>/dev/null; sleep 3; kill -9 $SP 2>/dev/null; wait 2>/dev/null
echo "$(date -u +%T) done"
