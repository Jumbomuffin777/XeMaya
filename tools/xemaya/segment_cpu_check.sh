#!/bin/bash
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# The hybrid segment link end to end on the OpenCL CPU device (no GPU is touched): the model whole in one engine, then
# its first CUT blocks in one engine and the rest in a segment worker over the loopback, forced along the same tokens;
# the two runs' logits are compared (tools/xemaya/compare_logits.py's numbers: the largest difference, the picks).
#
#   segment_cpu_check.sh ENGINE MODEL.gguf PROMPT.ids FOLLOW.ids CUT OUTDIR [engine arguments]
#
# PROMPT.ids and FOLLOW.ids: comma-separated token ids (the prompt, and the continuation both runs emit).
# Environment: SEG_MIB (the CPU device's budget for each engine, default 16000), SEG_MEMORY (each engine's cgroup
# limit, default 40G), SEG_TIMEOUT (seconds a run, default 3000), SEG_WIRE (f32 or f16, default f32),
# SEG_SKIP_WHOLE=1 (reuse OUTDIR/whole.f32).
set -u
ENGINE=$1; MODEL=$2; PROMPT=$3; FOLLOW=$4; CUT=$5; OUT=$6; shift 6
mkdir -p "$OUT"
NEW=$(tr ',' '\n' < "$FOLLOW" | grep -c .)
export ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 XEMAYA_CPU_DEVICE_MIB=${SEG_MIB:-16000} SYCL_CACHE_PERSISTENT=1
run() { systemd-run --user --scope -q -p MemoryMax="${SEG_MEMORY:-40G}" -p MemorySwapMax=0 nice -n 19 ionice -c3 \
            timeout -s TERM "${SEG_TIMEOUT:-3000}" "$@"; }
common=(--native "$MODEL" --greedy --max-context 1024 --mmap-experts --spec 0)
gen=(--tokens-file "$PROMPT" --max-new "$NEW" --spec-follow "$FOLLOW")

if [ "${SEG_SKIP_WHOLE:-0}" != 1 ] || [ ! -s "$OUT/whole.f32" ]; then
    run "$ENGINE" "${common[@]}" "${gen[@]}" --dump-logits "$OUT/whole.f32" "$@" > "$OUT/whole.out" 2> "$OUT/whole.err"
    echo "whole: rc=$?"
fi

XEMAYA_SEGMENT_TOKEN=$(head -c 24 /dev/urandom | base64)
export XEMAYA_SEGMENT_TOKEN
run "$ENGINE" "${common[@]}" --segment-first "$CUT" --segment-serve 127.0.0.1:0 "$@" > "$OUT/worker.out" 2> "$OUT/worker.err" &
WORKER=$!
PORT=
for _ in $(seq 1 "${SEG_TIMEOUT:-3000}"); do
    PORT=$(sed -n 's/.*"event":"listening","port":\([0-9]*\).*/\1/p' "$OUT/worker.out" | head -1)
    [ -n "$PORT" ] && break
    kill -0 $WORKER 2>/dev/null || break
    sleep 1
done
if [ -z "$PORT" ]; then
    echo "the worker did not start:"; tail -5 "$OUT/worker.err"; exit 1
fi
run "$ENGINE" "${common[@]}" "${gen[@]}" --segment-last "$CUT" --remote-segment "127.0.0.1:$PORT" \
    --segment-wire "${SEG_WIRE:-f32}" --dump-logits "$OUT/hybrid.f32" "$@" > "$OUT/hybrid.out" 2> "$OUT/hybrid.err"
echo "hybrid: rc=$?"
pkill -TERM -P $WORKER 2>/dev/null; kill -TERM $WORKER 2>/dev/null; wait $WORKER 2>/dev/null
python3 - "$OUT/whole.f32" "$OUT/hybrid.f32" "$NEW" <<'PY'
import array
import sys


def rows(path, n):
    a = array.array("f")
    with open(path, "rb") as f:
        a.frombytes(f.read())
    if len(a) == 0 or len(a) % n:
        sys.exit(f"FAIL: {path} holds {len(a)} floats, not {n} rows of logits")
    w = len(a) // n
    return [a[i * w:(i + 1) * w] for i in range(n)]


n = int(sys.argv[3])
a, b = rows(sys.argv[1], n), rows(sys.argv[2], n)
if len(a[0]) != len(b[0]):
    sys.exit("FAIL: the two runs' vocabularies differ")
same = sum(max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__) for x, y in zip(a, b))
worst = max(abs(p - q) for x, y in zip(a, b) for p, q in zip(x, y))
print(f"top-1 equal at {same} of {n}; largest |difference| {worst:.3e}; bit for bit: {'yes' if a == b else 'no'}")
sys.exit(0 if same == n else 1)
PY
