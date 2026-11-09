#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The report of a GPU window run by window3.sh: every engine_serve.py request (prompt length, reused tokens, TTFT,
prefill and decode speeds, drafts, expert hits) with the telemetry of its prompt and decode phases (GPU compute % and
VRAM per card from xpu-smi, host MemAvailable and page cache, NVMe reads), the agent sessions, and the kernel tests.

    window_report.py LOGDIR
"""
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

d = sys.argv[1]
CARDS = ((0, "B70"), (1, "B65"), (2, "B60"))


def xpu(dev):
    rows = []
    for path in [f"{d}/xpu-{dev}.csv"]:
        if not os.path.exists(path):
            continue
        day = datetime.fromtimestamp(os.path.getmtime(path), timezone.utc).strftime("%Y-%m-%d")
        for line in open(path):
            f = [x.strip() for x in line.split(",")]
            if len(f) < 13 or not re.match(r"\d\d:\d\d:\d\d", f[0]):
                continue
            ts = datetime.strptime(f"{day} {f[0]}", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            num = lambda v: float(v) if v not in ("N/A", "") else None
            rows.append((ts, num(f[3]), num(f[12])))
    return rows


XPU = {dev: xpu(dev) for dev, _ in CARDS}
HOST = [tuple(float(v) for v in l.split()) for l in open(f"{d}/host.txt")] if os.path.exists(f"{d}/host.txt") else []


def tele(a, b):
    out = []
    for dev, nm in CARDS:
        r = [x for x in XPU[dev] if a <= x[0] <= b]
        cu = [x[1] for x in r if x[1] is not None]
        mu = [x[2] for x in r if x[2] is not None]
        if cu or mu:
            out.append(f"{nm} {sum(cu) / len(cu) if cu else float('nan'):.0f}% {max(mu) / 1024 if mu else float('nan'):.1f}G")
    h = [x for x in HOST if a <= x[0] <= b and len(x) == 4]
    if len(h) >= 2:
        nv = (h[-1][3] - h[0][3]) * 512 / 1e9
        out.append(f"RAM avail>={min(x[1] for x in h) / 1024:.0f}G NVMe {nv:.1f} GB ({nv / max(b - a, 1e-9):.2f} GB/s)")
    return " | ".join(out) if out else "no telemetry"


def line(name, j):
    pt = j["prompt"] - j["reused"]
    pre = pt / (j["prompt_ms"] / 1e3) if j["prompt_ms"] > 0 else 0
    dec = (j["generated"] - 1) / (j["decode_ms"] / 1e3) if j["decode_ms"] > 0 and j["generated"] > 1 else 0
    hit = 100.0 * j["hits"] / j["lookups"] if j["lookups"] else 0
    s = (f"  {name:<26} prompt {j['prompt']:>6} reused {j['reused']:>6} | TTFT {j['ttft_s']:7.2f} s | prefill "
         f"{pre:7.1f} tok/s | decode {dec:6.2f} tok/s ({j['generated']}) | drafts {j['accepted']}/{j['offered']} | "
         f"hit {hit:4.1f}%")
    t0, t1 = j["start"], j["start"] + (j["ttft_s"] or 0)
    s += f"\n      prompt: {tele(t0, t1)}\n      decode: {tele(t1, j['end'])}"
    if j.get("compare"):
        last = j["compare"]["report"].splitlines()
        s += "\n      vs llama.cpp: " + " / ".join(l for l in last if l.startswith(("free run:", "PASS", "FAIL")))
    return s


for path in sorted(glob.glob(f"{d}/*.json")):
    name = os.path.basename(path)[:-5]
    if name.startswith(("plan-", "server-", "agent-")) or name.endswith(".engine"):
        continue
    try:
        r = json.load(open(path))
    except ValueError:
        continue
    if "jobs" not in r:
        continue
    print(f"{name}: load {r.get('load_s', 0):.1f} s, context {r.get('context')}, env {r.get('env')}")
    for j in r["jobs"]:
        if j.get("kind") == "checkpoint":
            print(f"  {j['name']}: B resumed at {j['B_resumed']['reused']} ({'ok' if j['resume_ok'] else 'NOT OK'}), "
                  f"tokens {'identical' if j['identical'] else 'differ at ' + str(j['first_difference'])} to the scratch run")
            for k in ("A", "B_resumed", "B_scratch"):
                print(line(f"{j['name']} {k}", j[k]))
        else:
            print(line(j["name"], j))
    log = path + ".engine.log"
    if os.path.exists(log):
        for l in open(log):
            if "GLM prompt of" in l or "prompt checkpoint" in l:
                print("    " + l.strip()[:400])
for path in sorted(glob.glob(f"{d}/agent-*.json")):
    r = json.load(open(path))
    print(f"{os.path.basename(path)}: {len(r['turns'])} turns, problems {r['problems'] or 'none'}")
    for t in r["turns"]:
        print(f"  turn {t['turn']}: prompt {t['prompt_tokens']} cached {t['cached_tokens']} TTFT {t['ttft_s']:.2f} s "
              f"turn {t['total_s']:.1f} s prefill {t['prompt_per_second']} decode {t['predicted_per_second']} "
              f"finish {t['finish']} calls {[c['name'] for c in t['calls']]}")
    print(f"  warm repeat {r['warm']}, cancel {r['cancel']}, anthropic {r['anthropic']}")
for path in sorted(glob.glob(f"{d}/bench-mla-*.txt")):
    print(os.path.basename(path)); print(open(path).read().rstrip())
for path in sorted(glob.glob(f"{d}/*_parity-l0-*.log")):
    with open(path) as f:
        last = f.read().strip().splitlines()[-1:]
    print(f"{os.path.basename(path)}: {last[0] if last else 'empty'}")
