#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-phase telemetry of one benchmark run (tools/xemaya/bench_run.sh + the window's monitor):
GPU compute activity and VRAM per card (xpu-smi dump), host MemAvailable / page cache and NVMe reads (/proc).

    phase_stats.py LOGDIR NAME     (reads bench-NAME.meta/.err, xpu-{0,1,2}.csv, host.txt)
"""
import re
import sys
from datetime import datetime, timezone

d, name = sys.argv[1], sys.argv[2]
meta = open(f"{d}/bench-{name}.meta").read()
t0 = float(re.search(r"start=([\d.]+)", meta).group(1))
t1 = float(re.search(r"end=([\d.]+)", meta).group(1))
err = open(f"{d}/bench-{name}.err").read()
m = re.search(r"load ([\d.]+) s, prompt (\d+) tokens ([\d.]+) s .*?, (\d+) generated ([\d.]+) s", err)
load, ptoks, pt, gtoks, gt = float(m[1]), int(m[2]), float(m[3]), int(m[4]), float(m[5])
# the engine's clock starts after process start; place the phases from the run's end backwards
end = t1 - 0.5
phases = {"decode": (end - gt, end), "prompt": (end - gt - pt, end - gt), "load": (end - gt - pt - load, end - gt - pt)}

day = datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%d")
def xpu(dev):
    rows = []
    for line in open(f"{d}/xpu-{dev}.csv"):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 13 or not re.match(r"\d\d:\d\d:\d\d", f[0]):
            continue
        ts = datetime.strptime(f"{day} {f[0]}", "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        num = lambda v: float(v) if v not in ("N/A", "") else None
        rows.append((ts, num(f[3]), num(f[12])))   # utilization.compute %, memory.used MiB
    return rows
host = [tuple(float(v) for v in l.split()) for l in open(f"{d}/host.txt") if len(l.split()) == 4]

print(f"{name}: load {load:.1f} s, prompt {ptoks} tok {pt:.2f} s ({ptoks / pt:.1f} tok/s), decode {gtoks} tok {gt:.2f} s ({gtoks / gt:.2f} tok/s)")
for ph, (a, b) in phases.items():
    out = [f"{ph:7s} {b - a:6.1f} s"]
    for dev, nm in ((0, "B70"), (1, "B65"), (2, "B60")):
        r = [x for x in xpu(dev) if a <= x[0] <= b]
        cu = [x[1] for x in r if x[1] is not None]
        mu = [x[2] for x in r if x[2] is not None]
        out.append(f"{nm} compute {sum(cu) / len(cu) if cu else float('nan'):5.1f}% VRAM {max(mu) / 1024 if mu else float('nan'):5.1f} GiB")
    h = [x for x in host if a <= x[0] <= b]
    if len(h) >= 2:
        nv = (h[-1][3] - h[0][3]) * 512 / 1e9
        out.append(f"RAM avail min {min(x[1] for x in h) / 1024:.1f} GiB, page cache {h[-1][2] / 1024:.1f} GiB, NVMe read {nv:.2f} GB ({nv / (b - a):.2f} GB/s)")
    print("  " + " | ".join(out))
