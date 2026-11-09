#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""One engine binary (upstream Strata, XeStrata or XeMaya) on Qwen3.8 Flash-Next with a serving profile's arguments,
driven over the `--serve` protocol with the same token ids (make_qwen_prompts.py): short and ~26.8K-token prompts,
prose and code, an agent's follow-up turn, a prefix-reuse pair and repeats.  Run inside the GPU lease and
safety_guard.sh.

    qwen_bench.py --exe STRATA --profile PROFILE.json --prompts DIR --out RESULT.json --label NAME
                  [--spec N] [--max-context N] [--libdir DIR] [--cwd DIR] [--drop FLAG ...] [--set FLAG=VALUE ...]

The profile's "args" and "env" are used as they are except: --vision is dropped, --spec and --max-context are set.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_serve import Engine, ids_of, summary  # noqa: E402

VALUED_DROP = ()


def set_arg(args, flag, value):
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]


def vram():
    try:
        r = subprocess.run([sys.executable, "/DATA/AppData/zability-gpu-lease/gpulease.py", "status"],
                           capture_output=True, text=True, timeout=20)
        return [{"name": d["name"], "used_mib": d["used"] // 2**20} for d in json.loads(r.stdout)["telemetry"]["devices"]]
    except Exception as e:  # telemetry only
        return str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--spec", default="4")
    ap.add_argument("--max-context", default="65536")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--libdir")
    ap.add_argument("--cwd")
    ap.add_argument("--drop", action="append", default=[], help="a bare flag, or FLAG+ for a flag with a value")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--add", action="append", default=[], help="a bare flag to add")
    a = ap.parse_args()
    prof = json.load(open(a.profile))
    args = [x for x in prof["args"] if x != "--vision"]
    for d in a.drop:
        if d.endswith("+") and d[:-1] in args:
            i = args.index(d[:-1])
            del args[i:i + 2]
        elif d in args:
            args.remove(d)
    set_arg(args, "--spec", a.spec)
    set_arg(args, "--max-context", a.max_context)
    for s in a.set:
        k, v = s.split("=", 1)
        set_arg(args, k, v)
    args += [f for f in a.add if f not in args]
    model = args[args.index("--native") + 1]
    i = args.index("--native")
    extra = args[:i] + args[i + 2:]
    os.environ.update(prof.get("env", {}))
    if a.libdir:
        os.environ["LD_LIBRARY_PATH"] = a.libdir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    if a.cwd:
        os.chdir(a.cwd)
    P = lambda n: ids_of(os.path.join(a.prompts, n + ".ids"))  # noqa: E731
    res = {"label": a.label, "exe": a.exe, "args": ["--native", model, "--serve", *extra], "env": prof.get("env", {}),
           "vram_before": vram(), "jobs": []}
    eng = Engine(a.exe, model, extra, a.out + ".engine.log")
    res.update(load_s=eng.load_s, context=eng.context, vram_loaded=vram())
    print(f"[{a.label}] ready in {eng.load_s:.1f} s, context {eng.context}, VRAM {res['vram_loaded']}", flush=True)

    def run(name, ids, max_new=a.max_new):
        d = eng.gen(ids, max_new)
        d["name"] = name
        res["jobs"].append(d)
        print(f"[{a.label}] " + summary(name, d), flush=True)
        json.dump(res, open(a.out, "w"))
        return d
    try:
        run("short-prose", P("short-prose"))
        run("short-code", P("short-code"))
        d = run("long-a-code", P("long-a-code"))
        run("long-a-code+turn2", P("long-a-code") + d["tokens"] + P("turn2"))
        d = run("long-b-prose", P("long-b-prose"))
        run("long-b-prose+turn2", P("long-b-prose") + d["tokens"] + P("turn2"))
        run("long-b-code (prefix of b)", P("long-b-code"))
        run("long-a-code again", P("long-a-code"), 64)
        run("short-prose again", P("short-prose"))
        res["vram_end"] = vram()
    finally:
        eng.close()
        json.dump(res, open(a.out, "w"))
    print(f"[{a.label}] done", flush=True)


if __name__ == "__main__":
    main()
