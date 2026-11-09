#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runs a plan of requests against one loaded plain-model engine (`strata --native GGUF --serve`), the protocol
server.py speaks: GEN <max_new> <ids> -> RESUME <reused>, T <id> per token, DONE <fields>. One model load serves every
request, so a GPU window measures many prompts without reloading. Run inside the GPU lease.

    engine_serve.py --exe STRATA --model GGUF --out RESULTS.json [--engine-arg=ARG ...] PLAN.json

The plan: {"jobs": [...]}, each job one of
  {"kind": "gen", "name": N, "prompt": IDS_FILE, "max_new": K, "ref": REF_JSON (optional, compared by compare_ref.py)}
  {"kind": "checkpoint", "name": N, "first": IDS_FILE, "then": IDS_FILE, "max_new": K, "reset": IDS_FILE}
      the prompt-end checkpoint: request A = first; request B = first + A's reply with one token changed (as a client
      re-renders a reply) + then. B should resume at len(first). Then `reset` (another conversation) and B again,
      read from scratch: its tokens are compared with the resumed B's.
Per request it records the client's time to the first token and the engine's DONE fields (prompt and decode ms,
reused and read tokens, expert hits, RAM and file reads), with wall-clock start and end for the telemetry.
"""
import argparse
import json
import os
import subprocess
import sys
import time

DONE_FIELDS = ["generated", "prompt", "prompt_ms", "decode_ms", "finish", "accepted", "offered", "reused", "hits",
               "lookups", "ram_blobs", "file_blobs", "file_mb", "read", "offloaded"]


def ids_of(path):
    return [int(v) for v in open(path).read().replace("\n", ",").split(",") if v.strip()]


class Engine:
    def __init__(self, exe, model, extra, log):
        self.log = open(log, "w")
        self.p = subprocess.Popen([exe, "--native", model, "--serve", *extra], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        t0 = time.time()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError(f"the engine exited before READY (see {log})")
            if line.startswith("READY"):
                self.context = int(line.split()[1])
                self.load_s = time.time() - t0
                return

    def gen(self, ids, max_new):
        t0 = time.time()
        self.p.stdin.write(f"GEN {max_new} {','.join(map(str, ids))}\n")
        self.p.stdin.flush()
        out, first, resume = [], None, None
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("the engine exited during a request")
            w = line.split()
            if not w:
                continue
            if w[0] == "RESUME":
                resume = int(w[1])
            elif w[0] == "T":
                if first is None:
                    first = time.time() - t0
                out.append(int(w[1]))
            elif w[0] == "DONE":
                d = {}
                for k, v in zip(DONE_FIELDS, w[1:]):
                    d[k] = v if k == "finish" else (float(v) if "." in v else int(v))
                d.update(tokens=out, ttft_s=first, wall_s=time.time() - t0, start=t0, end=time.time(),
                         resume_line=resume)
                return d
            elif w[0] in ("ERR", "FATAL"):
                raise RuntimeError(line.strip())

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=120)
        except Exception:
            self.p.kill()
        self.log.close()


def summary(name, d):
    pt = d["prompt"] - d["reused"]
    pre = pt / (d["prompt_ms"] / 1000) if d["prompt_ms"] > 0 else 0
    dec = (d["generated"] - 1) / (d["decode_ms"] / 1000) if d["decode_ms"] > 0 and d["generated"] > 1 else 0
    hit = 100.0 * d["hits"] / d["lookups"] if d["lookups"] else 0
    return (f"{name}: prompt {d['prompt']} (reused {d['reused']}), read {pt} in {d['prompt_ms'] / 1000:.2f} s "
            f"({pre:.1f} tok/s), TTFT {d['ttft_s']:.2f} s, decode {d['generated']} in {d['decode_ms'] / 1000:.2f} s "
            f"({dec:.2f} tok/s), drafts {d['accepted']}/{d['offered']}, experts hit {hit:.1f}%, file reads "
            f"{d['file_mb']:.0f} MB")


def compare(ref_path, tokens, here):
    tmp = os.path.join(here, f".free-{os.getpid()}.out")
    open(tmp, "w").write("tokens: " + ",".join(map(str, tokens)) + "\n")
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_ref.py"),
                        ref_path, "--free", tmp], capture_output=True, text=True)
    os.unlink(tmp)
    return r.returncode == 0, r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("plan")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    here = os.path.dirname(os.path.abspath(a.out))
    res = {"engine_args": a.engine_arg, "env": {k: v for k, v in os.environ.items() if k.startswith(("XEMAYA_", "STRATA_"))},
           "jobs": []}
    eng = Engine(a.exe, a.model, a.engine_arg, a.out + ".engine.log")
    res["load_s"] = eng.load_s
    res["context"] = eng.context
    print(f"engine ready in {eng.load_s:.1f} s, context {eng.context}", flush=True)
    bad = 0
    try:
        for job in plan["jobs"]:
            if job["kind"] == "gen":
                ids = ids_of(job["prompt"])
                if len(ids) + job["max_new"] > eng.context:
                    print(f"{job['name']}: skipped, {len(ids)} + {job['max_new']} past the context {eng.context}", flush=True)
                    continue
                d = eng.gen(ids, job["max_new"])
                d["name"] = job["name"]
                print(summary(job["name"], d), flush=True)
                if job.get("ref"):
                    ok, txt = compare(job["ref"], d["tokens"], here)
                    d["compare"] = {"ok": ok, "report": txt}
                    print("  " + txt.replace("\n", "\n  "), flush=True)
                    bad += not ok
                res["jobs"].append(d)
            elif job["kind"] == "checkpoint":
                first = ids_of(job["first"])
                a_ = eng.gen(first, job["max_new"])
                reply = list(a_["tokens"])
                if len(reply) > 2:
                    reply[len(reply) // 2] = 220 if reply[len(reply) // 2] != 220 else 198   # a re-rendered reply
                b_ids = first + reply + ids_of(job["then"])
                b1 = eng.gen(b_ids, job["max_new"])
                eng.gen(ids_of(job["reset"]), 4)                     # another conversation in between
                b2 = eng.gen(b_ids, job["max_new"])                  # the same request read from scratch
                same = b1["tokens"] == b2["tokens"]
                k = next((i for i in range(min(len(b1["tokens"]), len(b2["tokens"])))
                          if b1["tokens"][i] != b2["tokens"][i]), None)
                ok = b1["reused"] == len(first) and b2["reused"] == 0
                print(summary(job["name"] + " A (first turn)", a_), flush=True)
                print(summary(job["name"] + " B resumed", b1), flush=True)
                print(summary(job["name"] + " B from scratch", b2), flush=True)
                print(f"  checkpoint: B resumed at {b1['reused']} (want {len(first)}), the scratch run at {b2['reused']} "
                      f"(want 0); tokens {'identical' if same else f'first differ at {k}'}", flush=True)
                bad += not ok
                res["jobs"].append({"name": job["name"], "kind": "checkpoint", "A": a_, "B_resumed": b1, "B_scratch": b2,
                                    "resume_ok": ok, "identical": same, "first_difference": k})
    finally:
        eng.close()
        json.dump(res, open(a.out, "w"), indent=1)
    print("PASS" if not bad else f"{bad} FAILED", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
