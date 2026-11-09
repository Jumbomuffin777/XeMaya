#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A Markdown table of qwen_bench.py results: qwen_report.py RESULT.json [RESULT.json ...] (the order given).
With --slim OUT_DIR, also writes each result without its token lists (the record kept in the repository)."""
import json
import os
import sys


def rate(n, ms):
    return n / (ms / 1000) if ms > 0 else 0


def job(r, name):
    for j in r["jobs"]:
        if j["name"] == name:
            return j
    return None


def cell(j, what):
    if j is None:
        return "-"
    if what == "prefill":
        return f"{rate(j['prompt'] - j['reused'], j['prompt_ms']):.0f}"
    if what == "ttft":
        return f"{j['ttft_s']:.2f}"
    if what == "decode":
        return f"{rate(j['generated'] - 1, j['decode_ms']):.1f}"
    if what == "accept":
        return f"{100 * j['accepted'] / j['offered']:.0f}%" if j["offered"] else "-"
    if what == "reused":
        return f"{j['reused']}"
    return "?"


def main():
    args = sys.argv[1:]
    slim = None
    if "--slim" in args:
        i = args.index("--slim")
        slim = args[i + 1]
        del args[i:i + 2]
        os.makedirs(slim, exist_ok=True)
    print("| Run | Load s | 26.8K prefill tok/s (code / prose prompt) | TTFT s | Code decode tok/s (drafts kept) | "
          "Prose decode tok/s (drafts kept) | Short prompt: code / prose decode | Follow-up turn: TTFT s, decode | "
          "Prefix pair: reused, TTFT s | VRAM MiB B70 / B65 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for path in args:
        r = json.load(open(path))
        a, b = job(r, "long-a-code"), job(r, "long-b-prose")
        sc, sp = job(r, "short-code"), job(r, "short-prose again")
        t2, t2b = job(r, "long-a-code+turn2"), job(r, "long-b-prose+turn2")
        px = job(r, "long-b-code (prefix of b)")
        vr = r.get("vram_loaded")
        vram = " / ".join(str(d["used_mib"]) for d in vr if "B60" not in d["name"]) if isinstance(vr, list) else "-"
        print(f"| {r['label']} | {r.get('load_s', 0):.0f} | {cell(a, 'prefill')} / {cell(b, 'prefill')} | "
              f"{cell(a, 'ttft')} / {cell(b, 'ttft')} | {cell(a, 'decode')} ({cell(a, 'accept')}) | "
              f"{cell(b, 'decode')} ({cell(b, 'accept')}) | {cell(sc, 'decode')} / {cell(sp, 'decode')} | "
              f"{cell(t2, 'ttft')}, {cell(t2, 'decode')}; {cell(t2b, 'ttft')}, {cell(t2b, 'decode')} | "
              f"{cell(px, 'reused')}, {cell(px, 'ttft')} | {vram} |")
        if slim:
            for j in r["jobs"]:
                j["n_tokens"] = len(j.pop("tokens", []))
            json.dump(r, open(os.path.join(slim, os.path.basename(path)), "w"), indent=1)


if __name__ == "__main__":
    main()
