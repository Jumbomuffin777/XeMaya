#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compares XeMaya's GLM output with the saved llama.cpp reference (xemaya/reference/*.json, tools/xemaya/ref_cpu.sh).

    compare_ref.py REF.json [--forced LOGITS.f32] [--free OUT.txt]

--forced: the logits XeMaya wrote with `--spec-follow <reference tokens> --dump-logits` (one FP32 row of the vocabulary
per step, the prediction after the prompt and the reference's first i tokens). Each step's top-1 is checked against
the reference token, and the reference's top-5 log-probabilities against XeMaya's.
--free: the stdout of a free greedy run (its `tokens:` line); the first divergence from the reference is reported.

PASS: at every step where the reference's top-1 leads its top-2 by at least MARGIN (0.1 in log-probability), XeMaya's
forced top-1 is the reference token, and the median |difference| of the reference token's log-probability is at most
0.3. A step under the margin is a near-tie: a CPU/GPU rounding difference may flip it without a bug, so it is reported
but not failed. The free run must match up to its first near-tie step.
"""
import json
import math
import sys
from array import array

MARGIN, MEDIAN_TOL = 0.1, 0.3


def log_softmax_at(row, ids):
    m = max(row)
    lse = m + math.log(sum(math.exp(v - m) for v in row))
    return [row[i] - lse for i in ids]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    ref = json.load(open(argv[1]))
    toks, steps = ref["tokens"], ref["steps"]
    opts = dict(zip(argv[2::2], argv[3::2]))
    margins = []
    for s in steps:
        t = s["top5"]
        margins.append(t[0]["logprob"] - t[1]["logprob"] if len(t) > 1 else float("inf"))
    ok = True
    if "--forced" in opts:
        raw = array("f")
        with open(opts["--forced"], "rb") as f:
            raw.frombytes(f.read())
        n = len(toks)
        if len(raw) % n:
            print(f"forced: {len(raw)} floats is not {n} rows")
            return 1
        V = len(raw) // n
        diffs, misses, hits = [], [], 0
        print(f"forced run: {n} steps, vocabulary {V}")
        print(" step  ref token       margin  XeMaya top1     ref lp   XeMaya lp")
        for i in range(n):
            row = raw[i * V:(i + 1) * V]
            top = max(range(V), key=row.__getitem__)
            ids = [c["id"] for c in steps[i]["top5"]]
            lps = log_softmax_at(row, ids)
            d = lps[0] - steps[i]["top5"][0]["logprob"]
            diffs.append(abs(d))
            tie = margins[i] < MARGIN
            hit = top == toks[i]
            hits += hit
            if not hit and not tie:
                misses.append(i)
            print(f" {i:4d}  {toks[i]:6d} {steps[i]['text']!r:12.12s} {margins[i]:6.3f}  {top:6d} {'ok' if hit else ('near-tie' if tie else 'MISS'):8s}"
                  f" {steps[i]['top5'][0]['logprob']:8.3f}  {lps[0]:8.3f}")
        med = sorted(diffs)[len(diffs) // 2]
        print(f"forced: top-1 equals the reference at {hits} of {n} steps; misses outside near-ties {misses}; "
              f"median |d logprob| {med:.3f}, max {max(diffs):.3f}")
        ok = ok and not misses and med <= MEDIAN_TOL
    if "--free" in opts:
        line = next((l for l in open(opts["--free"]) if l.startswith("tokens:")), None)
        if line is None:
            print("free: no tokens line")
            return 1
        got = [int(v) for v in line.split(":", 1)[1].strip().split(",") if v]
        n = min(len(got), len(toks))
        div = next((i for i in range(n) if got[i] != toks[i]), None)
        first_tie = next((i for i in range(len(toks)) if margins[i] < MARGIN), len(toks))
        print(f"free run: first {n} tokens; " + ("identical to the reference" if div is None else
              f"first divergence at step {div} (reference near-tie from step {first_tie})"))
        print(f"free run tokens: {got[:n]}")
        ok = ok and (div is None or div >= first_tie)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
