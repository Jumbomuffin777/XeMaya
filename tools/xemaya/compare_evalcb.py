#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compares the engine's per-layer dump with llama.cpp's llama-eval-callback, tensor by tensor.

    compare_evalcb.py EVALCB.txt ENGINE.err [NAME ...]

EVALCB.txt is llama-eval-callback's standard output for a prompt; ENGINE.err the engine's standard error for the same
tokens with XEMAYA_DUMP_LAYERS=1 (plain_prompt.cpp: `dump NAME-LAYER = {width, rows} sum = S vals ...`, the excerpt
llama-eval-callback prints: the first and last 3 rows, the first and last 3 values of each).  Only 2-D tensors of the
same shape are compared (llama.cpp's last block keeps one row).  For each: the largest |difference| over the excerpt
relative to the excerpt's largest |value|, and both sums.  NAME limits the tensors (ffn_inp, l_out, ...).
"""
import re
import sys


def llama_tensors(path):
    out, name, shape, vals = {}, None, None, []
    head = re.compile(r"common_debug_cb_eval:\s+(\S+)(?: \(\w+\))? = \((\w+)\)\s+(\w+)\(.*\) = \{(\d+), (\d+), (\d+), (\d+)\}")
    num = re.compile(r"-?\d+\.\d+(?:e[-+]?\d+)?|-?inf|nan")
    for line in open(path, errors="replace"):
        m = head.match(line)
        if m:
            name, shape, vals = m.group(1), tuple(int(m.group(i)) for i in (4, 5, 6, 7)), []
            continue
        if name is None:
            continue
        m = re.match(r"\s+sum = (\S+)", line)
        if m:
            out[name] = (shape, vals, float(m.group(1)))   # the last tensor of a name wins
            name = None
            continue
        if "[" in line and re.search(r"\d\.\d", line):
            vals.extend(float(v) for v in num.findall(line))
    return out


def engine_tensors(path):
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"dump (\S+) = \{(\d+), (\d+)\} sum = (\S+) vals(.*)", line)
        if m:
            out[m.group(1)] = ((int(m.group(2)), int(m.group(3)), 1, 1), [float(v) for v in m.group(5).split()],
                               float(m.group(4)))
    return out


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    ll, en = llama_tensors(argv[1]), engine_tensors(argv[2])
    want = argv[3:]
    print(f"{'tensor':16s} {'worst rel':>10s} {'llama sum':>14s} {'engine sum':>14s}")
    for name, (shape, vals, total) in en.items():
        base = name.rsplit("-", 1)[0]
        if want and base not in want:
            continue
        if name not in ll:
            continue
        lshape, lvals, ltotal = ll[name]
        if lshape != shape or len(lvals) != len(vals):
            continue
        top = max(abs(v) for v in lvals) or 1.0
        worst = max(abs(a - b) for a, b in zip(lvals, vals)) / top
        print(f"{name:16s} {worst:10.4f} {ltotal:14.4f} {total:14.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
