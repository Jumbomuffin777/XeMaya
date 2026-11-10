#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compares two runs' logits along the same tokens (a forced run's rows): A.f32 against B.f32.

    compare_logits.py A.f32 B.f32 VOCAB [--ref REFERENCE.json]

A.f32 and B.f32: rows of VOCAB little-endian floats (the engine's --dump-logits, mm_host's or ds41_host's output), the
same number of rows.  Per row: each side's top token, A's margin over its second, the Kullback-Leibler divergence
KL(A || B) in nats and the difference of A's top token's log-probability.  The summary: the rows whose top tokens are
equal, the median and mean KL.  --ref: a llama.cpp reference (compare_ref.py's format) whose step tokens are printed
beside (the tokens the rows were forced along).

It is the measure where no side is exact: an engine and llama.cpp's CPU backend each round the activations their own
way, and a model that amplifies the rounding is judged by how far each is from the host reference in floats.
"""
import json
import math
import struct
import sys


def rows(path, vocab):
    data = open(path, "rb").read()
    n = len(data) // (4 * vocab)
    return [struct.unpack_from(f"<{vocab}f", data, i * 4 * vocab) for i in range(n)]


def logprobs(row):
    m = max(row)
    z = m + math.log(sum(math.exp(v - m) for v in row))
    return [v - z for v in row]


def main(argv):
    if len(argv) < 4:
        print(__doc__)
        return 2
    vocab = int(argv[3])
    a, b = rows(argv[1], vocab), rows(argv[2], vocab)
    ref = None
    if "--ref" in argv:
        ref = json.load(open(argv[argv.index("--ref") + 1]))["steps"]
    n = min(len(a), len(b))
    if n == 0:
        print("no rows")
        return 1
    same, kls = 0, []
    print(f"{'row':>4s} {'A top':>8s} {'B top':>8s} {'A margin':>9s} {'KL(A||B)':>9s} {'d logprob':>10s}")
    for i in range(n):
        la, lb = logprobs(a[i]), logprobs(b[i])
        ta = max(range(vocab), key=la.__getitem__)
        tb = max(range(vocab), key=lb.__getitem__)
        second = max(v for j, v in enumerate(la) if j != ta)
        kl = sum(math.exp(p) * (p - q) for p, q in zip(la, lb) if p > -40.0)
        kls.append(kl)
        same += ta == tb
        forced = f"  forced {ref[i]['token']}" if ref is not None and i < len(ref) else ""
        print(f"{i:4d} {ta:8d} {tb:8d} {la[ta] - second:9.3f} {kl:9.4f} {lb[ta] - la[ta]:10.3f}{'' if ta == tb else '  DIFFERENT'}{forced}")
    kls.sort()
    print(f"top tokens equal at {same} of {n} rows; KL(A||B) median {kls[n // 2]:.4f}, mean {sum(kls) / n:.4f}, max {kls[-1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
