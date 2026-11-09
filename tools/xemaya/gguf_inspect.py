#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Summarise a GGUF file's header: architecture, key metadata and tensor types by role.

Reads a local file or an https URL (HTTP range requests, only the header is fetched; a Hugging Face model's shards
can be named as hf:OWNER/REPO/PATH).  No dependency beyond the standard library.

  gguf_inspect.py FILE_OR_URL [--keys REGEX] [--tensors REGEX] [--json]
"""
import argparse, json, re, struct, sys, urllib.request
from collections import Counter, defaultdict

GGML_TYPES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 9: 'Q8_1', 10: 'Q2_K',
              11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K', 16: 'IQ2_XXS', 17: 'IQ2_XS', 18: 'IQ3_XXS',
              19: 'IQ1_S', 20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS', 24: 'I8', 25: 'I16', 26: 'I32',
              27: 'I64', 28: 'F64', 29: 'IQ1_M', 30: 'BF16', 34: 'TQ1_0', 35: 'TQ2_0', 39: 'MXFP4', 40: 'NVFP4',
              41: 'Q1_0', 42: 'Q2_0'}


class Source:
    """Byte source that fetches a file or URL lazily in growing chunks."""
    def __init__(self, path):
        if path.startswith('hf:'):
            owner, repo, rest = path[3:].split('/', 2)
            path = f'https://huggingface.co/{owner}/{repo}/resolve/main/{rest}'
        self.path, self.buf, self.url = path, b'', path.startswith('http')
        self.f = None if self.url else open(path, 'rb')

    def need(self, end):
        while len(self.buf) < end:
            want = max(end - len(self.buf), len(self.buf), 1 << 20)
            if self.url:
                req = urllib.request.Request(self.path, headers={
                    'Range': f'bytes={len(self.buf)}-{len(self.buf) + want - 1}', 'User-Agent': 'xemaya-gguf-inspect'})
                with urllib.request.urlopen(req, timeout=60) as r:
                    chunk = r.read()
            else:
                chunk = self.f.read(want)
            if not chunk:
                raise EOFError('file ended inside the header')
            self.buf += chunk


class Reader:
    def __init__(self, src):
        self.s, self.o = src, 0

    def take(self, n):
        self.s.need(self.o + n)
        b = self.s.buf[self.o:self.o + n]; self.o += n
        return b

    def u(self, fmt):
        return struct.unpack('<' + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.u('Q')).decode('utf-8', 'replace')

    def value(self, t):
        scal = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if t in scal:
            return self.u(scal[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.u('I'), self.u('Q')
            if et in scal and n > 64:     # long numeric arrays: skip the bytes, keep a summary
                self.take(n * struct.calcsize(scal[et]))
                return f'<array {n} x type {et}>'
            if et == 8 and n > 64:
                for _ in range(n): self.string()
                return f'<array {n} strings>'
            return [self.value(et) for _ in range(n)]
        raise ValueError(f'unknown metadata type {t}')


def role(name):
    """A tensor's role, from its llama.cpp-style name."""
    if not name.startswith('blk.'):
        return name.split('.')[0] if name.count('.') <= 1 else name.rsplit('.', 1)[0]
    rest = name.split('.', 2)[2].rsplit('.', 1)[0]
    return rest


def inspect(path):
    r = Reader(Source(path))
    if r.take(4) != b'GGUF':
        raise ValueError('not a GGUF file')
    version, n_t, n_kv = r.u('I'), r.u('Q'), r.u('Q')
    meta = {}
    for _ in range(n_kv):
        k = r.string(); meta[k] = r.value(r.u('I'))
    tensors = []
    for _ in range(n_t):
        name = r.string(); nd = r.u('I')
        dims = [r.u('Q') for _ in range(nd)]
        t = r.u('I'); off = r.u('Q')
        tensors.append({'name': name, 'dims': dims, 'type': GGML_TYPES.get(t, f'type{t}'), 'offset': off})
    return {'version': version, 'meta': meta, 'tensors': tensors, 'header_bytes': r.o}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path')
    ap.add_argument('--keys', default='', help='regex: print metadata keys matching it (default: all short ones)')
    ap.add_argument('--tensors', default='', help='regex: list tensors matching it')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    g = inspect(a.path)
    if a.json:
        json.dump(g, sys.stdout, default=str); return
    print(f"GGUF v{g['version']}: {len(g['tensors'])} tensors, {len(g['meta'])} keys, header {g['header_bytes']} B")
    kr = re.compile(a.keys) if a.keys else None
    for k, v in g['meta'].items():
        if (kr and kr.search(k)) or (not kr and not k.startswith('tokenizer.ggml.')):
            s = str(v)
            print(f'  {k} = {s[:160]}')
    by_role = defaultdict(Counter)
    for t in g['tensors']:
        by_role[role(t['name'])][t['type']] += 1
    print('tensor types by role:')
    for rl in sorted(by_role):
        print(f'  {rl:40s} ' + ', '.join(f'{k}:{v}' for k, v in sorted(by_role[rl].items())))
    if a.tensors:
        tr = re.compile(a.tensors)
        for t in g['tensors']:
            if tr.search(t['name']):
                print(f"  {t['name']:50s} {t['type']:8s} {t['dims']}")


if __name__ == '__main__':
    main()

