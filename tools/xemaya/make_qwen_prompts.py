#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Qwen3.8 Flash-Next benchmark prompts as token ids (the pack's tokenizer, bench_core's corpus), for
qwen_bench.py: the same ids go to every engine.

    make_qwen_prompts.py PACK OUT_DIR [LONG_TOKENS=26800]

short-prose / short-code: a system line and one request.  long-a-code / long-a-prose: LONG_TOKENS of source files and
a request for code or for prose (the two share everything but the request: a prefix-reuse pair).  long-b-prose /
long-b-code: the same over other files (a second cold read).  turn2: a follow-up user message, appended to a prompt and
its answer.  Thinking is off (an empty <think> block), so the decode is the answer itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_core as C  # noqa: E402

SYSTEM = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
OPEN = "<|im_start|>user\n"
CLOSE = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROSE = ("Write a long short story, in plain flowing prose with no lists, no headings and no code, about a lighthouse "
         "keeper who finds a letter in a bottle. Take your time with the scenery and the keeper's thoughts.")
CODE = ("Write a complete, well-structured Python module implementing an LRU cache with time-based expiry, a "
        "thread-safe wrapper, and a pytest test suite for it. Output only code.")
LONG_PROSE = ("\nAbove are source files of an inference engine. In plain flowing prose, with no lists, no headings and "
              "no code, write a long essay for a general reader describing what this software does and how its parts "
              "work together.")
LONG_CODE = ("\nAbove are source files of an inference engine. Write a new, complete C++ source file in the same style "
             "that adds a small command-line tool printing, for a prompt of N tokens, how the prompt would be split "
             "into chunks. Output only code.")
TURN2 = "Now continue for as long as you can in the same manner, without repeating yourself."


def main():
    pack, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 26800
    tok = C.load_tokenizer(pack / "tokenizer")

    def enc(s):
        return tok.encode(s, parse_special=True)
    body = []
    for path, text in C.load_corpus()["files"]:
        body += enc(C.file_block(path, text))
        if len(body) >= 2 * n + 4096:
            break
    assert len(body) >= 2 * n, len(body)
    head = enc(SYSTEM + OPEN)
    p = {"short-prose": enc(SYSTEM + OPEN + PROSE + CLOSE), "short-code": enc(SYSTEM + OPEN + CODE + CLOSE),
         "turn2": enc("<|im_end|>\n" + OPEN + TURN2 + CLOSE)}
    for name, lo in (("a", 0), ("b", n)):
        for kind, ask in (("prose", LONG_PROSE), ("code", LONG_CODE)):
            tail = enc(ask + CLOSE)
            p[f"long-{name}-{kind}"] = head + body[lo:lo + n - len(head) - len(tail)] + tail
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in p.items():
        (out / f"{name}.ids").write_text(",".join(map(str, ids)))
        print(name, len(ids))


if __name__ == "__main__":
    main()
