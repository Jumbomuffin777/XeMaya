#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The GLM-5.3-Flash prompts the GPU windows measure and check, as token ids (the server's own tokenizer, which gives
llama.cpp's ids), written to OUT/<name>.ids with OUT/prompts.json describing them.

    make_prompts.py TOKENIZER_DIR CHAT_TEMPLATE OUT

- chat-agent: an agent's first turn through the model's chat template (system prompt, a read_file tool, a question)
- code-512: 512 tokens of this repository's C++ (plain text completion)
- ctx-2377: 2,377 tokens of docs/DETAILS.md (the length of Maya's llama.cpp benchmark prompt)
- ctx-8k, ctx-32k, ctx-64k, ctx-128k: that many tokens of the repository's docs and sources, for long-context speed
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402

tok_dir, tpl, out = sys.argv[1:4]
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)
tok = ST.Tokenizer.from_dir(tok_dir)
meta = {}


def write(name, ids, about):
    (out / f"{name}.ids").write_text(",".join(map(str, ids)))
    meta[name] = {"tokens": len(ids), "about": about}
    print(f"{name}: {len(ids)} tokens")


ct = ChatTemplate(tpl)
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
msgs = [{"role": "system", "content": "You are a coding agent working in a C++ inference engine's repository. "
                                      "Use the tools to read files before you answer."},
        {"role": "user", "content": "Where does the engine decide how many layers each GPU gets? Read the file first."}]
write("chat-agent", tok.encode(ct.render(msgs, tools), parse_special=True),
      "an agent's first turn: system prompt, read_file tool, question, thinking opened by the template")

code = (ROOT / "src/core/plain_glm5.cpp").read_text()
write("code-512", tok.encode(code)[:512], "the first 512 tokens of src/core/plain_glm5.cpp")
details = (ROOT / "docs/DETAILS.md").read_text()
write("ctx-2377", tok.encode(details[:14000])[:2377], "2,377 tokens of docs/DETAILS.md")

corpus = []
for p in sorted(ROOT.glob("docs/*.md")) + sorted(ROOT.glob("src/core/*.cpp")) + sorted(ROOT.glob("src/kernels/xe/*.cpp")):
    corpus.append(f"\n\n// ==== {p.relative_to(ROOT)} ====\n\n" + p.read_text(errors="replace"))
    if sum(len(c) for c in corpus) > 700_000:
        break
ids = tok.encode("".join(corpus))
for n, name in ((8192, "ctx-8k"), (32768, "ctx-32k"), (65536, "ctx-64k"), (131072, "ctx-128k")):
    if len(ids) >= n:
        write(name, ids[:n], f"{n} tokens of the repository's docs and sources")
(out / "prompts.json").write_text(json.dumps(meta, indent=1))
