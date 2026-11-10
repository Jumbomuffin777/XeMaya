#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The models XeMaya can run, found on this machine, as JSON for a front-end (Frontier UI's XeMaya backend).

    catalog.py [--root DIR ...] [--configs DIR] [--engine EXE] [--write] [--tokenizers] [--pretty]

Walks the roots (default: $XEMAYA_MODEL_ROOTS, a colon-separated list) for GGUF files (a split model's first shard),
reads each header and keeps the architectures XeMaya's engine runs (`general.architecture`): glm5-next, deepseek41,
qwen4exp and minimax-m3.  Nothing is loaded and no GPU is opened.

Per model: its id, name, family, quantization, files and bytes, the context it was trained for, what the engine and
the server do with it (tools, reasoning, vision, drafting), how far it is checked (`state`: "verified" against a
reference on the Arc cards, or "experimental", with the reason in `note`), and the launch of serve/server.py for it:
the engine's arguments and the server's config file.

--configs DIR is where the server configs live (default: $XEMAYA_SERVE_CONFIGS): DIR/<id>.json and the model's
tokenizer folder DIR/<id>/tokenizer.  --write writes the configs; --tokenizers also extracts each tokenizer from its
GGUF (tools/strata_tokenizer.py; seconds a model).  A model without its tokenizer folder is listed with
`"ready": false` and what is missing in `"missing"`, so a front-end shows it without offering to start it.
"""
import argparse
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from gguf_inspect import Reader, Source

# general.architecture -> what XeMaya does with it.  `state` and `note` are the record of xemaya/STATUS.md: a family
# is "verified" once the engine agreed with a reference on the Arc cards.
FAMILIES = {
    "glm5-next": {
        "family": "glm-5.3", "label": "GLM-5.3", "state": "verified", "tools": True, "reasoning": True,
        "vision": False, "draft": True,
        "note": "Checked against llama.cpp on three Arc cards (126 of 128 forced tokens); agent sessions with tool "
                "calls ran through the server.",
        "args": ["--layer-split", "auto"]},
    "deepseek41": {
        "family": "deepseek-v4.1", "label": "DeepSeek-V4.1", "state": "verified", "tools": False, "reasoning": False,
        "vision": False, "draft": False,
        "note": "Checked against llama.cpp on three Arc cards (16 of 16 on both references). The server has no "
                "DeepSeek tool-call format yet, and thinking is not qualified.",
        "args": ["--layer-split", "auto", "--spec", "0"]},
    "qwen4exp": {
        "family": "qwen3.8", "label": "Qwen3.8", "state": "verified", "tools": True, "reasoning": True,
        "vision": False, "draft": True,
        "note": "Benchmarked against Strata on B70 + B65 (equal decode, prefill 10-17% behind).",
        "args": []},
    "minimax-m3": {
        "family": "minimax-m3.1", "label": "MiniMax-M3.1", "state": "experimental", "tools": False,
        "reasoning": True, "vision": False, "draft": False,
        "note": "Experimental: correct on its chat template (32 of 32 forced tokens on three Arc cards) but slow "
                "(2-4 tok/s decode): 30% of its experts do not fit in VRAM. No tool calls, vision or drafting yet.",
        "args": ["--layer-split", "auto", "--mmap-experts", "--spec", "0"]},
}
SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def header(path):
    """The metadata of a GGUF file (no tensor table)."""
    r = Reader(Source(str(path)))
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    _version, _n_t, n_kv = r.u("I"), r.u("Q"), r.u("Q")
    meta = {}
    for _ in range(n_kv):
        k = r.string()
        meta[k] = r.value(r.u("I"))
    return meta


def first_shards(roots):
    for root in roots:
        root = pathlib.Path(root)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.gguf")):
            m = SHARD.search(p.name)
            if m and m.group(1) != "00001":
                continue
            if p.name.startswith("mmproj"):
                continue
            yield p


def shards_of(first):
    m = SHARD.search(first.name)
    if not m:
        return [first]
    n = int(m.group(2))
    return [first.with_name(SHARD.sub(f"-{i:05d}-of-{m.group(2)}.gguf", first.name)) for i in range(1, n + 1)]


def slug(text):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9.]+", "-", text.lower())).strip("-")


def describe(first, configs, engine):
    try:
        meta = header(first)
    except (OSError, ValueError, KeyError, IndexError) as e:
        return {"path": str(first), "error": f"unreadable header: {e}"}
    arch = meta.get("general.architecture")
    fam = FAMILIES.get(arch)
    if fam is None:
        return None
    files = shards_of(first)
    missing_files = [str(f) for f in files if not f.is_file()]
    stem = SHARD.sub("", first.name).removesuffix(".gguf")
    mid = "xemaya-" + slug(stem)
    name = meta.get("general.name") or stem
    quant = next((q for q in ("IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "Q4_K_M", "Q5_K_M",
                              "Q6_K", "Q8_0") if q.lower() in stem.lower()), None)
    trained = int(meta.get(f"{arch}.context_length") or 0)
    # the context a start asks for by default: the K/V cache is VRAM the experts do not get
    default_ctx = min(trained or 32768, 32768)
    args = ["--native", str(first), "--max-context", str(default_ctx), *fam["args"]]
    out = {
        "id": mid, "name": f"{name}" + (f" · {quant}" if quant and quant not in name else ""),
        "family": fam["family"], "family_label": fam["label"], "architecture": arch, "quantization": quant,
        "path": str(first), "files": len(files), "bytes": sum(f.stat().st_size for f in files if f.is_file()),
        "context_trained": trained, "context_default": default_ctx,
        "state": fam["state"], "note": fam["note"],
        "capabilities": {"tools": fam["tools"], "reasoning": fam["reasoning"], "vision": fam["vision"],
                         "streaming": True, "draft": fam["draft"]},
        "engine_args": args, "ready": True, "missing": [],
    }
    if missing_files:
        out["ready"] = False
        out["missing"].append(f"{len(missing_files)} of {len(files)} model files")
    if configs is not None:
        tok = configs / mid / "tokenizer"
        out["config"] = str(configs / f"{mid}.json")
        out["tokenizer"] = str(tok)
        if not (tok / "tokenizer.json").is_file():
            out["ready"] = False
            out["missing"].append("the tokenizer folder (catalog.py --write --tokenizers)")
    if engine is None or not pathlib.Path(engine).is_file():
        out["ready"] = False
        out["missing"].append("the engine binary")
    return out


def server_config(model, engine):
    return {"exe": str(engine), "args": model["engine_args"], "cwd": str(pathlib.Path(model["tokenizer"]).parent),
            "tokenizer": model["tokenizer"], "model_name": model["id"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", help="a folder to search (repeatable; default $XEMAYA_MODEL_ROOTS)")
    ap.add_argument("--configs", default=os.environ.get("XEMAYA_SERVE_CONFIGS"),
                    help="the folder of the server configs and tokenizers")
    ap.add_argument("--engine", default=os.environ.get("XEMAYA_ENGINE"), help="the engine binary (strata)")
    ap.add_argument("--write", action="store_true", help="write each ready model's server config")
    ap.add_argument("--tokenizers", action="store_true", help="with --write: extract missing tokenizers first")
    ap.add_argument("--pretty", action="store_true")
    a = ap.parse_args()
    roots = a.root or [r for r in os.environ.get("XEMAYA_MODEL_ROOTS", "").split(":") if r]
    configs = pathlib.Path(a.configs) if a.configs else None
    models, skipped = [], []
    for first in first_shards(roots):
        d = describe(first, configs, a.engine)
        if d is None:
            continue
        (skipped if "error" in d else models).append(d)
    if a.write and configs is not None:
        configs.mkdir(parents=True, exist_ok=True)
        for m in models:
            tok = pathlib.Path(m["tokenizer"])
            if a.tokenizers and not (tok / "tokenizer.json").is_file():
                try:
                    import strata_tokenizer
                    tok.parent.mkdir(parents=True, exist_ok=True)
                    strata_tokenizer.extract(m["path"], str(tok.parent))   # it writes OUT/tokenizer
                    m["missing"] = [x for x in m["missing"] if "tokenizer" not in x]
                    m["ready"] = not m["missing"]
                except (ValueError, KeyError, OSError, ImportError) as e:   # a tokenizer it does not take: not ready
                    m["missing"] = [x for x in m["missing"] if "tokenizer" not in x] + [f"the tokenizer ({e})"]
                    m["ready"] = False
            if a.engine and (tok / "tokenizer.json").is_file():
                pathlib.Path(m["config"]).write_text(json.dumps(server_config(m, a.engine), indent=1) + "\n")
    doc = {"service": "xemaya", "catalog_version": 1, "roots": roots, "engine": a.engine,
           "families": sorted({v["family"] for v in FAMILIES.values()}), "models": models, "unreadable": skipped}
    json.dump(doc, sys.stdout, indent=1 if a.pretty else None, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
