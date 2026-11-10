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

--units DIR (with --write) also writes a systemd user unit for each ready model, `xemaya-<family>[-n].service`, which
takes the host's GPU lease (--lease) and runs serve/server.py under tools/xemaya/safety_guard.sh on its own port
(--port-base upwards, in the catalog's order), the GPUs --gpus names; the catalog then gives each model its `unit`,
`port` and `upstream` (--listen-host).  These are the facts Frontier UI's router reads: it starts and stops models
only through their units.  Machine facts (paths, the address, the GPUs) come from the arguments, never from here.

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
        # for a front-end that plans a hybrid segment: the blocks, and the floats a row has where they are cut
        "blocks": int(meta.get(f"{arch}.block_count") or 0), "embedding_length": int(meta.get(f"{arch}.embedding_length") or 0),
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
        out["hybrid_file"] = str(configs / f"{mid}.hybrid.json")   # serve/server.py's hybrid_plan: a front-end writes it
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
            "tokenizer": model["tokenizer"], "model_name": model["id"], "hybrid_file": model["hybrid_file"]}


UNIT = """[Unit]
Description=XeMaya {name} (Frontier router model {mid})
After=network-online.target
[Service]
Type=simple
{env_file}ExecStart={lease} run --holder {holder} --memory-budget-gib {budget} --note "XeMaya {family}; exclusive GPU lease; never evicts the always-on model" -- {guard} --log {log} -- /bin/bash -c {cmd}
MemoryHigh={mem_high}G
MemoryMax={mem_max}G
KillMode=mixed
KillSignal=SIGINT
TimeoutStopSec=300
Restart=no
SuccessExitStatus=130 143
"""


def unit_names(models):
    """xemaya-<family>, with -2, -3 ... for a family's further files: stable while the set of models is."""
    seen, out = {}, []
    for m in models:
        base = "xemaya-" + slug(m["family"]).replace(".", "")
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}-{seen[base]}")
    return out


def write_units(models, a, configs):
    units = pathlib.Path(a.units)
    units.mkdir(parents=True, exist_ok=True)
    logs = pathlib.Path(a.log_dir) if a.log_dir else configs / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    server = HERE.parent.parent / "serve" / "server.py"
    for i, (m, name) in enumerate(zip(models, unit_names(models))):
        m["port"] = a.port_base + i
        m["upstream"] = f"http://{a.listen_host}:{m['port']}"
        if not m["ready"]:
            continue
        m["unit"] = f"{name}.service"
        run = (f"{a.setup + ' >/dev/null 2>&1; ' if a.setup else ''}"
               f"export ONEAPI_DEVICE_SELECTOR={a.gpus} ZES_ENABLE_SYSMAN=1 SYCL_CACHE_PERSISTENT=1"
               f"{' XEMAYA_SLOW_LINK_GPUS=1' if a.slow_link_gpus else ''}; "
               f"exec {a.python} {server} --engine strata --config {m['config']} --host {a.listen_host} "
               f"--port {m['port']}{'' if a.env_file else ' --no-api-key'}")
        text = UNIT.format(name=m["name"], mid=m["id"], lease=a.lease, holder=name, budget=a.memory_gib,
                           family=m["family_label"], guard=a.guard or (HERE / "safety_guard.sh"),
                           log=logs / f"{name}.guard.log", cmd="'" + run.replace("'", "'\\''") + "'",
                           mem_high=a.memory_gib + 6, mem_max=a.memory_gib + 12,
                           env_file=f"EnvironmentFile={a.env_file}\n" if a.env_file else "")
        (units / m["unit"]).write_text(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", help="a folder to search (repeatable; default $XEMAYA_MODEL_ROOTS)")
    ap.add_argument("--configs", default=os.environ.get("XEMAYA_SERVE_CONFIGS"),
                    help="the folder of the server configs and tokenizers")
    ap.add_argument("--engine", default=os.environ.get("XEMAYA_ENGINE"), help="the engine binary (strata)")
    ap.add_argument("--write", action="store_true", help="write each ready model's server config")
    ap.add_argument("--tokenizers", action="store_true", help="with --write: extract missing tokenizers first")
    ap.add_argument("--units", help="with --write: write each ready model's systemd user unit into this folder")
    ap.add_argument("--port-base", type=int, default=18140, help="the first model's port (default 18140)")
    ap.add_argument("--listen-host", default="127.0.0.1", help="the address the servers listen on")
    ap.add_argument("--gpus", default="level_zero:0,1", help="ONEAPI_DEVICE_SELECTOR for the servers")
    ap.add_argument("--slow-link-gpus", action="store_true", help="let --layer-split auto use a slow-link GPU")
    ap.add_argument("--lease", default="gpulease.py", help="the GPU lease program (its `run` subcommand)")
    ap.add_argument("--guard", help="safety_guard.sh (default: the one beside this script)")
    ap.add_argument("--python", default=sys.executable, help="the Python that runs serve/server.py")
    ap.add_argument("--setup", default="", help="a shell command run before the server (the SYCL runtime's setvars)")
    ap.add_argument("--env-file", help="a systemd EnvironmentFile with STRATA_API_KEY (none: no API key)")
    ap.add_argument("--memory-gib", type=int, default=72, help="host memory a server may take (the lease budget)")
    ap.add_argument("--log-dir", help="the safety guard's logs (default: CONFIGS/logs)")
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
        if a.units:
            write_units(models, a, configs)
    doc = {"service": "xemaya", "catalog_version": 1, "roots": roots, "engine": a.engine,
           "families": sorted({v["family"] for v in FAMILIES.values()}), "models": models, "unreadable": skipped}
    json.dump(doc, sys.stdout, indent=1 if a.pretty else None, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
