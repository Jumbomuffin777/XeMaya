#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 the XeMaya contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""A coding agent's session against XeMaya's OpenAI-compatible server (serve/server.py), measured turn by turn.

    agent_session.py --url http://127.0.0.1:PORT --out RESULTS.json [--files N] [--file-chars C] [--effort low]

The agent is told to read N files of this repository with a read_file tool, one call a turn, then answer. Each tool
result is the file's first C characters, so the conversation grows by a few thousand tokens a turn, as a coding agent's
does. Greedy (temperature 0). Per turn (streamed): the time to the first streamed byte of any kind (TTFT), the whole
turn's time, the prompt's tokens and the ones the server reused (usage.prompt_tokens_details.cached_tokens), the
engine's prompt and decode speeds (timings), and checks:
  - every SSE line parses, the stream ends with [DONE]
  - a turn that calls a tool ends with finish_reason "tool_calls", its arguments are JSON with the required "path"
  - the reasoning arrives as reasoning_content, never as <think> inside the content
Then: a cancelled stream (the connection closed mid-answer) and the next request's recovery time; and the Anthropic
route (/v1/messages) with a tool, streamed.
"""
import argparse
import http.client
import json
import pathlib
import sys
import time
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file of the repository (its first part)",
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path from the repository root"}},
                   "required": ["path"]}}}]
FILES = ["src/core/plain_model.cpp", "src/core/plain_glm5.cpp", "src/prefill/plain_prompt_glm5.cpp",
         "src/program/plain_generate.cpp", "src/kernels/xe/glm5.cpp", "include/strata/core/plain_model.hpp",
         "src/core/expert_source.cpp", "src/core/moe_window.cpp", "serve/frontend.py", "serve/server.py",
         "docs/DETAILS.md", "setup.py"]


def stream(url, path, body, headers=None, cancel_after=None):
    """POSTs and reads the SSE stream: (events, first_byte_s, total_s, framing_ok, cancelled)."""
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    t0 = time.time()
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", **(headers or {})})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"{path}: HTTP {r.status}: {r.read()[:300]!r}")
    events, first, ok, done = [], None, True, False
    while True:
        line = r.readline()
        if not line:
            break
        line = line.decode().strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        try:
            ev = json.loads(data)
            events.append(ev)
            # the first token: a delta that carries text, reasoning or a call (not the role chunk the server sends
            # at once), or an Anthropic content delta
            d = ((ev.get("choices") or [{}])[0].get("delta") or {}) if isinstance(ev, dict) else {}
            if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls") or
                                  ev.get("type") == "content_block_delta"):
                first = time.time() - t0
        except ValueError:
            ok = False
        if cancel_after is not None and len(events) >= cancel_after:
            c.close()
            return events, first, time.time() - t0, ok, True
    c.close()
    return events, first, time.time() - t0, ok and (done or path.endswith("/messages")), False


def chat_turn(url, model, messages, effort, max_tokens, cancel_after=None):
    body = {"model": model, "messages": messages, "tools": TOOLS, "stream": True, "temperature": 0,
            "max_tokens": max_tokens, "reasoning_effort": effort, "stream_options": {"include_usage": True}}
    ev, first, total, framing, cancelled = stream(url, "/v1/chat/completions", body, cancel_after=cancel_after)
    content, reasoning, calls, finish, usage, timings = "", "", {}, None, None, None
    for e in ev:
        if e.get("usage"):
            usage = e["usage"]
        if e.get("timings"):
            timings = e["timings"]
        for ch in e.get("choices") or []:
            d = ch.get("delta") or {}
            content += d.get("content") or ""
            reasoning += d.get("reasoning_content") or ""
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": "", "id": None})
                f = tc.get("function") or {}
                slot["name"] += f.get("name") or ""
                slot["arguments"] += f.get("arguments") or ""
                slot["id"] = tc.get("id") or slot["id"]
            finish = ch.get("finish_reason") or finish
    return {"content": content, "reasoning": reasoning, "calls": [calls[k] for k in sorted(calls)], "finish": finish,
            "usage": usage, "timings": timings, "ttft_s": first, "total_s": total, "framing_ok": framing,
            "cancelled": cancelled, "events": len(ev)}


def read_file(path, chars):
    p = (ROOT / path).resolve()
    if ROOT not in p.parents or not p.is_file():
        return f"error: no file {path}"
    return p.read_text(errors="replace")[:chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="glm")
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--file-chars", type=int, default=12000)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    files = FILES[:a.files]
    task = ("Read these files with the read_file tool, one call per turn, in this order: " + ", ".join(files) +
            ". Do not answer before you have read all of them. Then explain in three sentences how the engine "
            "decides which GPU runs which layers.")
    messages = [{"role": "system", "content": "You are a coding agent in a C++ inference engine's repository."},
                {"role": "user", "content": task}]
    turns, problems = [], []
    for turn in range(a.files + 3):
        r = chat_turn(a.url, a.model, messages, a.effort, a.max_tokens)
        u = r["usage"] or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) if u else None
        t = r["timings"] or {}
        rec = {"turn": turn, "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": cached,
               "completion_tokens": u.get("completion_tokens"), "ttft_s": r["ttft_s"], "total_s": r["total_s"],
               "prompt_per_second": t.get("prompt_per_second"), "predicted_per_second": t.get("predicted_per_second"),
               "finish": r["finish"], "calls": r["calls"], "reasoning_chars": len(r["reasoning"]),
               "content_chars": len(r["content"]), "framing_ok": r["framing_ok"]}
        if not r["framing_ok"]:
            problems.append(f"turn {turn}: the SSE stream did not parse or end with [DONE]")
        if "<think>" in r["content"] or "</think>" in r["content"]:
            problems.append(f"turn {turn}: thinking markers in the content")
        for c in r["calls"]:
            try:
                args = json.loads(c["arguments"] or "{}")
                if c["name"] != "read_file" or "path" not in args:
                    problems.append(f"turn {turn}: call {c['name']} {c['arguments']!r} lacks read_file's path")
            except ValueError:
                problems.append(f"turn {turn}: call arguments are not JSON: {c['arguments']!r}")
        if r["calls"] and r["finish"] != "tool_calls":
            problems.append(f"turn {turn}: calls made but finish_reason {r['finish']}")
        print(f"turn {turn}: prompt {rec['prompt_tokens']} (cached {cached}), TTFT {r['ttft_s']:.2f} s, turn "
              f"{r['total_s']:.1f} s, prefill {rec['prompt_per_second']}, decode {rec['predicted_per_second']} tok/s, "
              f"{rec['completion_tokens']} tokens, finish {r['finish']}, calls "
              f"{[(c['name'], c['arguments']) for c in r['calls']]}", flush=True)
        turns.append(rec)
        msg = {"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]}
        if r["calls"]:
            msg["tool_calls"] = [{"id": c["id"] or f"call_{turn}_{i}", "type": "function",
                                  "function": {"name": c["name"], "arguments": c["arguments"]}}
                                 for i, c in enumerate(r["calls"])]
        messages.append(msg)
        if not r["calls"]:
            break
        for i, c in enumerate(r["calls"]):
            try:
                path = json.loads(c["arguments"] or "{}").get("path", "")
            except ValueError:
                path = ""
            messages.append({"role": "tool", "tool_call_id": msg["tool_calls"][i]["id"], "content": read_file(path, a.file_chars)})
    # a warm repeat: the last request again (all of it cached)
    warm = chat_turn(a.url, a.model, messages[:-1] if messages[-1]["role"] == "assistant" else messages, a.effort, 16)
    print(f"warm repeat: TTFT {warm['ttft_s']:.2f} s, cached {((warm['usage'] or {}).get('prompt_tokens_details') or {}).get('cached_tokens')}"
          f" of {(warm['usage'] or {}).get('prompt_tokens')}", flush=True)
    # cancellation: a long answer's stream closed after 20 events, then a short request
    cancel = chat_turn(a.url, a.model, [{"role": "user", "content": "Write a long essay about GPUs."}], "low", 2000,
                       cancel_after=20)
    t0 = time.time()
    after = chat_turn(a.url, a.model, [{"role": "user", "content": "Say OK."}], "low", 32)
    rec_s = time.time() - t0
    if not after["framing_ok"] or after["finish"] is None:
        problems.append("the request after a cancelled stream did not complete")
    print(f"cancel: closed after {cancel['events']} events; the next request done in {rec_s:.1f} s "
          f"(finish {after['finish']})", flush=True)
    # the Anthropic route with a tool
    ev, first, total, framing, _ = stream(a.url, "/v1/messages", {
        "model": a.model, "max_tokens": 512, "stream": True, "temperature": 0,
        "tools": [{"name": "read_file", "description": "Read a file", "input_schema": TOOLS[0]["function"]["parameters"]}],
        "messages": [{"role": "user", "content": "Read src/core/plain_glm5.cpp with the read_file tool."}]},
        headers={"anthropic-version": "2023-06-01"})
    kinds = [e.get("type") for e in ev]
    tool_use = any((e.get("content_block") or {}).get("type") == "tool_use" for e in ev)
    stop = next((e.get("delta", {}).get("stop_reason") for e in ev if e.get("type") == "message_delta"), None)
    if not tool_use or stop != "tool_use":
        problems.append(f"anthropic: tool_use block {tool_use}, stop_reason {stop}")
    print(f"anthropic: {len(ev)} events, tool_use {tool_use}, stop_reason {stop}, first byte {first:.2f} s", flush=True)
    json.dump({"turns": turns, "warm": {k: warm[k] for k in ("ttft_s", "total_s", "usage")},
               "cancel": {"events": cancel["events"], "next_request_s": rec_s, "next_finish": after["finish"]},
               "anthropic": {"events": len(ev), "tool_use": tool_use, "stop_reason": stop, "kinds": sorted(set(map(str, kinds)))},
               "problems": problems}, open(a.out, "w"), indent=1)
    print("PROBLEMS:\n  " + "\n  ".join(problems) if problems else "no problems", flush=True)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
