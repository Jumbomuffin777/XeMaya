<!--
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# The Windows worker's first run on an NVIDIA card

Nothing here has been executed: the Windows engine compiles and links, and has never been started.
Each step says what a pass looks like and where to stop. Stop at the first kernel message or wrong
result and record it; do not go on to the next step. Machine facts (addresses, folders): `LOCAL.md`.

## 0. Before touching the GPU

| Check | How | Pass |
|---|---|---|
| NVIDIA driver | `nvidia-smi` | the card and a driver that supports CUDA 12.4 |
| cuBLAS 12 | `where cublas64_12.dll` | a path (CUDA 12 toolkit, or `pip install nvidia-cublas-cu12` and its `bin` folder on `PATH`) |
| Visual C++ runtime | `where msvcp140.dll vcruntime140_1.dll` | both in `System32` |
| Worker files | `certutil -hashfile strata.exe SHA256` against `SHA256SUMS` | equal for every file |
| Firewall | inbound TCP on the worker's port from the server's address only | rule present |

The front-end's Hybrid pane lists what of this is missing and refuses to start the worker without it.

## 1. The program starts and sees the card (no model, seconds)

```
strata-device.exe --list-devices
```

Pass: the NVIDIA card is listed with its memory. A missing-DLL dialog or exit code 0xC0000135 is step 0.
Stop if the card is absent: the SYCL runtime did not load `ur_adapter_cuda.dll` (is it beside the programs?). `strata-device.exe --selftest` then runs the device self-test (seconds, a few MiB).

## 2. The segment loads alone (the PC only, no server, no network)

With the staged files of the PC's blocks (the front-end stages them; a staged shard is the full size
on disk only in name, its other blocks are holes):

```
set XEMAYA_SEGMENT_TOKEN=<64 hex characters>
strata.exe --native <staged shard 1> --segment-first <K> --segment-serve 127.0.0.1:7411 --mmap-experts --spec 0
```

Pass: `SEGMENT {"event":"loaded",...}` then `SEGMENT {"event":"listening","port":7411}`; VRAM use in
`nvidia-smi` near the plan's figure; Ctrl+C ends it cleanly. Start with the smallest model and the
fewest blocks (MiniMax-M3.1-Flash, its last 4 blocks). Stop at any `ERR` line.

## 3. One short request through the split (needs an approved GPU window on the server)

Server: the same model with `--segment-last <K> --remote-segment <PC>:7411` (the plan file does this).
One prompt, the chat template, 32 forced steps, then `tools/xemaya/compare_ref.py` against the saved
llama.cpp reference.

Pass: the same top-1 tokens as the unsplit run on that prompt (MiniMax, 211 tokens: 32 of 32), and
the link's lines in the engine's log. Record both sides' per-block times and the link's time.

## 4. Only then

Longer prompts, FP16 rows, GLM-5.3 and DeepSeek-V4.1 (its Engram blocks, 1 and 14 of 40, stay on the
server for any cut past block 14: the table is never copied to the PC), saved conversation states,
speed. Each is unverified until run.

## What to record

The commit in `COMMIT`, the driver and CUDA versions, the exact command lines (never the token), the
`SEGMENT` lines, `nvidia-smi` before and after, and the comparison's output.
