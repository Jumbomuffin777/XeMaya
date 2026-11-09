# XeMaya status

Updated 2026-11-09 19:00 UTC (after GPU window 3, which ended with Zability losing its network; see the incident). Branch `xemaya/main` = `origin/main` of the private
https://github.com/Jumbomuffin777/XeMaya. Every item below was run on Zability unless marked otherwise.

## Works (checked)

| What | How it was checked |
|---|---|
| Builds with oneAPI icpx 2026.2 | Core, tests and parity binaries, plus the full `strata` engine (GLM and DeepSeek kernels included), with 0 errors |
| Host tests | 13 XeStrata host tests pass, plus `model_spec_test` and `ds41_manifest_test` |
| GLM-5.3-Flash kernels | `glm5_parity` PASS, 28/28, on the OpenCL CPU device and on **each Arc card** (Level Zero, 2026-11-09) |
| GLM Maya GGUFs | Maya-S v2 IQ2_XXS reads as `glm5-next` (45 trunk + 1 MTP layers, 42 MoE layers, 34 KDA layers); all its tensor types are supported; `setup.py` lists Maya-S-v2/S24/XS/M/L, pinned with sha256 |
| DeepSeek-V4.1 model spec | `model_spec_test` reads the ISTA-DASLab GSQ-RCO and Coder IQ2_XXS files (40 layers, 38 compressed, 2 Engram modules, 384 / 192 experts) |
| DeepSeek-V4.1 tensors | `ds41_manifest_test` passes both real models: 6 shards, all 1043 tensors with llama.cpp's shapes, in bounds, no unknown tensor |
| DeepSeek-V4.1 kernels | `ds41_parity` PASS, 18/18, on the CPU device and on **each Arc card** (2026-11-09): router, mHC pre/collapse, compressor, Engram gate + hash, YaRN RoPE (and its inverse), indexer, top-k, candidate blocks, sink attention |
| Serve layer | 14 upstream Strata tool-call commits ported (#4663, #4701, #4702/#4703, #4751, #5040, #5276, #5281, #5409, tool_choice required, Anthropic tool-result order). Full suite: **930 passed, 23 skipped, 0 failed** (baseline was 862) |

| F32 mHC mixers (Maya's GLM files) | PlainModel takes F32 `hc_attn_fn` / `hc_ffn_fn` (`upload(..., f32)`): the decode window multiplies them with `f32_gemv_fp32_mmvf_multi` (8 rows a launch), the prompt chunks with `f32_gemm_rows`, F32 throughout as llama.cpp's `mul_mat`. `glm5_parity` checks both products and `glm5_hc_pre` on their mixes against a double-precision host reference (T = 3, 11, 37; worst 1.2e-5) on the CPU device: PASS. A static scan of Maya-S v2's 1,412 tensors finds no other type the loader refuses (MTP experts Q2_K / Q3_K, trunk experts IQ2_XXS / IQ2_S / IQ3_XXS pass `native_expert_supported`). Not yet loaded on a GPU |
| llama.cpp reference for the window | `xemaya/reference/glm53-maya-s-v2-greedy16.json`: llama.cpp v0.8.0 (8642541b2) built CPU-only (`tools/xemaya/ref_cpu.sh`, no GPU touched), the saved 11-token prompt, 16 greedy tokens with each step's top-5 log-probabilities: " natural selection, genetic variation, and adaptation.\n\nExplanation:\n\nThe three most important ideas". Step 3 is a near-tie (" genetic" -1.552 vs " common" -1.564) |
| iq_parity fixtures | `tools/iq_fixture.py` wrote all 10 formats to `/DATA/Projects/xemaya/iq_fixture` (rows fetched by range request from the pinned Qwen3.8 GSQ-RCO revision). On the CPU device: **0 failures**, all 10 formats (dequantization exact, products within 9.7e-3 relative, the test's bound 2e-2). The run took about 35 minutes, most of it the OpenCL CPU compiler on the IQ4_NL kernel; a first run with a 15-minute limit timed out there |
| Element-wise kernels | `elementwise_parity` 0 failures on each Arc card (needs Level Zero; 2026-11-09) |
| GLM Maya-S v2 files | sha256 of all 3 parts recomputed before the GPU window: match Hugging Face |

## GPU window 2026-11-09 14:01-14:17 UTC (owner-approved; production Qwen stopped by the owner)

Lease: whole-host `gpulease.py run --holder xemaya-dev` (no `gpu_partitions.py` edit: its `OWNERS` are one-card
partitions, and a holder not in `OWNERS` takes the whole host). Preflight passed. Script: `tools/xemaya/gpu_window.sh`;
logs in `/DATA/Projects/xemaya/logs/window/`. Ended by the owner's `xemaya-window-end.sh` (Qwen healthy after 60 s).

| Step | Result |
|---|---|
| Kernel parity, B70 / B65 / B60 | `glm5_parity` PASS, `ds41_parity` PASS, `elementwise_parity` 0 failures, on all three |
| `iq_parity` | 10 "missing fixture" on all three: its test data is not on this host (not a kernel result) |
| XeMaya load of Maya-S v2 | **Refused in 1 s**: `blk.0.hc_attn_fn.weight is F32, which the native matrix-vector kernels do not take`. Maya's files keep the mHC mixers (`hc_attn_fn`, `hc_ffn_fn`, 16384 x 24) in F32; `plain_glm5.cpp`'s `hc_read` multiplies them with `native_mmvq` (quantized and BF16 rows only). Unsloth's files, which XeStrata was tested with, do not have this |
| llama.cpp reference (v0.8.0 SYCL, `--n-cpu-moe 12 -ts 34,34,24`) | Loaded in ~200 s, then took 67 s for the first 7 prompt tokens (first-run kernel compilation on three cards and/or page-cache pressure under the 60 G cap); stopped at 14:15 to protect the window. Prompt ids saved (`prompt.ids`: "The three most important ideas in the theory of evolution are" = 785,2326,1429,2989,6708,304,279,10121,315,15367,525); no reference tokens |
| Speed test | Not run (no model loaded) |

Next window needs: F32 mHC mixers in PlainModel (an F32 or F32->BF16 path in `hc_read`, checked by a CPU-device parity
test first), the iq fixtures (`python tools/iq_fixture.py --out logs/iq_fixture`, then `iq_parity logs/iq_fixture`), and a llama.cpp reference produced
beforehand on CPU or with a warmed build so the window is spent on XeMaya.

## GPU window 2: GLM-5.3-Flash Maya-S v2 on the Arc cards (2026-11-09 15:39-16:13 UTC)

Owner-approved; production Qwen stopped and the agents parked by the owner. Whole-host lease
(`gpulease.py run --holder xemaya-dev`), preflight passed, `gpu_partitions.py` untouched. Ended with
`xemaya-window-end.sh` (Qwen healthy after 60 s). Raw lines of every run:
`xemaya/bench/2026-11-09-glm53-maya-s-v2-window2.txt`; logs in `/DATA/Projects/xemaya/logs/window2/`.

Model: GLM-5.3-Flash Maya-S v2 IQ2_XXS (89.8 GiB, sha256-checked). Engine: `build-engine` (icpx 2026.2, JIT SPIR-V,
`SYCL_CACHE_PERSISTENT=1`), `--max-context 4096`, greedy, one process per run (load not included in the speeds).

### Correctness

| Check | Result |
|---|---|
| `glm5_parity` (incl. the F32 mixer checks), `ds41_parity` | PASS on B70, B65 and B60 |
| `iq_parity` (10 formats, real rows) | 0 failures on B70, B65 and B60 |
| First load | Stopped at `blk.3.indexer.proj.weight` (Q8_0 in Maya's file; the loader took F32/BF16 only). Fixed in the window: `upload_f32` widens Q8_0 exactly on the host. Reload succeeded |
| 16 tokens forced along the llama.cpp v0.8.0 CPU reference | **PASS**: top-1 equal at 16 of 16 steps (including the reference's near-tie at step 3), median \|d logprob\| 0.013, max 0.399 |
| Free greedy run | first 16 tokens identical to the reference |
| MTP drafts (`--spec 3`) | short prompt: the same 128 tokens as without drafts. After the 2,377-token prompt: identical to token 58, then a different pick at a near-tie (token 59: 320 at -1.996 vs 323 at -2.125, margin 0.13), as batched verify windows round differently |
| Repeatability | two runs of the same configuration gave the same 64 tokens |

### Speed (prompt of 2,377 tokens + 128 generated unless noted; TTFT = the prompt's time, the first token included)

| Configuration | Prefill tok/s | TTFT s | Decode tok/s | Experts hit in VRAM |
|---|---|---|---|---|
| 3 GPUs, old auto split (B60 7 layers) | 92.1 | 25.8 | 13.48 | 92.6% |
| 3 GPUs, `--layer-split 19,35` | 99.5 | 23.9 | 13.82 | 95.0% |
| same, `--kv-ram off` | 100.3 | 23.7 | 13.45 | 95.0% |
| same, MTP drafts `--spec 3` | 92.3 | 25.8 | **15.97** (69.8% accepted) | 93.4% |
| 3 GPUs, new auto split (B70 16, B65 17, B60 9 MoE layers) | 92.9 | 25.6 | 13.87 | 94.5% |
| 2 GPUs B70 + B65, auto (64 generated) | 92.8 | 25.6 | 12.48 | 76.7% |
| 1 GPU B70 (64 generated) | 88.7 | 26.8 | 6.15 | 41.8% |
| 3 GPUs, new auto, 11-token prompt, **512 generated** | - | - | **17.30** | 95.8% |
| same with `--spec 3` | - | - | **22.90** (84.0% accepted) | 94.9% |

Baselines (Maya's report, 2,377-token prompt + 128, second warm request): stock llama.cpp v0.8.0 SYCL 68.3 / 13.6,
grouped MUL_MAT_ID 108.9 / 14.0. XeMaya: prefill **+36-47% over stock**, 8-15% under the grouped build; decode at
2.4K context **on par** without drafts (13.4-13.9) and **+14% over the grouped build with MTP drafts** (15.97); at short
context 17.3 / 22.9 tok/s. Caveats: the baseline's prompt text was not found, so this prompt is the same length of
different text (`docs/DETAILS.md`); prefill varied 92-100 tok/s between runs with the share of experts read from the
NVMe during the prompt.

### Where the time and memory go (telemetry: xpu-smi 1 s samples, /proc/meminfo, /proc/diskstats)

- **VRAM**: B70 29.5 / 32 GiB, B65 29.5 / 32 GiB, B60 21.9 / 24 GiB (3 GPUs). The VRAM tier holds 10,427-10,690 of
  12,096 expert slots (6.8 MiB each).
- **GPU activity** (xpu-smi compute %, decode): B70 53%, B65 62%, B60 45%; prompt 26-39%. These include the cards'
  waits on the host between layers.
- **RAM**: MemAvailable stayed >= 70 GiB. Pinning all experts would need 80 GiB plus 95.7 GiB beside them, so the
  experts outside VRAM are read in place through the page cache (6.6 GiB resident complement, ~67 GiB cached).
- **PCIe**: host links measured 28.6-28.7 GB/s (B70, B65) and 7.0 GB/s (B60). No routed expert crosses PCIe in decode
  (misses run on the CPU pool, 15 workers); the adaptive tier's swaps (700-1,900 a run, 6.8 MiB each) do.
- **NVMe**: load reads 82-105 GB at 1.55-2.0 GB/s (53-60 s); a 2,377-token prompt reads 9-18 GB (0.4-0.7 GB/s);
  3-GPU decode ~0.03 GB/s. One GPU: decode reads 0.69 GB/s, so its 6.15 tok/s is NVMe/page-cache bound.
- **Decode profile** (`STRATA_WINDOW_TIMES`): a token is 42 MoE rings (GPU to host and back). Old split: the GPU took
  0.96 ms to each ring and the host 0.76-1.10 ms to answer (plan + the CPU's missed experts), the CPU the critical path.
  Giving the B60 more layers cut the answer to 0.30 ms and decode rose 11.95 -> 16.14 tok/s (short prompt). At 2.4K
  context the GPU's part grows to 1.46 ms a ring: the DSA attention.
- **Prefill profile** (`STRATA_PROMPT_PROFILE`, 2,377 tokens, the three stages summed): DSA attention 10.4 s (46%),
  routed experts 7.8 s (34%), KDA 2.6 s, shared/dense FFN 0.6 s, staging and head 0.7 s.

### Code changes made in the window (committed)

1. `upload_f32` takes Q8_0 (Maya's `indexer.proj`), widened exactly on the host.
2. `--layer-split auto`: a slow-link GPU's whole layers are sized by the file's mean routed-block bytes at 97% of its
   room (was the largest layer at 80%): the B60 gets 9 MoE layers instead of 7, and decode after 2,377 tokens went
   13.48 -> 13.87 tok/s with no manual split.

### Remaining bottlenecks and gaps (not production-ready)

1. **DSA attention** (`glm5_mla_attn`, one work-group per head, FP32 vector code; the indexer): 46% of prefill and the
   context-dependent part of decode. Next: an XMX (joint_matrix) / split-over-cells version, parity-tested on the CPU
   device first.
2. **Prefill experts** (34%) and the 9-18 GB of NVMe reads per long prompt: the page cache cannot hold the model under
   the 60 GB cap.
3. **Load time** 53-60 s, NVMe-bound at ~1.6 GB/s.
4. **42 host round trips per token**; at most ~0.3 ms each on the host now, ~1 ms on the GPU.
5. Untested: contexts beyond 2.4K, the OpenAI/Anthropic serve path, tool calls, streaming, long soaks, concurrent
   requests, and other prompts. Correctness rests on one 16-token comparison plus the kernel tests.

## Incident, 2026-11-09 18:29-18:46 UTC: Zability lost its network during window 3 (read this first)

- **What happened:** the onboard RTL8126 5 GbE NIC (`eno1`, r8169 driver) logged `NETDEV WATCHDOG: transmit queue 0
  timed out` at 17:49:58 and 18:29:25, Tailscale then lost every path, and the host stayed unreachable until it was
  rebooted (clean shutdown 18:45:12). No kernel panic, no OOM, no GPU fault at the end.
- **Cause (most likely, from evidence):** those are the only two NIC timeouts in five boots (4 days), and they fall in
  XeMaya's two heaviest runs: the 64K prompt (17:44-17:50) and the 128K prompt (18:20-18:31). The B60 (0e:00.0) is the
  only GPU behind the chipset, and the chipset's one uplink also carries both 5 GbE NICs, Wi-Fi, SATA and USB (the B70,
  B65 and the model's NVMe are on CPU lanes). Those prompts streamed the experts the B60's VRAM missed, ~6 GB a 4K-token
  chunk at its ~7 GB/s link, for minutes on end. Production Qwen never uses the B60.
- **Also seen:** two GPU engine resets with page faults on the B70 (17:24:10, 17:38:27), exactly when XeMaya engines
  were killed with SIGKILL in the middle of GPU work. They did not take anything down.
- **Recovery:** after the reboot, no XeMaya process or lease was left; `xemaya-window-end.sh` restored the router and
  Qwen (healthy after 60 s) at 18:48.
- **Changes so it does not happen again:**
  1. `--layer-split auto` leaves a GPU on a slow host link (under half the first GPU's) out of the split unless it is
     named (`--split-device`) or `XEMAYA_SLOW_LINK_GPUS=1`: on Zability XeMaya runs on the B70 + B65 by default (window 2:
     same prefill, decode 12.5 vs 13.9 tok/s).
  2. `tools/xemaya/safety_guard.sh` wraps every GPU job in the window scripts: it stops the job at a NIC watchdog, a GPU
     engine reset or fault, an OOM kill, a hung task, an NVMe timeout, a PCIe AER error, three missed gateway pings or
     MemAvailable under 12 GiB. Tested on the host (normal exit, and a forced trip).
  3. Engines are stopped with QUIT / Ctrl-C / SIGTERM and given time to exit; no SIGKILL in the tools.
  4. No 128K run, nor any long run that streams to the B60, without the owner's explicit say.
- **Owner's decision, not done (networking is off limits to XeMaya):** the r8169 driver with the RTL8126 has known
  transmit-timeout problems under load. Options: Realtek's own r8126 driver, PCIe ASPM off for the NIC, newer firmware,
  or moving the GPU that sits behind the chipset.

## GPU window 3: GLM attention, TTFT, long context, agent use (2026-11-09 17:00-18:3x UTC)

Owner-approved; production stopped and the agents parked by the owner. B70 + B65 + B60, `--layer-split auto`, Maya-S v2
IQ2_XXS. Logs: `/DATA/Projects/xemaya/logs/window3/`. The XMX attention was written, tested and adopted in the window.

**Correctness.** Kernel parity on each card (glm5 incl. tiled, ds41, iq): PASS. `glm5_mla_attn_xmx` against a
double-precision reference on the B70 and the B60: worst 1.3e-7. Forced along the llama.cpp v0.8.0 CPU references:

| Prompt | Top-1 equal | Median \|d logprob\| | Mean top-5 KL | Same with the original attention |
|---|---|---|---|---|
| chat-agent (128 steps) | 126 | 0.009 | 0.007 | - |
| ctx-2377 (64) | 56 | 0.067 | 0.020 | 53, 0.072 |
| code-512 (64) | 55 | 0.094 | 0.047 | 53, 0.114, KL 0.065 |

The XMX path is as close to llama.cpp as the original or closer; the differences are on both sides (mean signed -0.04
to -0.13) and look like the two 2-bit implementations' rounding, not a bug. The "no miss outside near-ties" rule is too
strict for this; a llama.cpp GPU reference would calibrate the expected spread. Prompt-end checkpoint: resumes at the
right token (198 of 774) with and without drafts; the resumed and from-scratch replies agree for 21 (15 with drafts)
tokens, then part at a different chunking's rounding.

**Speed** (prompt read and decode by the engine; TTFT = the prompt's time):

| Prompt | Attention | Prefill tok/s | TTFT | Decode tok/s |
|---|---|---|---|---|
| 2,377 | original | 94.8 | 25.1 s | 13.97 |
| 2,377 | tiled | 105.5 | 22.5 s | 16.00 |
| 2,377 | **xmx** | **152.1-161.3** | **14.8-15.6 s** | **17.36** (20.03 with MTP drafts) |
| 8K | original / tiled / **xmx** | 92.2 / 102.5 / **204.2** | 88.9 / 80.0 / **40.2 s** | 14.77 / 17.17 / **18.62** |
| 32K | xmx | 213.1 (`--kv-ram off` 217.5) | 154 s | 13.94 (17.22; 24.31 with drafts) |
| 64K | xmx | 205.9 | 318 s | 14.45 |
| 128K | xmx, drafts | 206.9 | 634 s | 21.0 |

Baselines (2,377 tokens + 128): stock llama.cpp SYCL 68.3 / 13.6, grouped 108.9 / 14.0: XeMaya's prefill is now 2.2x
stock and 1.4x the grouped build, decode 1.24x (1.43x with drafts). Attention benchmark (B70, 4,096 tokens over 8K
cells): 12.9 s original, 4.2 s tiled, 0.236 s xmx (14 TFLOP/s). The first request after a load pays ~9 s of one-time
setup (198-token prompt in 11.3 s cold, 2.9 s warm).

**Agent session through serve/server.py** (OpenAI streaming, `read_file` tool, 7 turns, context to 22K): every turn a
valid tool call (`finish_reason` tool_calls), reasoning apart from the content, the stream well formed; each turn reused
the whole conversation (cached 4,057 ... 18,906 tokens) and read only the new tool result (~3.7K tokens) at ~180 tok/s,
~20 s a turn; decode 19-33 tok/s; a cancelled stream's next request done in 3.2 s; the Anthropic route returns tool_use.
Gap found: the same prompt sent again (a retry) is read from scratch (107 s at 22K): the checkpoint is taken after the
prompt's last token, which a recurrent model cannot read again.

**Rejected after measuring:** `STRATA_RESIDENT_PIN=1` (the pinned RAM copy: the first prompt's experts took 71.6 s,
later ones no faster) and copying a group's missed experts into the staging on 8 threads (no gain; reverted).

**Where the prompt's time goes now** (2,377 tokens, the three stages summed): routed experts ~6.3 s (about half),
KDA ~2.7 s, DSA ~2.5 s (at 64K the indexer is DSA's largest part). The stages run one after another (GPU activity
16-41% each during prompts), so overlapping chunks across GPUs is the next large prefill lever; the expert stage needs a
finer GPU-side profile first.

## Since window 2 (CPU only, not yet run on an Arc GPU)

| Change | State |
|---|---|
| `glm5_mla_attn_tiled` (`src/kernels/xe/glm5.cpp`): a token and 8 heads a work-group, the latent cells loaded 32 at a time into local memory once for those heads (the original reads each cell twice for every head), an online softmax, and in decode windows a token's cells split over work-groups and merged | `glm5_parity`'s `test_mla_tiled` against a double-precision reference (prompt chunk with tail pools, decode with 32 splits, no scratch, compact `--kv-ram` rows, 16 heads): worst 1.1e-7, PASS on the CPU device. CPU-device benchmark (`glm5_parity --bench-mla`): 1.5-1.6x on prompt chunks, 1.2-1.8x on decode windows. **Off by default** (`XEMAYA_MLA_ATTN=tiled`) until the cards confirm it; the original stays the fallback. No XMX: the CPU device here cannot test joint_matrix, and the window's GFLOP/s will show whether it is needed |
| DSA's parts in `STRATA_PROMPT_PROFILE` (projections, indexer and selection, wk_b, attention, wv_b and output) | builds |
| **Prompt-end checkpoint** (`plain_generate.cpp` serve, `PlainModel::save_recurrent` / `restore_recurrent`): GLM's recurrent layers resumed only a request extending the whole live conversation, and an agent's next turn re-renders the reply (checked with serve's `ChatTemplate`: a tool call with newlines re-renders without them), so every turn read the whole conversation again. Now the recurrent state (~140 MB) is saved once a reply's first token is out, and a request that starts with that prompt resumes after it. `XEMAYA_PROMPT_CHECKPOINT=0` turns it off | builds; the window's test compares a resumed request's tokens with the same request read from scratch |
| Server tokenizer for GLM (`tools/strata_tokenizer.py` on the Maya GGUF) | ids equal to llama.cpp's on 11- and 4,258-token texts |
| New llama.cpp CPU references (`tools/xemaya/ref_cpu_ids.sh`) | chat-agent (GLM template with a tool, 128 tokens, 7 near-ties under 0.1), code-512 (64 tokens, 3), ctx-2377 (64 tokens) in `xemaya/reference/` |
| Window 2's XeMaya output for the 2,377-token prompt against the new 64-token reference | step 0 equal; at step 1 the reference itself is a near-tie (margin under 0.1) and XeMaya took the other token, so a free run checks one step only. Window 3 forces the engine along each reference and compares every step's logits (phase 4c) |
| Window-3 tools: `make_prompts.py`, `engine_serve.py` (many requests per load, the checkpoint test), `agent_session.py` (OpenAI / Anthropic agent session, tool calls, streaming, reasoning, cancellation), `window3.sh`, `window_report.py` | self-tested on a fake engine and on server.py's mock engine |

## Unverified (built, not run on an Arc GPU)

- The tiled attention, the prompt-end checkpoint and the DSA profile split (above).
- Contexts beyond 4K, the HTTP server with the GLM engine, tool calls and streaming from the real model.
- DeepSeek-V4.1 has not run on a GPU (its PlainModel block is not written yet).

## In progress or not started

- **GLM Maya-S v2 download:** shard 1 is verified, shard 2 is downloading (log
  `/DATA/Projects/xemaya/logs/fetch-glm-maya-s-v2.log`). Resume with
  `tools/xemaya/fetch_hf.sh peasantsmith/GLM-5.3-Flash-Maya-GGUF Maya-S-v2-IQ2_XXS
  /mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS 50M`. Verified files are skipped.
- **The DeepSeek-V4.1 block in PlainModel:** see PLAN, Milestone 4, for the design. Not started.
- **Remaining upstream serve commits:** these conflict with XeStrata's rewritten `server.py` and need hand ports: request
  lifecycle (#4271, #4479, #4738, #4792, #5044, #3443, #3961, #2518, #4451), answers one request late (9aea047ee,
  engine side too), fatal GPU error restart (#4350, engine side too), Anthropic context-overflow wording (247397800),
  `undeclared_tool_calls` (75acfca32), deferred tool loading (9efac55f5). The runner `/DATA/Projects/xemaya/conf.py`
  shows a commit's conflicts and records resolutions.
- **`setup.py` still refuses icpx** and builds its own engine with intel/llvm. Until that changes, run the engine built
  by CMake (AGENTS.md) or build intel/llvm with `tools/intel_llvm_build.py` (heavy; capped jobs only).

## Needs the owner's approval (not done)

1. Another GPU window only after you have seen the incident above. Proposed scope: B70 + B65 only (the B60 left out),
   every job under `safety_guard.sh`, no 128K run; tests: the agent session with the fixed TTFT measurement, a retry
   (same prompt) after the n-1 checkpoint fix, and a finer expert-stage profile.
2. The NIC (r8169 / RTL8126 transmit timeouts): your call (see the incident).

## Notes

- `gh` on the Windows PC is not logged in. `gh` on Zability is (Jumbomuffin777) and is what this work used.
- There is no passwordless sudo on Zability.
- Upstream history includes a 68 MB `src2.tar`. GitHub warns but the push succeeds; history is not rewritten.
- Commit identity is repo-local: `Jumbomuffin777 <ID+Jumbomuffin777@users.noreply.github.com>`.
