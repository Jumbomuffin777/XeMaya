# XeMaya status

> **Documentation audit, 2026-11-10 (source `b59488e2d`).** Read the latest handoff first. The dated
> sections below retain historical checkpoints; their “not run”, “not started” and “unverified” statements
> describe those checkpoints, not necessarily the current implementation. No GPU work was run for this audit.
> The subsequent model catalog in `tools/xemaya/catalog.py` adds local GGUF discovery, config generation and
> DeepSeek/MiniMax tokenizer support. Template/tokenizer checks do not qualify MiniMax's end-to-end serve path.
> The reset stop in the latest handoff remains unresolved. See [CHANGELOG](../CHANGELOG.md) for audit evidence.


Updated 2026-11-10 12:30 UTC (MiniMax-M3.1-Flash runs on the three Arc cards; the handoff is the next section).
https://github.com/Jumbomuffin777/XeMaya. Every item below was run on Zability unless marked otherwise.

## Handoff: where the work stands (2026-11-10 12:30 UTC, read this first)

- **GPU work is stopped: a GPU engine reset at 12:07:39 UTC (the owner decides whether it goes on).**
  `xe 0000:08:00.0: Fault response: Unsuccessful -EBUSY`, then `Engine reset: engine_class=bcs` (the card's copy
  engine), 16 s into the first request of a MiniMax run started for the first time **without `--mmap-experts`** (the
  experts outside VRAM in RAM: 26.29 GiB, 1,877 experts; three cards; `logs/mm-gpu/w2-warm.*`). No process had been
  signalled before it. `safety_guard.sh` stopped the job at once (exit 99); afterwards the three cards were idle
  (42 MB used each), the lease free, the gateway answering, no NIC watchdog, AER or OOM message, RAM 83 GiB available.
  The eight MiniMax runs before it, all with `--mmap-experts`, logged nothing. **Not established:** the cause. What
  differs in that run is the copies from the pinned RAM copy to the cards (the adaptive tier's exchanges, on the copy
  engine); whether a copy's address or length was wrong, or the driver refused a page of registered memory, the logs
  do not say. Until the owner says otherwise: no GPU job; and when it goes on, `--mmap-experts` stays on for MiniMax
  and the RAM copy is tried first on one card with a short request.

- **Operations.** Production Qwen (`zability-strata-qwen38.service`) is stopped and masked by the owner: leave it off,
  do not restore it. The Sonnet supervisor, Hermes and Pi stay parked. GPU work is authorized on B70 + B65 + B60, one
  job at a time through `tools/xemaya/gpu_job.sh` (the lease, the safety guard, a memory cap, a time limit; it
  refuses while the production unit is active or the kernel log has a NIC, GPU, AER or OOM message). The B60 is
  named with `XEMAYA_GPUS=level_zero:0,1,2 XEMAYA_SLOW_LINK_GPUS=1`. No kernel message in any run of this session.
- **DeepSeek-V4.1** (the section "Three GPUs for DeepSeek-V4.1" below): committed and pushed (`61bd5d654`); both
  forced references 16 of 16 on B70 + B65 + B60 with the XMX attention, 207-225 tok/s prefill, 19.5-21.7 decode.
- **MiniMax-M3.1-Flash Maya-S** (the section below): runs natively on the three cards, 32 of 32 on its chat-template
  reference. Slow so far (about 60 tok/s prefill, 2-4 tok/s decode): 30% of its experts do not fit in VRAM.
  **Next, in order:**
  1. the experts outside VRAM: the RAM copy instead of the files (the run without `--mmap-experts` ended in the
     engine reset above: find out why first, on one card), the B60's spare 3.9 GiB (a tenth layer), a tier profile at
     load, the missed experts over PCIe in decode;
  2. `mm_msa_select`'s scores are a work-item a block (FP32): a decode token at 3.8K context went 3.7 -> 2.4 tok/s
     with the selection on; score on the matrix engines or a work-group a block;
  3. the prompt profile (`STRATA_PROMPT_PROFILE=1`), then the chunked path's levers as GLM's and DeepSeek's;
  4. serve: MiniMax's chat template and tool calls (`serve/`), the vision encoder and the DSpark drafter
     (`/mnt/ai/models/minimax-m3.1/{vision,dspark}`, safetensors: not started; Project Maya's `src/mm3/mm3_vision.*`
     and `mm3_dspark.hpp` are the references);
  5. GLM-5.3 and Qwen3.8 forced checks again (`canon_gu_rows` now applies the experts' SwiGLU limit; the direct
     tiers are shared code): not rerun in this session.

## MiniMax-M3.1-Flash Maya-S IQ2_XXS on the Arc cards (2026-11-10)

Model: `/mnt/ai/models/minimax-m3.1/Maya-S/` (3 shards, 114 GB; 60 blocks, 3 leading dense, 57 x 128 experts of
14.3 MiB, 4 used; 64 heads of 128 on 4 K/V heads; the sparse attention: blocks of 128 positions, 16 taken, 1 local).
The model runs in PlainModel's generic block (`mm_`): qwen3moe's attention, `glm5_router_sigmoid` with the selection
bias, swiglu-oai (alpha 1.702, limit 7) in the dense FFN, the shared expert and the experts, and the sparse attention
of `include/strata/kernels/minimax.hpp` (`mm_msa_attn_xmx` on the matrix engines).

**What decides correctness here.** llama.cpp's CPU backend rounds a quantized product's activations to int8 a block of
256 (Q8_K); the engine rounds them a block of 32; `src/reference/mm_host.cpp` keeps floats. On the model's chat
template the three agree. On raw text without the template's first tokens (the first two references, `short` and
`mid`) the first token's row amplifies every rounding (a 2e-4 difference in a product's input moved its output by
6e-3) and no two implementations agree: those references do not decide anything.

| Check | Result |
|---|---|
| `minimax_parity` (selection, attention whole and in 16 parts, swiglu-oai) | PASS on the OpenCL CPU device and on B70, B65, B60 (the matrix-engine attention within 1.4e-6 of a double-precision reference) |
| `mm_host --ggml-acts` against llama-eval-callback, 23 tokens | token 0 equal to 4 decimals through the first block; the other tokens within 1% through the first routed blocks |
| Chat template, 211-token prompt, 32 steps: `mm_host` in floats against llama.cpp | 32 of 32, median \|d logprob\| 0.000, max 0.125 |
| the same: `mm_host --acts32 1` (the engine's rounding) against the floats | 32 of 32, mean KL 0.0003 |
| the same: **the engine on B70 + B65 + B60** (blocks 0-26, 27-50, 51-59; XMX attention; chunked prompt) | **32 of 32, median 0.000, max 0.131** |
| 3,832-token chat prompt (30 blocks: the selection is on), 32 steps, the engine on the three cards | 28 of 32, median 0.013, max 1.035; the 4 misses at reference margins 0.11, 0.18, 0.62, 1.21 |
| the same with `XEMAYA_MM_DENSE=1` (no selection) | 28 of 32, median 0.014: this prompt does not tell the selection from none |
| Raw text, 24 tokens, 16 steps: llama.cpp against `mm_host` in floats | 9 of 16 (llama.cpp's own rounding); the engine against `mm_host` 8-13 of 16 from run to run |
| the same on the host: int8 a block of 32 against floats; with the remainder rounded again (`--acts32 2`) | 9 of 16, median KL 0.20; 15 of 16, median KL 0.002 |

**Unverified:** the selection inside the model against llama.cpp's own selection (only the kernel's parity test checks
it); `mm_host` at 3.8K tokens (hours on the host); the 528-token raw reference; contexts past 3.8K; the serve path,
the chat template in `serve/`, tool calls; vision and the drafter (not started); the free mode and the Windows
cross build (built with icpx 2026.2 only).

**Speed, first runs** (three cards, `--layer-split auto`, `--mmap-experts`, greedy, `--spec 0`; one load, requests in
turn; the host's CPU was shared with reference jobs during this run, so the figures are a floor):

| Request | Prefill tok/s | TTFT s | Decode tok/s | Experts hit in VRAM |
|---|---|---|---|---|
| chat, 211 tokens, first after the load | 2.0 | 108 | 1.78 | 78% |
| 3,832 tokens (3,645 read) | 50.9 | 72 | 2.16 | 86% |
| chat again | 7.4 | 29 | 2.94 | 84% |
| 3,832 tokens again | 59.7 | 61 | 2.32 | 89% |

VRAM: B70 31.6 GB, B65 31.7 GB, B60 23.0 GB. The tier holds 4,608 of 7,296 experts (B70 69 a layer, B65 75, the
B60's 9 layers whole). Each request read 95-198 GB from the model's files: the experts outside VRAM are read from the
drive every time under `--mmap-experts`, and the CPU pool took 92% of the decode windows' time.

## Three GPUs for DeepSeek-V4.1 (2026-11-10 04:30-06:12 UTC, commit `61bd5d654`)

Coder GSQ-RCO IQ2_XXS on B70 + B65 + B60 (`--layer-split auto` with `XEMAYA_SLOW_LINK_GPUS=1`: blocks 0-13, 14-28,
29-39, every expert in VRAM), greedy, forced along the llama.cpp v0.8.0 CPU references. Logs:
`/DATA/Projects/xemaya/logs/window6/`.

- A layer split now cuts anywhere (`PlainModel::DsCarry`: the closed compressed rows, the last selection and the
  candidate blocks cross a cut with the streams); a stage whose tier holds every expert plans its windows' experts on
  the GPU (`MoeWindow::direct`: no ring, a window one graph); `ds41_sink_attn_xmx` in prompt chunks and windows.
- Correctness: three stages on the OpenCL CPU device 16 of 16 on both prompts; three GPUs with the XMX attention
  16 of 16 (median 0.009, max 0.096) and 16 of 16 (0.010, 0.155). GLM-5.3 on three GPUs: 126 of 128, its baseline.
- Warm 576-token prompt: two GPUs chunked 79.9-100.0 tok/s, three GPUs 99.2-110.3, + direct 124.7-125.4, + XMX
  **206.9-209.2**; 2,876 tokens: 93.2, 93.2, 100.3, **224.8** tok/s; the first prompt after a load 15.9 -> 133.9
  (the load 6 -> 37 s: it reads all 68 GB). Decode after those prompts: 15.6-16.6, 15.1-16.1, 15.6-18.0,
  **19.5-21.7** tok/s.
- Not checked: Qwen with direct tiers on a GPU, the full 384-expert model on three GPUs.

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

## Earlier handoff (2026-11-10 01:00 UTC; superseded by the one at the top)

- **2026-11-10 03:54 UTC: DeepSeek-V4.1 runs natively on the B70 + B65** (window 5, the section below this handoff).
  Steps 1 and 2 of the list here are done; step 3 (the chunked prompt path) is next.
- **No GPU window is open.** Window 4 was closed at 22:50 UTC (production Qwen healthy, lease `qwen38-strata-iq3s`).
  The Sonnet supervisor, Hermes and Pi agents are PARKED by the owner (HOLD at the top of `/home/dan/supervisor/CLAUDE.md`):
  do not run `supervisor/start.sh`. A new window needs the owner's word; then B70 + B65 only, every job under
  `tools/xemaya/safety_guard.sh`, preflight, the lease (`window4_qwen.sh` shows the form).
- **DeepSeek-V4.1 is the main line.** Done and pushed: `src/reference/ds41_host.cpp` (the whole model on the host, the
  oracle), `src/core/plain_ds41.cpp` (the block in PlainModel), the state kernels in `ds41.hpp`, IQ1_M down rows.
  Checked: see the table below. **Next, in order:**
  1. (done 2026-11-10 00:55 UTC) the 576-token prompt on the OpenCL CPU device passes: the engine's top-k and
     candidate blocks agree with llama.cpp (table below; `logs/ds41-cpu/run-long.sh mid`);
  2. in a GPU window (`tools/xemaya/window5_ds41.sh`, prepared, refuses without an open window): `ds41_parity` again, then the Coder file on B70 + B65 forced along
     `xemaya/reference/ds41-coder-{short,mid}.json` (`--spec-follow`, `--dump-logits`, `compare_ref.py --forced`),
     then prefill / decode / TTFT / VRAM residency; GLM's forced check again (the checkpoint loop was regrouped);
  3. the chunked prompt path (the prompt runs in decode windows of 8 now: correct, slow), then the full 384-expert
     model, the DeepSeek chat template and DSML tool calls in serve.
- **Qwen3.8:** benchmarked (below). XeMaya equals XeStrata 0.2.2 and trails upstream Strata v0.2.12 by 17% in prefill.
  The lever found: `STRATA_PREFILL_MMQ=0` (+7%); making it the measured default for 8K chunks is open work.
- Tools of this window: `tools/xemaya/{qwen_bench.py, make_qwen_prompts.py, window4_qwen.sh, qwen_report.py,
  ref_ds41.sh}`; raw results `xemaya/benchmarks/2026-11-09-window4-qwen/` and `/DATA/Projects/xemaya/logs/window4/`.

## GPU window 5: DeepSeek-V4.1-Flash's first runs on Arc (2026-11-10 03:35-03:54 UTC)

Owner-authorized (60 minutes, deadline guard at 04:26); B70 + B65 only (`level_zero:0,1`), every job under the lease and
`safety_guard.sh`; no kernel message (NIC watchdog, engine reset, AER, OOM) during the window. Closed with
`xemaya-window-end.sh` at 03:54 (Qwen loaded after 54 s, lease `qwen38-strata-iq3s`); the agents stayed parked. Logs:
`/DATA/Projects/xemaya/logs/window5/`. Engine: `build-engine` (icpx 2026.2), `--layer-split auto` (B70 blocks 0-19, B65
20-39; 20 MoE layers each, 172 and 156 slots a layer of 192), `--mmap-experts`, `--spec 0`, greedy.

**Correctness** (Coder GSQ-RCO IQ2_XXS, forced along the llama.cpp v0.8.0 CPU references):

| Check | Result |
|---|---|
| `ds41_parity`, `glm5_parity` on the B70 and the B65 | PASS |
| 576-token prompt, 16 steps | top-1 equal at **16 of 16**, median \|d logprob\| 0.015, max 0.076 |
| 24-token prompt, 16 steps | top-1 equal at **15 of 16**: step 3 flips at a reference margin of 0.211 (':' -0.962 in the reference, -1.136 here), which the script's near-tie rule does not excuse; median 0.002, max 0.175. The OpenCL CPU device took the reference's token there (-1.096) |
| GLM-5.3 Maya-S v2 forced along chat-agent (the checkpoint loop was regrouped for DeepSeek) | 126 of 128, median 0.012, max 0.395: window 3's count; no regression seen |
| 2,876-token prompt (coder-mid's text five times) + 64 tokens | runs to the end, no fault; not compared with a reference |
| The full 384-expert model (GSQ-RCO IQ2_XXS), 24-token prompt + 48 tokens | loads (9.1 s) and generates; **not compared with a reference** (there is none for this file yet) |

**Speed** (Coder; the prompt runs in decode windows of 8 tokens, there is no chunked path yet):

| Run | Prefill tok/s | TTFT s | Decode tok/s | Experts hit in VRAM |
|---|---|---|---|---|
| 576-token prompt, first after a load (cold tier, the files read from the drive: 234 GB) | 10.8 | 53.4 | 14.7 | 81.3% |
| 576 tokens of other order, same process (warm) | **48.5-48.9** | 11.8 | **16.3** | 99.0% |
| 24-token prompt, warm | 21.1 | 1.14 | 15.8 | 97.0% |
| 2,876-token prompt, one process from cold | 26.3 (whole prompt) | 109 | 15.9 | 95.2% |
| Full 384-expert model, 24 tokens, cold (585 of 6,540 slots filled) | 2.2 | 11.1 | 3.0 | 28.0% |

VRAM while loaded: B70 31.6 GB, B65 31.5 GB. Load 6 s (Coder). Decode does not slow with context (15.9 tok/s at 2.9K).

**Where the time goes** (`STRATA_WINDOW_TIMES`; a token or window is 40 MoE rings):

- Decode, one token a window: the GPU reaches a ring in about 0.76 ms; a warm token takes 61 ms, so about half of it is
  the host's answer and the round trip of each of the 40 rings, as GLM's 42.
- Prompt windows of 8: the GPU's part is 1.5-3.9 ms a ring (rising with the context), 60-150 ms a window, which alone
  caps the windowed prompt near 50 tok/s. **The chunked prompt path is the prefill lever**, not tuning the windows.
- Cold: the first prompt after a load reads 234 GB from the model's files in 53 s (the load drops them from the page
  cache; the tier's slots are filled a window at a time, 5,592 swaps), with the host answering a ring in 22 ms.
  Filling the tier at load from a saved profile (`--profile`, as Qwen's) is the cold-TTFT lever.

**Since the window (CPU only, 2026-11-10 04:05 UTC): the chunked prompt path is written** (`src/prefill/plain_prompt_ds41.cpp`,
chunks of up to 1,024 tokens; `XEMAYA_DS41_PROMPT=windows` gives the old path, `XEMAYA_DS41_CHUNK=N` the chunk). On the
OpenCL CPU device, one stage, forced along the llama.cpp references: 24-token prompt 16 of 16 (median 0.007, max 0.073),
576-token prompt 16 of 16 (median 0.015, max 0.099), so the chunk's compressor, candidate blocks, top-k and Engram
agree with the windows'. **Unverified: on an Arc GPU, and the two-GPU hand-off of the streams and the lagging pre**
(the CPU device runs one stage). The CPU device's profile of 576 tokens (not GPU timings): the experts 60%, the
attention and its output 23%, the projections 7%. For the CPU device `event_record` now marks with a barrier where a
device has no profiling tags.

**Next:** a GPU window for the chunked path (forced checks on B70 + B65, then prefill speed and its profile); a tier profile at
load; a llama.cpp reference for the full model and longer prompts; then decode (fewer host round trips).

## GPU window 4: Qwen3.8 Flash-Next on three engines, DeepSeek-V4.1 (2026-11-09 21:28-22:50 UTC)

Owner-approved; B70 + B65 only (`ONEAPI_DEVICE_SELECTOR=level_zero:0,1`), no B60, every job under the lease and
`safety_guard.sh` (no trip; no NIC, xe or AER message in the kernel log). Ended with `xemaya-window-end.sh`.

**Qwen3.8 Flash-Next ISTA GSQ-RCO IQ3_S**, the production profile's arguments (`qwen38-ista-iq3s.json`: layer split
26, int8 KV, 8192-token prompt chunks, `--pipeline-windows 1`, context 65,536 here, vision off), one load per run,
the same token ids to every engine (`prompts/qwen`: 26,800-token prompts of source files with a code or a prose
request, 256 greedy tokens). XeStrata and XeMaya do not have upstream's `--stream-experts`, `--trim-stage-weights`
and `--vram-reserve-mib`: they ran with `--mmap-experts` (every expert was in VRAM in all runs: 100% hits).

| Engine | 26.8K prefill tok/s | TTFT | Code decode tok/s | Prose decode tok/s | Short prompt code / prose |
|---|---|---|---|---|---|
| Upstream Strata v0.2.12 (engine 0.1.58 SYCL, the production binary), `--spec 4` | 2,211 / 2,216 | 12.1 s | 78.3 (85% of drafts kept) | 59.2 (56%) | 83.9 / 58.7 |
| XeStrata 0.2.2 (tag `xe0.2.2-1`, built here with icpx 2026.2), `--spec 4` | 1,859 / 1,876 | 14.4 s | 76.5 (83%) | 57.3 (53%) | 80.4 / 61.4 |
| **XeMaya** (`f85353c17`), `--spec 4` | 1,842 / 1,884 | 14.6 s | 76.6 (83%) | 57.1 (53%) | 80.4 / 61.4 |
| **XeMaya**, `--spec 4`, `STRATA_PREFILL_MMQ=0` | **1,989 / 1,996** | 13.5 s | 70.1 (76%) | 60.9 (63%) | 80.2 / 61.7 |
| Upstream Strata v0.2.12, `--spec 2` | 2,188 / 2,181 | 12.3 s | 67.3 | 58.1 | 68.8 / 59.5 |
| XeStrata 0.2.2, `--spec 2` | 1,859 / 1,887 | 14.4 s | 67.6 | 59.3 | 68.9 / 62.0 |
| XeMaya, `--spec 2` | 1,832 / 1,886 | 14.7 s | 67.7 | 59.1 | 68.9 / 61.9 |

- **Against the targets** (XeStrata's historical 1,939 / 55.7 / 77.3 on B70 + B65, other prompts): prefill 1,989-1,996
  with `STRATA_PREFILL_MMQ=0` (beats it; 1,842-1,884 by default does not), prose 57-61 (beats it), code 76.6 after a
  long prompt and 80.4 after a short one (the historical figure's prompt is unknown). In this controlled A/B XeMaya
  does **not** beat upstream Strata v0.2.12's prefill (2,211) or its code decode after a long prompt (78.3).
- **Drafting cannot be turned off** for a native IQ pack on any of the three (`--spec T` needs T >= 2); `--spec 2` is
  the least drafting. Agent turns: a follow-up to a 26.8K conversation starts in 0.23-0.27 s on all three (whole
  prefix reused); a second request sharing 26.7K of prefix reuses 16,384 tokens (two chunks) on all three.
- **VRAM:** Strata 30.9 / 30.5 GiB (B70 / B65), XeStrata and XeMaya 28.5 / 27.9 GiB. Load: 10 s, 13-63 s, 15-19 s.
- **Where XeMaya loses the prefill:** the routed experts' products. Upstream's timed run spends 453 ms dequantizing
  and 2.4 s in the experts' matrix products of 26.8K tokens; XeMaya's timed run (its timing adds syncs: 1,138 tok/s)
  spends 7.1 s and 3.9 s there. The measured int8 / FP16 choice (int8 below 362 rows an expert on the B70, 181 on
  the B65) loses to FP16 everywhere at 8K chunks (+7%); 4K and 16K chunks do not help (1,844-1,872, 1,783-1,803).
- **Where it loses the decode:** the verify window costs the same (32.09 ms on both); XeMaya's draft step is 4.4 ms
  against 3.1 ms. Fixed on the way: the 2-bit draft head was never built for this file's Q6_K head (the engine's
  split row order had no host dequantizer, `f85353c17`); built, it measured slower here (214 vs 185 us a step) and
  the engine keeps the whole head, so the decode did not change. The remaining gap is the draft layer itself.
- The router's label for production still says `strata-0.2.11-sycl`; the installed release is v0.2.12
  (`DEPLOYMENT-v0.2.12.json`, engine sha256 223c2440...). Not changed (the router's config is the owner's).

**DeepSeek-V4.1-Flash** (ISTA-DASLab GSQ-RCO IQ2_XXS; Coder: 192 experts, 68 GB; full: 384 experts, 138 GB; Engram
table 110 GB IQ4_NL):

| Check | Result |
|---|---|
| `ds41_host` (the whole model on the host in F32) forced along llama.cpp v0.8.0's CPU reference, Coder, 24-token prompt | top-1 equal 16 of 16, median \|d logprob\| 0.003, max 0.110 |
| the same, 576-token prompt (compressed rows past the top-k of 512, candidate blocks) | 16 of 16, median 0.011, max 0.043 |
| `ds41_parity` with the new state kernels (ring store, block positions, streaming compressor, cache store), B70 and B65 | PASS on both |
| The engine's block (`plain_ds41.cpp`) on the OpenCL CPU device, Coder, 24-token prompt, forced | 16 of 16, median 0.007, max 0.134 |
| The engine's block, 576-token prompt (compressed rows past the top-k, candidate blocks), CPU device | 16 of 16, median 0.009, max 0.062 |
| On an Arc GPU | **not run yet** (the block was finished after the window closed) |

Tensor types of the Coder file and the engine's kernels: Q8_0 and Q5_K projections (native_mmvq), BF16 compressor,
router and indexer weights (widened at upload), F32 mixers, IQ1_M / IQ2_XS experts (IQ1_M down rows added), IQ4_NL
Engram rows (dequantized on the host). The full model's reference and the DSML serve path are not done.

**GLM-5.3:** no GLM run in this window; the kernels are unchanged. `plain_model.cpp`'s recurrent checkpoint loop was
regrouped for the DeepSeek state: GLM's forced check must be repeated in the next window before any claim.

## The 18:46 UTC reset of 2026-11-09: what the logs show (read-only review)

- **It was not an orderly reboot and not a kernel crash that left a trace.** The previous boot's journal ends at
  18:45:24 in the middle of normal messages (containers starting, firewall lines), with no shutdown sequence; the new
  boot starts at 18:46:01 and finds the journal "corrupted or uncleanly shut down". The firmware's reset reason is
  `0x00200800: ACPI power state transition occurred` (the two clean reboots before it show `0x00080800: software
  wrote 0x6 to reset control register 0xCF9`). kdump was armed and left no dump; no panic, oops, MCE, thermal, OOM,
  hung-task, xe, NVMe or AER message in the last 16 minutes; the watchdog-reset and thermal bits are not set.
  That pattern fits a power-off and power-on from outside the OS (a held power button or a power cycle); the logs
  cannot say who or what did it.
- **No XeMaya command asked for a reboot.** None of its scripts contains one, and no reboot, poweroff or sysrq
  appears in the journal. XeMaya's last job (`window3.sh`) had finished and unloaded at 18:41:46, 3.6 minutes before
  the reset; the GPUs were idle.
- **The network:** `eno1` (RTL8126, r8169) logged a transmit timeout at 17:49:58 and at 18:29:25 (during the 64K and
  128K runs). It **recovered** after 18:29: SSH logins over Tailscale succeeded at 18:31:48 and 18:32:35. The final
  loss began at 18:37:09 (DNS queries failing even to the gateway 192.168.1.1, Tailscale without UDP, then "network
  is unreachable"), with **no** kernel message and no link change, while XeMaya's agent session was running with the
  B60 loaded (its copy engine busy in bursts) and about 30 s before XeMaya reloaded the model. The earlier STATUS text
  ("clean shutdown 18:45:12", "unreachable from 18:29") was wrong on both points and is corrected here.
- **What links it to XeMaya:** timing only: three network failures in one evening, all while XeMaya used the B60
  behind the chipset, none in the four days before or in window 4 (B70 + B65 only, 80 minutes of GPU load, no
  event). **Not established:** the mechanism (chipset uplink contention is a hypothesis; a NIC firmware or driver
  stall, or something on the LAN side, would look the same from these logs), and why the third failure logged
  nothing.
- **Mitigations and what they do not cover:** the B60 is out of auto splits, every GPU job runs under
  `safety_guard.sh` (it would have stopped the job at both logged timeouts, and at the third through its gateway
  ping), engines are not SIGKILLed. Not validated: that the B60 is the cause (that needs a deliberate, owner-run
  test), and the guard cannot restore a NIC that has already stalled.

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

1. A GPU window for DeepSeek-V4.1's first run on Arc (B70 + B65, the safety guard, the Coder file; bounded: parity,
   two forced prompts, one speed run) and GLM's forced re-check.
2. The NIC (r8169 / RTL8126) and whether to test the B60 hypothesis deliberately: your call.
3. The router's stale `strata-0.2.11-sycl` label for the v0.2.12 production engine: your config.

## Notes

- `gh` on the Windows PC is not logged in. `gh` on Zability is (Jumbomuffin777) and is what this work used.
- There is no passwordless sudo on Zability.
- Upstream history includes a 68 MB `src2.tar`. GitHub warns but the push succeeds; history is not rewritten.
- Commit identity is repo-local: `Jumbomuffin777 <ID+Jumbomuffin777@users.noreply.github.com>`.
