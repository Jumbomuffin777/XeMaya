# XeMaya plan

XeMaya is an Intel Arc-first native inference engine for GLM-5.3-Flash, DeepSeek-V4.1-Flash and Qwen3.8-Flash-Next.
It is not a CUDA compatibility layer and not a llama.cpp wrapper.

## Architecture decision (2026-11-09)

**Foundation: XeStrata (MistVVK/XeStrata, LGPL-3.0-or-later AND MIT), branch `xemaya/main`.**

Inspected sources and what they offered:

| | XeStrata | Strata v0.2.12 (Niko1221) | Project Maya (mw00) |
|---|---|---|---|
| Intel GPU code | Hand-written SYCL / Level Zero kernels (`src/kernels/xe`, ~400 files): XMX through joint_matrix, DP4a fallbacks, device-reported tile shapes | `sycl/`: SYCLomatic-migrated CUDA with manual fixes, kept as a side tree beside the CUDA engine | None (CUDA/HIP only) |
| Models | Qwen3.8 (main), PlainModel: GLM-5.3-Flash (`glm5-next`, also Maya's `glm5next` spelling), Ornith, llm-jp, Qwen3.6, Qwen3.5 | Qwen3.8; DeepSeek-V4.1 engine `src/dsv41` (CUDA/HIP, ~2.1 MB of source, 858 KB in `ds4_model.cu` alone) | GLM engine (~21K lines CUDA), DeepSeek `src/ds4` (CUDA/HIP) |
| Multi-GPU | Layer split over several GPUs, mixed makers | Layer split (CUDA) and SYCL port | CUDA |
| Experts | VRAM cache + RAM tier + SSD (direct I/O), per-layer cache, CPU AVX-512 expert lane | Same lineage, newer CUDA ideas | Hot/cold tiers |
| Status vs upstream | Synced to Strata v0.2.8 (merge `721afe51f`) | v0.2.12 | - |

Why XeStrata:

1. Its kernels are written for Intel GPUs (sub-group and XMX aware) rather than translated from CUDA. The earlier
   attempt on this host to SYCLomatic-port Maya's DeepSeek engine (`/DATA/AppData/maya-upstream/native-port/AUDIT.md`)
   stalled on cross-device pointer faults that come from dpct's one-context-per-device model.
2. It already runs GLM-5.3-Flash natively. Its `glm5_parity` test passes on XeMaya's build (28/28 checks; run on the CPU
   device, see STATUS).
3. Its PlainModel is the right base for DeepSeek-V4.1: the mHC hyper-connections, clamped SwiGLU, sigmoid-family
   routers, expert tiers, multi-GPU layer split and the PLE n-gram SSD reader that Engram needs are already there.
4. It dropped CUDA, Metal and the desktop GUI, so it is a much smaller surface.

How the other sources are used:

- **Strata**: `serve/` fixes (tool calls, request lifecycle) are ported commit by commit as 3-way patches restricted to
  `serve/`, and kept only where the serve test suite stays green. `src/dsv41` is a design reference for DeepSeek
  streaming, Engram I/O and expert rules, not code to translate.
- **llama.cpp** (`src/models/deepseek41.cpp`, `deepseek4.cpp` at master 8642541b2; local copy
  `/DATA/AppData/maya-upstream/llama.cpp-master-20261029`): the exact reference graph. XeStrata's GLM port is transcribed
  from llama.cpp's `glm5-next.cpp` the same way.
- **Project Maya**: its GGUF quantisations (GLM Maya-S/S24/XS/M/L, DeepSeek Maya-XS/S) and ideas.

Toolchain: Intel oneAPI `icpx` 2026.2 (libsycl 9) with `-ffp-model=precise` (`XEMAYA_ALLOW_ICPX`, default ON). XeStrata
removed icpx on 2026-10-15 to keep one compiler for all GPU makers; XeMaya is Intel-first and the hosts already have
oneAPI. intel/llvm DPC++ 7+ still works.

## Milestones

| # | Milestone | State |
|---|---|---|
| 1 | Foundation: repo, history, licences, icpx build, CPU-device kernel testing | done |
| 2 | Upstream Strata v0.2.8 -> v0.2.12: `serve/` correctness (tool calls, request lifecycle, Anthropic route) | in progress |
| 3 | Native GLM-5.3-Flash: Maya-S v2 (IQ2_XXS) on the Arc cards, output checked against llama.cpp | blocked on GPU access (see STATUS) |
| 4 | Native DeepSeek-V4.1-Flash: PlainModel `deepseek41` block, SYCL kernels with CPU-device parity tests | started (model spec) |
| 5 | Performance and reliability: decode/prefill, expert residency, PCIe, multi-GPU scheduling, API soak | not started |

### Milestone 3: GLM-5.3-Flash

1. Done: the kernels pass `glm5_parity` with icpx. The engine builds. Maya-S v2 uses only types XeStrata has SYCL and CPU
   kernels for (Q6_K, Q8_0, IQ2_XXS, IQ2_S, IQ3_XXS, F32) under arch `glm5-next`.
2. Model: Maya-S v2 IQ2_XXS (89.8 GiB) is being downloaded to
   `/mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/` (sha256-checked).
3. Needs GPU (approval): load on B70+B65+B60, then a greedy 16-token check against llama.cpp SYCL on the same file. The
   historical reference is 68.3 / 13.6 tok/s on stock llama.cpp v0.8.0.
4. Then the serve path: OpenAI and Anthropic routes, tool calls, streaming.

### Milestone 4: DeepSeek-V4.1-Flash

Target files on disk: ISTA-DASLab GSQ-RCO IQ2_XXS (+ Coder) with Engram IQ4_NL (`/mnt/ai/models/ISTA-DASLab/`). Tensor
types are Q8_0, Q5_K, BF16, IQ1_M, IQ2_XXS and F32.

Steps (each with a CPU-device parity test against a host reference transcribed from llama.cpp):

1. Done: `deepseek41` model spec with llama.cpp's checks (compress ratios, KV/top-k sources, candidate blocks, Engram).
2. Kernels, in `src/kernels/xe/ds41.cpp` + `ds41_parity`: sqrt-softplus router with bias; the KV compressor (ratio 1/2,
   gated); the lightning indexer and candidate blocks; MQA with attention sinks over window + selected compressed rows;
   grouped low-rank output; YaRN RoPE on 64 of 512 dims, plus the compressed RoPE base. Reuse `glm5` mHC and
   `swiglu_clamp`.
3. Engram: the n-gram hash (llama.cpp `engram_index` / qwen4exp PLE) on the PLE reader (`src/ngram`), streamed from
   the 110 GB IQ4_NL table by direct I/O.
4. `plain_deepseek41.cpp`: the block in PlainModel, decode first, then prompt chunks.
5. CPU expert lane: IQ1_M is missing from `src/kernels/cpu/native_expert.cpp`.
6. GPU validation (approval): greedy tokens against llama.cpp-ds41 SYCL on the same file.

### Milestone 2: upstream serve fixes

Port order (one commit each, serve tests green after each): tool-call parser (#4701, #4663, #4751, #5276, #5281,
#5409, #5040), request lifecycle (#4271, #4479, #4738, #5044, #4792, #3443, answers one request late), Anthropic context
overflow wording, fatal GPU error restart. Skipped: DFlash2, replicas/shared arena, CUDA/HIP-only defaults, GUI, Metal.
