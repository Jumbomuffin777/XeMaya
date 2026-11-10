<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeMaya

**An Intel Arc-first, native inference engine for today's large open mixture-of-experts models:
Qwen3.8-Flash-Next, GLM-5.3-Flash and DeepSeek-V4.1-Flash.**

XeMaya runs these models directly on Intel Arc GPUs through SYCL and Level Zero, with kernels written for Intel's
hardware (Xe2 / Battlemage first) rather than translated from CUDA. It is not a llama.cpp wrapper and not a CUDA
compatibility layer. The goal is an open, efficient engine that gets everything Arc GPUs can give, on one card or
several, with performance competitive with native CUDA/HIP engines wherever the hardware allows.

> **Status: early development.** GLM-5.3-Flash and DeepSeek-V4.1-Flash run natively on Arc GPUs and agree with
> llama.cpp on the checked prompts, but neither is production-ready (few prompts checked, DeepSeek's prompt path is
> slow, no API soak). Qwen3.8-Flash-Next support is inherited from XeStrata. See [Status](#status) and the [roadmap](#roadmap).

## Where XeMaya comes from

XeMaya stands on three projects and keeps their licences and credit:

| Project | What XeMaya takes from it |
| --- | --- |
| **[XeStrata](https://github.com/MistVVK/XeStrata)** by MistVVK (LGPL-3.0-or-later) | The foundation. XeStrata rewrote Strata's GPU code by hand for SYCL and Level Zero: sub-group and XMX-aware Intel kernels, multi-GPU layer split, VRAM/RAM/SSD expert tiers, and an experimental native GLM-5.3-Flash. XeMaya is a fork of it. |
| **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 (MIT) | The engine XeStrata is a port of. XeMaya brings over newer Strata fixes selectively (tool calling and the request lifecycle in the server) and uses Strata's DeepSeek-V4.1 engine as a design reference. |
| **[Project Maya](https://github.com/mw00/project-maya)** by mw00 (MIT) | GLM-5.3-Flash and DeepSeek-V4.1-Flash on CUDA/HIP, and the **Maya GGUF quantisations** XeMaya targets (GLM Maya-S / S24 / XS / M / L, DeepSeek Maya-XS / S). XeMaya is in effect "Project Maya's models, natively on Intel". |

[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) serves as the exact reference for each model's graph and as
the correctness oracle in tests; it is not XeMaya's backend.

## Models

| Model | Size | Formats | State in XeMaya |
| --- | --- | --- | --- |
| **Qwen3.8-Flash-Next** | 125B MoE | XeStrata's GGUFs (ISTA-DASLab GSQ-RCO, Swift 1.5, Coder, Gyro, Unsloth) | Runs (inherited from XeStrata) |
| **GLM-5.3-Flash** | 320B MoE (18B active) | Project Maya GGUFs (Maya-S v2, S24, XS, M, L) | Runs on Arc (B70 + B65 + B60); checked against llama.cpp on four prompts; tuning |
| **DeepSeek-V4.1-Flash** | 748B MoE | ISTA-DASLab GSQ-RCO and Coder, Maya-XS / S | Runs on Arc (B70 + B65): the Coder file checked against llama.cpp on two prompts; the prompt path is not chunked yet (slow prefill) |

## Status

Everything below was checked on the development machine (Ubuntu 26.04, Intel Arc Pro B70 32 GB, B65 32 GB and
B60 24 GB, Intel oneAPI `icpx` 2026.2).

| Area | Result |
| --- | --- |
| Build | Core, tests, parity binaries and the full engine (GLM and DeepSeek kernels included) build with 0 errors |
| Host tests | XeStrata's 13 host tests, plus `model_spec_test` and `ds41_manifest_test` |
| GLM-5.3-Flash kernels | `glm5_parity` (with the F32 mixer checks) on the CPU device and on each Arc card (B70, B65, B60) |
| GLM-5.3-Flash on Arc | Maya-S v2 IQ2_XXS runs natively on B70 + B65 + B60. Its first 16 greedy tokens match llama.cpp's (16 of 16 top-1 when forced along them). 2,377-token prompt: prefill 93-100 tok/s (stock llama.cpp SYCL 68.3), decode 13.9 tok/s (13.6), 16.0 with MTP drafts; 512 tokens after a short prompt: 17.3, 22.9 with drafts. Not production-ready: one prompt checked, 4K context, no API soak |
| llama.cpp reference | 16 greedy tokens of Maya-S v2 from llama.cpp v0.8.0 on the CPU, with each step's top-5, saved for the GPU comparison |
| DeepSeek-V4.1 model files | All 1043 tensors of both ISTA-DASLab models match llama.cpp's shapes |
| DeepSeek-V4.1 kernels | `ds41_parity`: 18/18 on the CPU device and on each Arc card (router, mHC, compressor, Engram gate and hash, YaRN RoPE, indexer, top-k, sink attention) |
| DeepSeek-V4.1-Flash on Arc | The Coder GSQ-RCO IQ2_XXS file runs natively on B70 + B65 (2026-11-10). Forced along llama.cpp's CPU reference: 16 of 16 top-1 on a 576-token prompt, 15 of 16 on a 24-token prompt (one flip at a 0.21 margin). Warm: prefill 48.5-48.9 tok/s (the prompt still runs in decode windows), decode 16.3 tok/s; first prompt after a load 10.8 tok/s. A 2,876-token prompt runs (decode 15.9 tok/s). The full 384-expert file loads and generates, not yet compared with a reference |
| i-quant kernels | `iq_parity`: 0 failures over 10 formats (IQ1_M to IQ4_XS, Q2_0, Q3_K) on the CPU device and each Arc card |
| Element-wise kernels | `elementwise_parity`: 0 failures on each Arc card |
| Server | 14 newer Strata tool-call fixes ported; serve tests 930 passed, 0 failed |

GPU window 4 (2026-11-09): on Qwen3.8 Flash-Next (B70 + B65, the same prompts) XeMaya reads a 26.8K-token prompt at 1,842-1,884 tok/s (1,989-1,996 with STRATA_PREFILL_MMQ=0), as XeStrata 0.2.2 does, behind upstream Strata v0.2.12's 2,211; decode 76.6 / 57.1 tok/s (code / prose) against 78.3 / 59.2. DeepSeek-V4.1-Flash: the whole model runs on the host as a reference (16 of 16 tokens against llama.cpp) and the engine's block passes its first forced check on the CPU device; it has not run on an Arc GPU yet. GPU window 3 (2026-11-09): with the attention on Intel's XMX matrix engines, a 2,377-token prompt reads at 152 tok/s (TTFT 15.6 s; stock llama.cpp SYCL 68.3, grouped 108.9) and decodes at 17.4 tok/s (20.0 with MTP drafts); prefill holds ~205 tok/s from 8K to 128K; a coding-agent session through the OpenAI API made valid tool calls and reused the conversation every turn. Not production-ready. The window ended with the host losing its network under XeMaya's heaviest load; the cause and the safeguards are in xemaya/STATUS.md. Detailed notes: [xemaya/STATUS.md](xemaya/STATUS.md).

## Roadmap

1. **Foundation**: fork, licences, Intel oneAPI build, kernel tests. *Done.*
2. **Catch up with Strata** where it matters for Intel: server correctness (tool calls done; request lifecycle next).
3. **GLM-5.3-Flash on Arc**: load Maya-S v2 across several GPUs, match llama.cpp token for token, then tune. *Running and matching (2026-11-09); tuning: the DSA attention kernel next.*
4. **DeepSeek-V4.1-Flash on Arc**: the model block on XeStrata's PlainModel (mHC, Engram, expert tiers, multi-GPU).
5. **Production quality**: tool calling, long-run stability, and prefill/decode performance against the baselines below.

The design decision and the plan for each step: [xemaya/PLAN.md](xemaya/PLAN.md).

### Baselines to beat

Earlier runs of the same models on the same machine through llama.cpp's SYCL backend:

| Model | Prefill | Decode |
| --- | ---: | ---: |
| DeepSeek-V4.1 Maya-XS (with a custom loader patch) | 45.6 tok/s | 10.9 tok/s |
| GLM-5.3 Maya-S v2 (stock llama.cpp) | 68.3 tok/s | 13.6 tok/s |
| GLM-5.3 Maya-S v2 (experimental grouped SYCL matmul) | 108.9 tok/s | 14.0 tok/s |
| *XeMaya, GLM-5.3 Maya-S v2, 3 GPUs, same prompt length (window 3, XMX attention)* | *152 tok/s* | *17.4 tok/s; 20.0 with MTP drafts* |

These are reference points, not like-for-like targets.

## Building

XeMaya is Intel-first and builds with Intel oneAPI's `icpx` (XeStrata's intel/llvm DPC++ toolchain still works).

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S XeMaya -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DSTRATA_BUILD_TESTS=ON
ninja -C build strata            # the engine
ninja -C build glm5_parity ds41_parity model_spec_test
```

Kernel parity on the CPU device (no GPU needed):
`ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 build/glm5_parity` (and the same for `ds41_parity`).

Installing and running Qwen3.8-Flash-Next works as in XeStrata (the command is still `xestrata` for now):
see [docs/XESTRATA.md](docs/XESTRATA.md), [docs/BUILD.md](docs/BUILD.md) and [docs/DETAILS.md](docs/DETAILS.md).
The server speaks the OpenAI and Anthropic APIs, so any compatible client or coding agent can use it.

## Contributing

Issues and pull requests are welcome, especially from people with Intel Arc hardware (Alchemist, Battlemage or Arc Pro).
Kernel work comes with a parity test against a host reference, and model work with a token-level comparison against
llama.cpp on the same file. See [CONTRIBUTING.md](CONTRIBUTING.md).

XeStrata's documentation is also available in Japanese in the [XeStrata repository](https://github.com/MistVVK/XeStrata).

## Licence

XeMaya as a whole is under the **GNU LGPL, version 3 or later**, like XeStrata ([COPYING.LESSER](COPYING.LESSER),
[COPYING](COPYING)). Code from Strata, Project Maya, llama.cpp and other projects keeps its own licence (mostly MIT),
file by file. The full list of copyright holders and licences is in [NOTICE](NOTICE) and [LICENSES/](LICENSES).
Each model's own licence applies to its files.

## Thanks

To **Niko1221** and the Strata contributors for the engine, to **MistVVK** and the XeStrata contributors for the
Intel-native rewrite this project is built on, to **mw00** and the Project Maya contributors for the GLM and DeepSeek
work and the Maya quantisations, and to the ggml / llama.cpp authors for the reference implementations.
