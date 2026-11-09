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

> **Status: early development.** XeMaya builds, its kernels pass parity tests, and it reads the GLM and DeepSeek model
> files, but GLM-5.3-Flash and DeepSeek-V4.1-Flash do not run end to end on an Arc GPU yet. Qwen3.8-Flash-Next support
> is inherited from XeStrata. See [Status](#status) and the [roadmap](#roadmap).

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
| **GLM-5.3-Flash** | 320B MoE (18B active) | Project Maya GGUFs (Maya-S v2, S24, XS, M, L) | Kernels pass parity; model files read correctly; first GPU run in progress |
| **DeepSeek-V4.1-Flash** | 748B MoE | ISTA-DASLab GSQ-RCO and Coder, Maya-XS / S | Model-specific kernels pass parity; tensor manifest verified on real files; model block not written yet |

## Status

Everything below was checked on the development machine (Ubuntu 26.04, Intel Arc Pro B70 32 GB, B65 32 GB and
B60 24 GB, Intel oneAPI `icpx` 2026.2).

| Area | Result |
| --- | --- |
| Build | Core, tests, parity binaries and the full engine (GLM and DeepSeek kernels included) build with 0 errors |
| Host tests | XeStrata's 13 host tests, plus `model_spec_test` and `ds41_manifest_test` |
| GLM-5.3-Flash kernels | `glm5_parity`: 28/28 on the CPU device and on each Arc card (B70, B65, B60) |
| GLM Maya GGUFs | Maya-S v2 IQ2_XXS reads as `glm5-next`. Its F32 mHC mixer weights, which stopped the first GPU load, are now taken in F32 (checked on the CPU device against a double-precision reference); not yet loaded on a GPU |
| llama.cpp reference | 16 greedy tokens of Maya-S v2 from llama.cpp v0.8.0 on the CPU, with each step's top-5, saved for the GPU comparison |
| DeepSeek-V4.1 model files | All 1043 tensors of both ISTA-DASLab models match llama.cpp's shapes |
| DeepSeek-V4.1 kernels | `ds41_parity`: 18/18 on the CPU device and on each Arc card (router, mHC, compressor, Engram gate and hash, YaRN RoPE, indexer, top-k, sink attention) |
| i-quant kernels | `iq_parity`: 0 failures over 10 formats (IQ1_M to IQ4_XS, Q2_0, Q3_K) on the CPU device, on rows of a real model |
| Element-wise kernels | `elementwise_parity`: 0 failures on each Arc card |
| Server | 14 newer Strata tool-call fixes ported; serve tests 930 passed, 0 failed |

The GLM and DeepSeek kernel tests passed on the Arc cards in a GPU window on 2026-11-09. Not verified yet: no GLM or
DeepSeek model has generated on a GPU (Maya-S v2 is ready for its first GPU run), and there is no llama.cpp
reference output for Maya-S v2 yet. Detailed notes: [xemaya/STATUS.md](xemaya/STATUS.md).

## Roadmap

1. **Foundation**: fork, licences, Intel oneAPI build, kernel tests. *Done.*
2. **Catch up with Strata** where it matters for Intel: server correctness (tool calls done; request lifecycle next).
3. **GLM-5.3-Flash on Arc**: load Maya-S v2 across several GPUs, match llama.cpp token for token, then tune.
4. **DeepSeek-V4.1-Flash on Arc**: the model block on XeStrata's PlainModel (mHC, Engram, expert tiers, multi-GPU).
5. **Production quality**: tool calling, long-run stability, and prefill/decode performance against the baselines below.

The design decision and the plan for each step: [xemaya/PLAN.md](xemaya/PLAN.md).

### Baselines to beat

Earlier runs of the same models on the same machine through llama.cpp's SYCL backend:

| Model | Prefill | Decode |
| --- | ---: | ---: |
| DeepSeek-V4.1 Maya-XS (with a custom loader patch) | 45.6 tok/s | 10.9 tok/s |
| GLM-5.3 Maya-S v2 (stock llama.cpp) | 68.3 tok/s | 13.6 tok/s |
| GLM-5.3 Maya-S v2 (experimental grouped SYCL matmul) | 108.9 tok/s | |

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
