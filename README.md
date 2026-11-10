<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeMaya

**Native inference for Intel Arc. Built on XeStrata. Targeting Project Maya's models.**

[![Status: experimental](https://img.shields.io/badge/status-experimental-orange)](xemaya/STATUS.md)
[![Backend: SYCL / Level Zero](https://img.shields.io/badge/backend-SYCL%20%2F%20Level%20Zero-blue)](docs/XE.md)
[![Licence: LGPL v3+](https://img.shields.io/badge/licence-LGPL--3.0--or--later-blue)](COPYING.LESSER)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Jumbomuffin777-FFDD00?logo=buymeacoffee&logoColor=000)](https://buymeacoffee.com/Jumbomuffin777)

[Models](#models) · [Recorded results](#status) · [Build](#building) ·
[Model discovery](#model-discovery-and-serving) · [Documentation](DOCUMENTATION.md) · [Changelog](CHANGELOG.md)

XeMaya is an Intel Arc-first inference engine for Qwen3.8-Flash-Next, GLM-5.3-Flash,
DeepSeek-V4.1-Flash and experimental MiniMax-M3.1-Flash. It runs native SYCL / Level Zero kernels,
including Intel XMX paths, with multi-GPU layer splitting and VRAM / RAM / SSD expert tiers.
The engine is built on XeStrata, rather than a llama.cpp server wrapper or a CUDA compatibility layer.

> **Early development — snapshot: 2026-11-10, source `b59488e2d`.** Reference agreement is limited
> to the models, prompts and hardware recorded below; it does not establish production readiness.
> MiniMax's RAM-copy run triggered a GPU engine reset; the cause is undetermined and development GPU
> work was stopped. See the latest [status handoff](xemaya/STATUS.md) before running development jobs.

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

| Model | Implemented path / files | Recorded qualification and limits |
| --- | --- | --- |
| **Qwen3.8-Flash-Next** | Inherited XeStrata path; `qwen4exp` GGUF architecture | Benchmarked on B70 + B65; tools, reasoning and drafting advertised by the catalog. Direct-tier changes made later still need a fresh GPU check. |
| **GLM-5.3-Flash** | `glm5-next`; setup lists Maya-S v2 / S24 / XS / M / L | Maya-S v2 checked on B70 + B65 + B60; 126/128 forced tokens on the chat-agent reference. A seven-turn tool session ran through the server. This is not validation of every listed quantisation. |
| **DeepSeek-V4.1-Flash** | `deepseek41`; compressed attention, Engram, chunked prefill and XMX attention | Coder GSQ-RCO IQ2_XXS: 16/16 forced tokens on both saved references on three GPUs. Full 384-expert file previously generated on two GPUs, without reference comparison. DeepSeek tool-call format, qualified thinking and drafting are absent from the catalog. |
| **MiniMax-M3.1-Flash** | `minimax-m3`; sparse attention and SwiGLU-OAI; Maya-S IQ2_XXS | Experimental: 32/32 forced tokens for a 211-token chat prompt; 28/32 for a 3,832-token prompt. About 2–4 tok/s decode in recorded runs. Tools, vision and DSpark drafting are not implemented/qualified; no end-to-end server qualification. |

These are specific checked files and paths, not a promise that arbitrary GGUFs or all model variants work.
The [catalog's family definitions](tools/xemaya/catalog.py) describe current advertised capabilities.

## Status

The following are **previously recorded results**, not tests rerun for this documentation update.
They used Intel oneAPI `icpx` 2026.2 on Linux and Arc Pro B70 (32 GB), B65 (32 GB) and B60 (24 GB).
See [STATUS.md](xemaya/STATUS.md) for dates, settings, mismatches, incidents and historical runs.

| Checked configuration | Prefill | Decode | Qualification |
| --- | ---: | ---: | --- |
| GLM-5.3-Flash Maya-S v2, three GPUs, XMX attention, 2,377-token prompt (window 3) | 152–161 tok/s | 17.4 tok/s; 20.0 with MTP | 126/128 on the separate chat-agent reference; other prompts have larger differences. |
| DeepSeek-V4.1-Flash Coder, three GPUs, direct expert tiers + XMX (window 6) | 207–209 tok/s at 576 tokens; 225 at 2,876 | 19.5–21.7 tok/s | 16/16 forced tokens on each of the 24- and 576-token references. |
| MiniMax-M3.1-Flash Maya-S, three GPUs, mmap experts | 51–60 tok/s on the longer warm prompt | 2.2–2.3 tok/s on that prompt | Short chat: 32/32; longer chat: 28/32. CPU was shared with reference jobs. |
| Qwen3.8-Flash-Next, B70 + B65, 26.8K prompt (window 4) | 1,842–1,884 tok/s; 1,989–1,996 with `STRATA_PREFILL_MMQ=0` | 76.6 / 57.1 tok/s (code / prose) | Behind Strata v0.2.12's 2,211 tok/s prefill and 78.3 / 59.2 decode in that comparison. |

GLM's earlier llama.cpp SYCL baselines were 68.3 / 13.6 tok/s (stock) and 108.9 / 14.0
(experimental grouped matmul), prefill / decode at 2,377 tokens. Results across different prompts,
files, GPU counts or cold/warm states are not interchangeable; no general speed superiority is claimed.

Recorded checks also include host tests, GLM / DeepSeek / MiniMax kernel parity, IQ-format parity,
and the server suite (930 passed, 23 skipped at the recorded checkpoint). Later source changes have not
all been rerun across every model, build mode or OS. Windows and free-mode qualification for the new
MiniMax path remain unverified.

## Building

From the XeMaya checkout on Linux, with Intel oneAPI available:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
  -DSTRATA_LICENSE=contrib -DXEMAYA_ALLOW_ICPX=ON \
  -DSTRATA_CUDA_ARCHS= -DSTRATA_HIP_ARCHS= \
  -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build --target strata glm5_parity ds41_parity minimax_parity model_spec_test --parallel 4
```

`CMakeLists.txt` accepts `icpx` through `XEMAYA_ALLOW_ICPX` (default ON); its build is Intel-only.
Native expert support uses ggml-cpu from the pinned source fetched by CMake, or a llama.cpp checkout
selected with `STRATA_GGML_DIR`. It does not use llama.cpp as the serving backend.

CPU-device kernel parity requires an installed OpenCL CPU device:

```bash
ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 build/glm5_parity
ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 build/ds41_parity
ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 build/minimax_parity
```

The inherited setup/packaging workflow still uses the `xestrata` name and intel/llvm toolchain;
`setup.py` does not adopt the direct CMake `icpx` workflow above. Read the inherited
[installation guide](docs/XESTRATA.md) and [build guide](docs/BUILD.md) with that distinction in mind.
These commands were checked against source, not executed on the development GPUs for this audit.

## Model discovery and serving

[`tools/xemaya/catalog.py`](tools/xemaya/catalog.py) scans local GGUF headers without loading a model
or opening a GPU. It recognises `qwen4exp`, `glm5-next`, `deepseek41` and `minimax-m3`, groups split files
from their first shard, and emits JSON with model files, trained/default context, capability flags,
qualification notes and engine arguments.

```bash
python3 tools/xemaya/catalog.py --root /path/to/models \
  --engine /path/to/build/strata --configs /path/to/server-configs --pretty
```

Add `--write --tokenizers` to extract missing GGUF tokenizers and write server configs.
Inspect `ready`, `missing`, `state` and `note` before using a result. `ready` checks prerequisites
(including model shards, engine and, when configs are requested, tokenizer); it is not a runtime test.
`verified` is a family-level record of limited reference checks, not certification of each discovered file.
The catalog caps its default context at 32,768 tokens; that default is not a tested context guarantee.

The inherited [`serve/server.py`](serve/server.py) exposes OpenAI- and Anthropic-compatible routes.
Compatibility and model capabilities vary: GLM has a recorded tool session, DeepSeek lacks its tool-call
format, and MiniMax's end-to-end serving remains unverified. See the inherited
[API reference](docs/DETAILS.md#using-it) and the [documentation map](DOCUMENTATION.md).

## Roadmap

1. Diagnose MiniMax's RAM-copy reset before resuming development GPU work.
2. Improve MiniMax expert residency, sparse selection and prefill; qualify server/tools, then vision and DSpark.
3. Recheck GLM and Qwen after the shared expert-tier changes; expand DeepSeek file and serving coverage.
4. Continue selective Strata request-lifecycle fixes and test long-run stability across supported paths.

The detailed milestones and remaining work are in [PLAN.md](xemaya/PLAN.md).

## Contributing

Reports and pull requests are welcome. Include the exact file, settings and hardware for reproducible issues.
Kernel changes need host-reference parity; model changes need token-level comparisons. Mark untested build
modes and hardware in reports. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Support development

If you would like to support XeMaya, [buy Jumbomuffin777 a coffee](https://buymeacoffee.com/Jumbomuffin777).
Testing on Arc hardware, clear bug reports and documentation improvements also help.

## Licence

XeMaya as a whole is under the **GNU LGPL, version 3 or later**, like XeStrata ([COPYING.LESSER](COPYING.LESSER),
[COPYING](COPYING)). Code from Strata, Project Maya, llama.cpp and other projects keeps its own licence (mostly MIT),
file by file. The full list of copyright holders and licences is in [NOTICE](NOTICE) and [LICENSES/](LICENSES).
Each model's own licence applies to its files.

## Thanks

To **Niko1221** and the Strata contributors for the engine, to **MistVVK** and the XeStrata contributors for the
Intel-native rewrite this project is built on, to **mw00** and the Project Maya contributors for the GLM and DeepSeek
work and the Maya quantisations, and to the ggml / llama.cpp authors for the reference implementations.
