<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
-->
# XeMaya

**Native inference for Intel Arc. Built on XeStrata. Targeting Project Maya's models.**

[![Release: 0.1.0-beta.1](https://img.shields.io/badge/release-0.1.0--beta.1-brightgreen)](https://github.com/Jumbomuffin777/XeMaya/releases/latest)
[![Status: beta](https://img.shields.io/badge/status-beta-green)](xemaya/STATUS.md)
[![Backend: SYCL / Level Zero](https://img.shields.io/badge/backend-SYCL%20%2F%20Level%20Zero-blue)](docs/XE.md)
[![Licence: LGPL v3+](https://img.shields.io/badge/licence-LGPL--3.0--or--later-blue)](COPYING.LESSER)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Jumbomuffin777-FFDD00?logo=buymeacoffee&logoColor=000)](https://buymeacoffee.com/Jumbomuffin777)

[Install](#install) · [Models](#models) · [Recorded results](#status) · [Build](#building) ·
[Model discovery](#model-discovery-and-serving) · [Documentation](DOCUMENTATION.md) · [Changelog](CHANGELOG.md)

XeMaya is an Intel Arc-first inference engine for Qwen3.8-Flash-Next, GLM-5.3-Flash and
DeepSeek-V4.1-Flash. It runs native SYCL / Level Zero kernels,
including Intel XMX paths, with multi-GPU layer splitting and VRAM / RAM / SSD expert tiers.
The engine is built on XeStrata, rather than a llama.cpp server wrapper or a CUDA compatibility layer.

> **Beta (0.1.0-beta.1, 2026-11-12): Linux, Intel Arc GPUs.** Reference agreement is limited to the
> models, prompts and hardware recorded below; it does not establish production readiness. Each release
> build passes its host tests, CPU-device kernel parity, a clean install and a DeepSeek-V4.1 reference
> check on Arc cards before it is published (its release notes have the result); the other GPU
> results below come from development builds of earlier commits. Read the [known limitations](INSTALL.md#known-limitations).

## Install

Download `xemaya-<version>-linux-x86_64.tar.xz` and `SHA256SUMS` from the
[releases](https://github.com/Jumbomuffin777/XeMaya/releases), then:

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xf xemaya-*-linux-x86_64.tar.xz && cd xemaya-*/
./xemaya.sh setup                          # a venv in the folder: the server's packages, Intel's SYCL and oneMKL run-time
./xemaya.sh doctor                         # the engine's version and the GPUs it sees
./xemaya.sh models /path/to/your/models    # what it found, as JSON; writes configs and tokenizers
./xemaya.sh serve <id> --port 8095 --no-api-key
```

It needs an x86-64 CPU with AVX2, an Intel Arc GPU with Intel's compute runtime (Level Zero), glibc 2.38 or later
and Python 3.10 or later. [INSTALL.md](INSTALL.md) has the requirements by distribution, several GPUs, and what is
known not to work.

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
| **DeepSeek-V4.1-Flash** | `deepseek41`; compressed attention, Engram, chunked prefill and XMX attention | Coder GSQ-RCO IQ2_XXS: 16/16 forced tokens on both saved references on three GPUs. The regular (non-Coder) 384-expert IQ2_XXS file has speed results on two and three GPUs with the `0.1.0-alpha.1` build ([benchmark](xemaya/benchmarks/2026-11-10-ds41-flash-384/README.md)), without reference comparison. The server reads the model's DSML tool calls (checked against a fake engine and golden prompts, not with the model); qualified thinking and drafting are absent from the catalog. |

MiniMax-M3.1-Flash is in the source and **held back from this release**: see
[Experimental](#experimental-outside-this-releases-scope-2026-11-10-no-gpu-used).

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
| DeepSeek-V4.1-Flash **regular (384 experts, not Coder)** IQ2_XXS, **B70 + B65**, published `0.1.0-alpha.1` build, 576-token prompt, 1,024-token decode | 11.4 tok/s | **14.56 tok/s** | Speed only, no reference comparison. 93.7% of routed experts in VRAM; misses run in the CPU pool. |
| Same file and build, **B70 + B65 + B60** (`XEMAYA_SLOW_LINK_GPUS=1`) | 11.2 tok/s | **11.58 tok/s** | Speed only. Slower than two GPUs: the B60 lifts the hit rate only to 94.2% and its blocks add ~17.6 ms a token ([why](xemaya/benchmarks/2026-11-10-ds41-flash-384/README.md#why-two-gpus-are-faster-than-three-here)). |
| DeepSeek-V4.1-Flash **regular (384 experts)** IQ2_XXS, **B70 + B65**, development build of 2026-11-12, shards on a PCIe Gen 5 SSD ([benchmark](xemaya/benchmarks/2026-11-11-ds41-flash-prefill/README.md)) | **36.3 tok/s** at 576 tokens, **71.6** at 2,048, **91.8** at 8,192 | 14.7–16.2 tok/s (the 63 tokens after the prompt) | 16/16 forced tokens on the 576-token reference. With the `0.1.0-alpha.2` code on the same drive: 20.0 / 36.5 / 35.2 tok/s prefill. |
| Qwen3.8-Flash-Next, B70 + B65, 26.8K prompt, development build of 2026-11-12 ([benchmark](xemaya/benchmarks/2026-11-11-qwen-ab/README.md)) | 1,864–1,940 tok/s | **79.2–79.4 / 59.9 tok/s** (code / prose) | Speed only. Strata v0.2.12 in the same comparison: 78.7 / 59.2 decode. |
| Qwen3.8-Flash-Next, B70 + B65, 26.8K prompt (window 4) | 1,842–1,884 tok/s; 1,989–1,996 with `STRATA_PREFILL_MMQ=0` | 76.6 / 57.1 tok/s (code / prose) | Behind Strata v0.2.12's 2,211 tok/s prefill and 78.3 / 59.2 decode in that comparison. |

GLM's earlier llama.cpp SYCL baselines were 68.3 / 13.6 tok/s (stock) and 108.9 / 14.0
(experimental grouped matmul), prefill / decode at 2,377 tokens. Results across different prompts,
files, GPU counts or cold/warm states are not interchangeable; no general speed superiority is claimed.
The Coder and regular DeepSeek-V4.1-Flash rows are different files (192 and 384 experts a layer). The regular file
(129 GB) is larger than the cards' VRAM, and its shards and Engram file were read from a PCIe Gen 3 NVMe SSD.
XMX matrix acceleration was enabled for the regular file's runs; their attention kernel path was not independently
verified (the engine does not log it).

Recorded checks also include host tests, GLM / DeepSeek kernel parity, IQ-format parity,
and the server suite (959 passed, 23 skipped on the development host at the release commit's tree; 938 passed,
44 skipped in a clean Ubuntu 24.04 container). Later source changes have not
all been rerun across every model, build mode or OS.

### Experimental, outside this release's scope (2026-11-10, no GPU used)

- **Hybrid segment:** a model's last blocks can run in a segment worker on another machine
  (`--segment-last` / `--segment-serve`). Checked on the CPU device over the loopback only (32/32 against
  the llama.cpp reference); never run on GPUs or across two machines.
- **MiniMax-M3.1-Flash** (`minimax-m3`; sparse attention, SwiGLU-OAI): held back from this release. On three
  GPUs it matched its reference on a short chat prompt (32/32; 28/32 at 3,832 tokens) but decodes at 2–4 tok/s
  because 30% of its experts do not fit in VRAM, and one run that held them in RAM ended in a GPU engine reset
  of unknown cause. The catalog lists it only with `XEMAYA_UNRELEASED=1`; no tools, vision or drafting.
- **Windows with the CUDA target:** the engine cross-compiles and links (`tools/xemaya/win_build.sh`);
  it has never been run on Windows. The first-run procedure is [xemaya/WINDOWS-FIRST-RUN.md](xemaya/WINDOWS-FIRST-RUN.md).
- **DeepSeek-V4.1 tool calls:** the server reads the model's DSML call block (`serve/dsml.py`, from
  upstream Strata) through both APIs; checked against the fake engine and the golden prompts only.
- **NVIDIA and AMD GPUs:** XeStrata's paths for them are in the source; XeMaya has not built or run them.

## Building

From the XeMaya checkout on Linux, with Intel oneAPI available:

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
  -DSTRATA_LICENSE=contrib -DXEMAYA_ALLOW_ICPX=ON \
  -DSTRATA_CUDA_ARCHS= -DSTRATA_HIP_ARCHS= \
  -DSTRATA_ENABLE_XE=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build --target strata glm5_parity ds41_parity model_spec_test --parallel 4
```

`CMakeLists.txt` accepts `icpx` through `XEMAYA_ALLOW_ICPX` (default ON); its build is Intel-only.
Native expert support uses ggml-cpu from the pinned source fetched by CMake, or a llama.cpp checkout
selected with `STRATA_GGML_DIR`. It does not use llama.cpp as the serving backend.

CPU-device kernel parity requires an installed OpenCL CPU device:

```bash
ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 build/glm5_parity
ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 build/ds41_parity
```

The inherited setup/packaging workflow still uses the `xestrata` name and intel/llvm toolchain;
`setup.py` does not adopt the direct CMake `icpx` workflow above. Read the inherited
[installation guide](docs/XESTRATA.md) and [build guide](docs/BUILD.md) with that distinction in mind.
[`tools/xemaya/release.sh`](tools/xemaya/release.sh) runs this build from a clean export of a commit, runs the
host tests and the CPU-device parity tests, and packs the release folder with its checksums.

## Model discovery and serving

[`tools/xemaya/catalog.py`](tools/xemaya/catalog.py) scans local GGUF headers without loading a model
or opening a GPU. It recognises `qwen4exp`, `glm5-next` and `deepseek41`, groups split files
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
Compatibility and model capabilities vary: GLM has a recorded tool session, DeepSeek's tool calls are read
but were not exercised with the model. See the inherited
[API reference](docs/DETAILS.md#using-it) and the [documentation map](DOCUMENTATION.md).

## Roadmap

1. Recheck the release on more Arc hardware; fix what the first reports find.
2. MiniMax-M3.1-Flash: diagnose the RAM-copy reset, improve expert residency and prefill, then release it.
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
