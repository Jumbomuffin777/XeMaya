<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeMaya 0.1.0-beta.1

Native inference for **Intel Arc GPUs on Linux** (SYCL / Level Zero, XMX), for Qwen3.8-Flash-Next, GLM-5.3-Flash and
DeepSeek-V4.1-Flash. XeMaya is a fork of [XeStrata](https://github.com/MistVVK/XeStrata) by MistVVK, itself a port of
[Strata](https://github.com/Niko1221/Strata) by Niko1221, and targets the models and quantisations of
[Project Maya](https://github.com/mw00/project-maya) by mw00.

**The first beta.** The alphas asked whether it runs; since then each of the three model families has been run again
on GPUs against a reference or a recorded benchmark after the changes below. It is still for Linux and Intel Arc
only, and what was not checked is listed under Known limitations.

## What changed since 0.1.0-alpha.2

- **DeepSeek-V4.1-Flash reads a prompt much faster when the model is larger than the VRAM.** The regular 384-expert
  IQ2_XXS file (123 GiB of experts) on an Arc Pro B70 + B65:

  | Prompt | 0.1.0-alpha.2's code | This release's code | The 63 tokens after it |
  | --- | ---: | ---: | ---: |
  | 576 tokens | 20.0 tok/s | **36.3 tok/s** | 12.8 -> 14.7 tok/s |
  | 2,048 tokens | 36.5 | **71.6** | 11.3 -> 16.2 |
  | 8,192 tokens | 35.2 | **91.8** | 8.4 -> 15.9 |

  The generated tokens are the same. Three changes: an expert the VRAM does not hold is read from the files
  straight into pinned memory, the VRAM tier's fill after a prompt reads 16 experts side by side, and the experts a
  prompt's chunk uploads stay in the tier's free slots (an 8,192-token prompt read 804 GB from the drive before,
  218 GB now). Each has a switch: `XEMAYA_PROMPT_PINNED_READS=0`, `XEMAYA_TIER_FILL_BATCH=0`, `XEMAYA_PROMPT_KEEP=0`.
- **Qwen3.8-Flash-Next decodes faster with drafts:** 77.3 -> 79.4 tok/s on code and 58.0 -> 59.9 on prose after a
  26.8K-token prompt (B70 + B65); Strata v0.2.12 measured 78.7 / 59.2 in the same comparison. The draft layer's
  attention runs on the matrix engines and the draft head's 2-bit rows in bit planes.
- **Opt-in:** `XEMAYA_EXPERT_UNBUFFERED=1` reads a prompt's experts past the file cache (32.8 / 60.3 / 60.8 tok/s at
  the three lengths before the kept experts; the first tokens after a short prompt come slower).
- GLM-5.3-Flash prompts whose experts come from the files take the same faster path (a 198-token prompt: 20.3 s ->
  12.2 s on B70 + B65).

## Install

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xf xemaya-0.1.0-beta.1-linux-x86_64.tar.xz && cd xemaya-0.1.0-beta.1
./xemaya.sh setup
./xemaya.sh doctor
./xemaya.sh models /path/to/your/models
./xemaya.sh serve <id> --port 8095 --no-api-key
```

Needs an x86-64 CPU with AVX2, an Intel Arc GPU with Intel's compute runtime (Level Zero), glibc 2.38 or later
(Ubuntu 24.04+, Fedora 40+) and Python 3.10 or later. `setup` installs Intel's SYCL and oneMKL run-time libraries
from PyPI into the folder. Details: [INSTALL.md](https://github.com/Jumbomuffin777/XeMaya/blob/xemaya-0.1.0-beta.1/INSTALL.md).

## Recorded results

Arc Pro B70 (32 GB) + B65 (32 GB), Intel oneAPI 2026.2, development builds of this release's code on 2026-11-11 and
2026-11-12 unless the row says otherwise. Each row is one file, one prompt set and one configuration; they do not
compare with each other.

| Model and file | Prefill | Decode | Agreement with the llama.cpp CPU reference |
| --- | ---: | ---: | --- |
| DeepSeek-V4.1-Flash regular (384 experts) GSQ-RCO IQ2_XXS, B70 + B65 | 36.3 tok/s at 576 tokens, 91.8 at 8,192 | 14.7–16.2 tok/s | 16 of 16 forced tokens on the 576-token reference |
| DeepSeek-V4.1-Flash Coder GSQ-RCO IQ2_XXS, B70 + B65 | 52.1 tok/s at 576 tokens (the first prompt of a process) | 16.7 tok/s | 16 of 16 forced tokens on the 576-token reference |
| GLM-5.3-Flash Maya-S v2 IQ2_XXS, B70 + B65, 198-token prompt | 16.3 tok/s (the first prompt of a process) | 12.8 tok/s | 125 of 128 on the chat-agent reference |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S, B70 + B65, 26.8K-token prompt | 1,864–1,940 tok/s | 79.2–79.4 / 59.9 tok/s (code / prose) | Speed only; Strata v0.2.12 on the same cards: 78.7 / 59.2 decode |

Earlier three-GPU results (DeepSeek Coder 207–209 tok/s prefill, GLM 152–161 at 2,377 tokens) are in the
[README](https://github.com/Jumbomuffin777/XeMaya/blob/xemaya-0.1.0-beta.1/README.md#status); they were not run again.

## How this release was checked

- Built from a clean export of the tagged commit with `tools/xemaya/release.sh`; its host tests and the GLM, DeepSeek
  and MiniMax kernel parity tests on the OpenCL CPU device pass (MiniMax's is not shipped).
- Installed from the archive in a fresh Ubuntu 24.04 container: checksums, `setup`, the libraries resolve, the
  CPU-device parity tests pass, the server answers against its mock engine.
- On GPUs: the published build's own check (DeepSeek-V4.1-Flash, its Coder file and GLM-5.3 against their
  references on B70 + B65) is recorded at the end of these notes on the release page. Qwen3.8 was **not** run with
  the release build: its numbers above are from a development build of the same code.

## Known limitations

- Reference agreement is recorded for the files and prompts above only. Other quantisations and long contexts are
  not qualified.
- GLM-5.3-Flash Maya-S v2 agrees with its chat-agent reference at 125 of 128 forced tokens, one fewer than in the
  alphas, with the same median and maximum difference: the VRAM tier holds other experts after the prompt, and near
  ties fall the other way. `XEMAYA_PROMPT_KEEP=0` gives the earlier tier.
- A model larger than the VRAM still reads the experts that do not fit once for every 1,024 tokens of a prompt.
- The release is built with Intel oneAPI's icpx. Builds with intel/llvm, the three-GPU configurations and the
  slow-link card were not checked again for this release.
- MiniMax-M3.1-Flash is not in this release: its code is in the source and is held back (slow, and one development
  run ended in a GPU engine reset of unknown cause).
- DeepSeek-V4.1's tool calls are read by the server but were not exercised with the model; its thinking is not
  qualified.
- A GPU on a slow or shared link can be starved by long expert streaming.
- Experimental and unsupported in this release: the hybrid segment (a model's last blocks on another machine), the
  Windows build with the CUDA target (compiles, never run), NVIDIA and AMD GPUs, the inherited deb/rpm packaging and
  `setup.py` installer.

## Files

| File | |
| --- | --- |
| `xemaya-0.1.0-beta.1-linux-x86_64.tar.xz` | The engine built for Intel GPUs, the server, the `xemaya.sh` command, and the complete source of the build |
| `xemaya-0.1.0-beta.1-source.tar.gz` | The source alone |
| `SHA256SUMS` | Checksums of both |

Licence: LGPL-3.0-or-later, with MIT code from Strata, Project Maya and ggml / llama.cpp
([NOTICE](https://github.com/Jumbomuffin777/XeMaya/blob/xemaya-0.1.0-beta.1/NOTICE)).
Maintainer: Jumbomuffin777.
