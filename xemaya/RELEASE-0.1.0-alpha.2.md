<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeMaya 0.1.0-alpha.2

The first public release of XeMaya: native inference for **Intel Arc GPUs on Linux** (SYCL / Level Zero, XMX),
for Qwen3.8-Flash-Next, GLM-5.3-Flash and DeepSeek-V4.1-Flash.
XeMaya is a fork of [XeStrata](https://github.com/MistVVK/XeStrata) by MistVVK, itself a port of
[Strata](https://github.com/Niko1221/Strata) by Niko1221, and targets the models and quantisations of
[Project Maya](https://github.com/mw00/project-maya) by mw00.

## What changed since 0.1.0-alpha.1

- **Fixed:** the server no longer fails to start for a DeepSeek-V4.1 model whose pack's chat template is not valid
  UTF-8; it falls back to the V4.1 template and says so. Thanks to **@bettercallcaleb** for the fix
  ([#1](https://github.com/Jumbomuffin777/XeMaya/pull/1)), XeMaya's first outside contribution.
- Recorded speeds of the regular DeepSeek-V4.1-Flash file (384 experts) were added to the repository.

**This is an alpha.** It is for Arc owners who want to try it and report what happens, not for anything you depend on.

## Install

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xf xemaya-0.1.0-alpha.2-linux-x86_64.tar.xz && cd xemaya-0.1.0-alpha.2
./xemaya.sh setup
./xemaya.sh doctor
./xemaya.sh models /path/to/your/models
./xemaya.sh serve <id> --port 8095 --no-api-key
```

Needs an x86-64 CPU with AVX2, an Intel Arc GPU with Intel's compute runtime (Level Zero), glibc 2.38 or later
(Ubuntu 24.04+, Fedora 40+) and Python 3.10 or later. `setup` installs Intel's SYCL and oneMKL run-time libraries
from PyPI into the folder. Details: [INSTALL.md](https://github.com/Jumbomuffin777/XeMaya/blob/xemaya-0.1.0-alpha.2/INSTALL.md).

## Recorded results

Arc Pro B70 (32 GB) + B65 (32 GB) + B60 (24 GB), Intel oneAPI 2026.2, 2026-11-09 and 2026-11-10, development builds
of earlier commits. Each row is one file, one prompt set and one configuration; they do not compare with each other.

| Model and file | Prefill | Decode | Agreement with the llama.cpp CPU reference |
| --- | ---: | ---: | --- |
| DeepSeek-V4.1-Flash Coder GSQ-RCO IQ2_XXS, three GPUs | 207–209 tok/s at 576 tokens, 225 at 2,876 | 19.5–21.7 tok/s | 16 of 16 forced tokens on both references |
| GLM-5.3-Flash Maya-S v2 IQ2_XXS, three GPUs, 2,377-token prompt | 152–161 tok/s | 17.4 tok/s (20.0 with MTP drafts) | 126 of 128 on the chat-agent reference |
| Qwen3.8-Flash-Next GSQ-RCO IQ3_S, B70 + B65, 26.8K-token prompt | 1,842–1,884 tok/s | 76.6 / 57.1 tok/s (code / prose) | Speed only; Strata v0.2.12 on the same cards: 2,211 prefill, 78.3 / 59.2 decode |

On the same GLM file and prompt, llama.cpp's SYCL backend measured 68.3 tok/s prefill and 13.6 tok/s decode on this
machine. Qwen is behind upstream Strata, as the table says.

## How this release was checked

- Built from a clean export of the tagged commit with `tools/xemaya/release.sh`; 13 host tests and the GLM, DeepSeek
  and MiniMax kernel parity tests on the OpenCL CPU device pass (MiniMax's is not shipped).
- The fix: `serve/test_dsml.py`, 26 passed; its new test fails without the change. CI (the server's tests, licence
  lint) passes on the tagged commit.
- Installed from the archive in a fresh Ubuntu 24.04 container: checksums, `setup`, the libraries resolve, the
  CPU-device parity tests pass, the server answers against its mock engine. (0.1.0-alpha.1, the same engine source,
  was also installed in Ubuntu 26.04 and Fedora 44 containers.)
- On GPUs: the published build's own check is recorded at the end of these notes on the release page. GLM-5.3 and
  Qwen3.8 were **not** run with the release build: their numbers above are from development builds.

## Known limitations

- Reference agreement is recorded for the files and prompts above only. Other quantisations and long contexts are
  not qualified.
- MiniMax-M3.1-Flash is not in this release: its code is in the source and is held back (slow, and one development
  run ended in a GPU engine reset of unknown cause).
- DeepSeek-V4.1's tool calls are read by the server but were not exercised with the model; its thinking is not
  qualified.
- Qwen3.8 and GLM-5.3 were not rechecked on GPUs after the last shared expert-tier changes.
- A GPU on a slow or shared link can be starved by long expert streaming.
- Experimental and unsupported in this release: the hybrid segment (a model's last blocks on another machine), the
  Windows build with the CUDA target (compiles, never run), NVIDIA and AMD GPUs, the inherited deb/rpm packaging and
  `setup.py` installer.

## Files

| File | |
| --- | --- |
| `xemaya-0.1.0-alpha.2-linux-x86_64.tar.xz` | The engine built for Intel GPUs, the server, the `xemaya.sh` command, and the complete source of the build |
| `xemaya-0.1.0-alpha.2-source.tar.gz` | The source alone |
| `SHA256SUMS` | Checksums of both |

Licence: LGPL-3.0-or-later, with MIT code from Strata, Project Maya and ggml / llama.cpp
([NOTICE](https://github.com/Jumbomuffin777/XeMaya/blob/xemaya-0.1.0-alpha.2/NOTICE)).
Maintainer: Jumbomuffin777.
