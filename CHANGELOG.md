<!--
SPDX-FileCopyrightText: 2026 the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Changelog

## 0.1.0-alpha.1 — 2026-11-10

The first public release: an alpha for **Linux and Intel Arc GPUs**. Built on XeStrata xe0.2.8-1.

### What is in it

- The engine for three model families, each as far as its record goes ([README](README.md#models)):
  Qwen3.8-Flash-Next (inherited from XeStrata), GLM-5.3-Flash and DeepSeek-V4.1-Flash. Layer split across several Arc GPUs, expert tiers in VRAM, RAM and on the drive,
  XMX attention for GLM and DeepSeek.
- The OpenAI- and Anthropic-compatible server, with newer Strata tool-calling and request-lifecycle fixes brought
  over, and DeepSeek-V4.1's DSML tool calls (read by the server; not exercised with the model).
- `tools/xemaya/catalog.py`: finds local GGUFs of the three families and writes server configs and tokenizers.
- A release folder for Linux (`tools/xemaya/release.sh`): the engine built with Intel oneAPI 2026.2, the
  `xemaya.sh` command (`setup`, `doctor`, `selftest`, `models`, `serve`), [INSTALL.md](INSTALL.md), checksums.
- The engine says its XeMaya version beside XeStrata's (`strata --version`).

### How it was checked

- The release build, from a clean export of the commit: 13 host tests and the GLM, DeepSeek and MiniMax kernel
  parity tests on the OpenCL CPU device pass. No GPU was opened by the release build or its checks.
- A clean install of the release archive in fresh containers without a GPU: checksums, `xemaya.sh setup`, the
  libraries resolve, the three CPU-device parity tests pass, the server answers against its mock engine.
- The server suite: 959 passed, 23 skipped on the development host; 938 passed, 44 skipped in a clean
  Ubuntu 24.04 container.
- On GPUs (2026-11-10): the release candidate built from the parent commit (the same source but for these documents), installed from its archive with the PyPI run-time, ran DeepSeek-V4.1-Flash Coder IQ2_XXS on Arc Pro B70 + B65 + B60 and matched the llama.cpp reference at 16 of 16 forced tokens on both saved prompts (median |d logprob| 0.009 and 0.010, as the development build; first prompt after the
  load 128 tok/s at 576 tokens, decode 15.6-16.1 tok/s; no kernel message). That check found the release folder
  lacking oneMath's backend under the name oneMath opens, which is fixed.
- The other GPU results (the README's table) are records of 2026-11-09 and 2026-11-10 made with development builds
  of earlier commits on Arc Pro B70 + B65 + B60. GLM-5.3 and Qwen3.8 were not rerun with the release build.
- Licences: `reuse lint` passes; gitleaks finds nothing in XeMaya's own commits.

### Experimental, not part of this release's support

- The hybrid segment (a model's last blocks in a worker on another machine): checked on the CPU device over the
  loopback only.
- The Windows build with the CUDA target: cross-compiles and links; never run.
- MiniMax-M3.1-Flash: in the source, held back from the release (the catalog lists it only with
  `XEMAYA_UNRELEASED=1`). Slow, and one run with its experts held in RAM ended in a GPU engine reset of unknown cause.
- NVIDIA and AMD GPUs, the free and contrib-llvm builds with intel/llvm, and the inherited deb/rpm packaging:
  not built or run by XeMaya.

### Known problems

- After the shared expert-tier changes, the GLM-5.3 and Qwen3.8 forced-token checks were not rerun on GPUs.

## Documentation audit (2026-11-10)

Audited source: `32c979a0a6de0c538e524ee4e8092577157f3a93`. This update changes documentation
and funding metadata, not engine behaviour. Dates below refer to existing records, not new measurements.

### Updated

- Reworked README navigation, model matrix and experimental-status presentation; added a documentation map.
- Added the implemented MiniMax-M3.1-Flash path with its limited qualification: 32/32 on the short chat
  reference, 28/32 at 3,832 tokens, slow expert streaming, and an unresolved GPU reset in a RAM-copy run.
  Vision and DSpark remain unimplemented; end-to-end serving remains unverified.
- Replaced outdated DeepSeek “decode windows only” / “not run on Arc” summary claims with the recorded
  three-GPU chunked/XMX results. Kept the full-model and API qualification gaps explicit.
- Replaced the stale one-prompt GLM summary with the later forced-reference results and recorded agent session.
  Preserved Qwen's measured gap against Strata v0.2.12 instead of asserting general superiority.
- Documented catalog discovery, optional config/tokenizer writes, family capability flags, and the distinction
  between prerequisite readiness, limited family verification and per-file runtime qualification.
- Corrected the direct CMake build instructions for Intel-only icpx, while explaining that setup retains
  the inherited intel/llvm workflow. Labeled inherited docs and repaired their README installation links.
- Standardised XeMaya branding. The case-insensitive repository search found no spaced product-name variant
  in the audited snapshot, so no existing occurrence required replacement. Upstream names, licence notices,
  commands and configuration identifiers retain their actual names.
- Added the owner's Buy Me a Coffee link to README and documentation, plus `.github/FUNDING.yml`.

### Evidence

| Documentation claim | Implemented source / existing record |
| --- | --- |
| Four catalog families, capability flags and readiness semantics | [catalog.py](tools/xemaya/catalog.py): `FAMILIES`, `describe`, `server_config`, `main` |
| MiniMax generic model block and kernels | [plain_model.cpp](src/core/plain_model.cpp), [minimax.hpp](include/strata/kernels/minimax.hpp), [minimax.cpp](src/kernels/xe/minimax.cpp), [XMX attention](src/kernels/xe/minimax_attn_xmx.cpp) |
| DeepSeek chunked/XMX path | [plain_ds41.cpp](src/core/plain_ds41.cpp), [XMX attention](src/kernels/xe/ds41_attn_xmx.cpp) |
| Recorded reference checks, speed and unresolved reset | [STATUS.md](xemaya/STATUS.md): latest handoff, MiniMax, three-GPU DeepSeek, windows 3 and 4 |
| icpx acceptance, Intel-only guard, native experts and parity targets | [CMakeLists.txt](CMakeLists.txt): `XEMAYA_ALLOW_ICPX`, IntelLLVM checks, `STRATA_NATIVE_EXPERTS`, `minimax_parity` |
| Setup's separate compiler policy and GLM model list | [setup.py](setup.py): `recorded_compiler` and GLM Maya entries |
| DeepSeek/MiniMax tokenizer additions | [strata_tokenizer.py](tools/strata_tokenizer.py): `joyai-llm`, `minimax-m2` |

### Validation scope

Documentation links and changed-file whitespace were checked locally, and the repository text was searched
for spaced XeMaya branding. No model inference, GPU job, production service change or new benchmark was run.
The full Linux lint suite and free/contrib-llvm/Windows builds were not available for this documentation audit;
those checks remain unverified. Historical test counts are attributed to STATUS, not presented as fresh runs.
