<!--
SPDX-FileCopyrightText: 2026 the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Changelog

## Unreleased — documentation audit (2026-11-10)

Audited source: `b59488e2d5dc69337c3dc30664d26e1b7fe577bb`. This update changes documentation
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
