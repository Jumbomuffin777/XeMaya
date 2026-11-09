# XeMaya status

Updated 2026-11-09 14:20 UTC (after the first GPU window). Branch `xemaya/main` = `origin/main` of the private
https://github.com/Jumbomuffin777/XeMaya. Every item below was run on Zability unless marked otherwise.

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

## Unverified (built, not run on an Arc GPU)

- Kernels other than glm5/ds41/elementwise have only run on the CPU device.
- The `strata` engine has not loaded a model (Maya-S v2 is refused for its F32 mHC mixers; see the GPU window).
- The DeepSeek kernels are written for correctness, not speed.

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

1. **Another GPU window** (as on 2026-11-09: production Qwen stopped by the owner, XeMaya under the whole-host lease,
   about 30 minutes) once the F32 mHC mixer support is in and checked on the CPU device. Impact: production Qwen and the
   Hermes agents are down for the window. Rollback: the owner's `xemaya-window-end.sh`.
2. For kernel-only tests without stopping production, an `OWNERS` entry `'xemaya-dev': B60` (one card, beside the
   isolated Strata unit as `comfyui` is) would do; ComfyUI and marrow-qa could not use the B60 meanwhile. Rollback:
   remove the entry. Not needed for full-model tests, which take the whole host.

## Notes

- `gh` on the Windows PC is not logged in. `gh` on Zability is (Jumbomuffin777) and is what this work used.
- There is no passwordless sudo on Zability.
- Upstream history includes a 68 MB `src2.tar`. GitHub warns but the push succeeds; history is not rewritten.
- Commit identity is repo-local: `Jumbomuffin777 <ID+Jumbomuffin777@users.noreply.github.com>`.
