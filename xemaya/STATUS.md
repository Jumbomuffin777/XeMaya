# XeMaya status

Updated 2026-11-09 06:30 UTC (end of the first session). Branch `xemaya/main` = `origin/main` of the private
https://github.com/Jumbomuffin777/XeMaya. Every item below was run on Zability unless marked otherwise.

## Works (checked)

| What | How it was checked |
|---|---|
| Builds with oneAPI icpx 2026.2 | Core, tests and parity binaries, plus the full `strata` engine (GLM and DeepSeek kernels included), with 0 errors |
| Host tests | 13 XeStrata host tests pass, plus `model_spec_test` and `ds41_manifest_test` |
| GLM-5.3-Flash kernels | `glm5_parity` PASS, 28/28, on the OpenCL **CPU** device |
| GLM Maya GGUFs | Maya-S v2 IQ2_XXS reads as `glm5-next` (45 trunk + 1 MTP layers, 42 MoE layers, 34 KDA layers); all its tensor types are supported; `setup.py` lists Maya-S-v2/S24/XS/M/L, pinned with sha256 |
| DeepSeek-V4.1 model spec | `model_spec_test` reads the ISTA-DASLab GSQ-RCO and Coder IQ2_XXS files (40 layers, 38 compressed, 2 Engram modules, 384 / 192 experts) |
| DeepSeek-V4.1 tensors | `ds41_manifest_test` passes both real models: 6 shards, all 1043 tensors with llama.cpp's shapes, in bounds, no unknown tensor |
| DeepSeek-V4.1 kernels | `ds41_parity` PASS, 18/18, on the CPU device: router, mHC pre/collapse, compressor, Engram gate + hash, YaRN RoPE (and its inverse), indexer, top-k, candidate blocks, sink attention |
| Serve layer | 14 upstream Strata tool-call commits ported (#4663, #4701, #4702/#4703, #4751, #5040, #5276, #5281, #5409, tool_choice required, Anthropic tool-result order). Full suite: **930 passed, 23 skipped, 0 failed** (baseline was 862) |

## Unverified (built, not run on an Arc GPU)

- Every SYCL kernel has only run on the CPU device, which exercises no XMX paths and no Level Zero features.
- The `strata` engine has not loaded a model (GPU access is blocked; see below).
- `elementwise_parity` needs Level Zero host-memory registration and was not run.
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

1. **GPU access for XeMaya tests.** Production Strata holds the whole-host lease, and `gpu_partitions.py` only lets
   `comfyui` and `marrow-qa` claim the B60 meanwhile. Two options:
   - (a) Add `'xemaya-dev': B60` to `OWNERS` and allow it beside the isolated Strata unit, as `comfyui` is
     (`claim()`'s `legacy_strata_is_isolated` branch). Impact: XeMaya uses only the B60 (24 GB), so ComfyUI and
     marrow-qa cannot use it at the same time. Leases and the memory guard stay in force. Rollback: remove the entry.
     This is enough for kernel parity on real Xe2 hardware (`glm5_parity`, `ds41_parity`, `elementwise_parity`,
     everything that hits XMX). It is **not** enough for a full model: GLM Maya-S v2 is 89.8 GiB and ~50 GB of host
     RAM is free.
   - (b) A **maintenance window**: you stop `zability-strata-qwen38.service`, XeMaya runs GLM Maya-S v2 across the B70,
     B65 and B60 under the lease (`gpulease.py run --holder <name>` + `preflight.sh`), then you restart the service.
     Impact: production Qwen and the Hermes agents are down for the window (about 30 minutes for the first load, a
     16-token correctness check against llama.cpp SYCL on the same file, and a short benchmark). Rollback: start the
     service again.
2. Nothing else is pending. The `/DATA/Projects/xemaya` permission fix was done by the owner on 2026-11-09.

## Notes

- `gh` on the Windows PC is not logged in. `gh` on Zability is (Jumbomuffin777) and is what this work used.
- There is no passwordless sudo on Zability.
- Upstream history includes a 68 MB `src2.tar`. GitHub warns but the push succeeds; history is not rewritten.
- Commit identity is repo-local: `Jumbomuffin777 <ID+Jumbomuffin777@users.noreply.github.com>`.
