# XeMaya status

Updated 2026-11-09 (first session). Branch `xemaya/main`; private remote https://github.com/Jumbomuffin777/XeMaya.

## Works (checked)

| What | How it was checked |
|---|---|
| Builds with oneAPI icpx 2026.2 | core, host tests, parity binaries (104 steps) and the full `strata` engine with GLM (177 steps), 0 errors |
| Host tests | 13 pass: gguf_split, gguf_reader, model_spec, share_rules, ple_reader, suffix_drafter, controller, draft_policy, conv_cache, split_rules, batch_groups, conversation_cache, logit_bias |
| GLM-5.3-Flash kernels | `glm5_parity` PASS, 28/28 (mHC, Sinkhorn, router, KDA, indexer, MLA, IQ1_S, clamped SwiGLU), on the OpenCL **CPU** device |
| DeepSeek-V4.1 model spec | `model_spec_test` PASS; reads ISTA-DASLab's GSQ-RCO and Coder IQ2_XXS files (40 layers, 38 compressed, 2 Engram modules) |
| Maya GLM GGUF compatibility (header level) | Maya-S v2 IQ2_XXS is arch `glm5-next` with types Q6_K, Q8_0, IQ2_XXS, IQ2_S, IQ3_XXS, F32, all supported |
| Serve suite baseline | 862 passed, 23 skipped, 0 failed |

## Unverified (built but not run on Intel Arc)

- Every SYCL kernel compiled with icpx has been run on the CPU device only. XMX paths are not exercised there.
- The `strata` engine has not loaded a model (no GPU access; see below).
- `elementwise_parity` needs Level Zero host-memory registration and was not run.

## In progress

- GLM Maya-S v2 IQ2_XXS download (sha256-checked, 50 MB/s cap): log `/DATA/Projects/xemaya/logs/fetch-glm-maya-s-v2.log`.
  Re-run `tools/xemaya/fetch_hf.sh peasantsmith/GLM-5.3-Flash-Maya-GGUF Maya-S-v2-IQ2_XXS
  /mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS 50M` to resume it.
- Strata v0.2.8..v0.2.12 `serve/` ports (PLAN, Milestone 2).

## Needs the owner's approval (not done)

1. **A GPU partition owner for XeMaya.** Production Strata holds the whole-host lease; `gpu_partitions.py` lets only
   `comfyui` and `marrow-qa` claim the B60 meanwhile. Proposed: add `'xemaya-dev': B60` to `OWNERS` and allow it beside
   the isolated Strata unit, as `comfyui` is (`claim()`'s `legacy_strata_is_isolated` branch). Impact: XeMaya tests use
   only the B60 (24 GB), so ComfyUI and marrow-qa cannot use it at the same time. The lease and memory guard stay
   in force. Rollback: remove the line.
   Even then, GLM Maya-S v2 (89.8 GiB) does not fit one B60 plus the ~50 GB of free host RAM, so a full GLM or
   DeepSeek run needs the B70+B65 in a **maintenance window** with production Qwen stopped by the owner.
   Rollback: restart `zability-strata-qwen38.service`.
2. **Moving the working tree**: done by the owner on 2026-11-09 (`/DATA/Projects/xemaya` now writable).

## Blockers and notes

- `gh` on the Windows PC is not logged in; `gh` on Zability is (Jumbomuffin777) and is used instead.
- No passwordless sudo on Zability.
- Upstream history includes a 68 MB `src2.tar` (GitHub warns, the push succeeds). History is not rewritten.
