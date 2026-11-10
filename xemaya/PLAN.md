# XeMaya plan

XeMaya is an Intel Arc-first native inference engine for GLM-5.3-Flash, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash and
Qwen3.8-Flash-Next.
It is not a CUDA compatibility layer and not a llama.cpp wrapper.

## Architecture decision (2026-11-09)

**Foundation: XeStrata (MistVVK/XeStrata, LGPL-3.0-or-later AND MIT), branch `xemaya/main`.**

Inspected sources and what they offered:

| | XeStrata | Strata v0.2.12 (Niko1221) | Project Maya (mw00) |
|---|---|---|---|
| Intel GPU code | Hand-written SYCL / Level Zero kernels (`src/kernels/xe`, ~400 files): XMX through joint_matrix, DP4a fallbacks, device-reported tile shapes | `sycl/`: SYCLomatic-migrated CUDA with manual fixes, kept as a side tree beside the CUDA engine | None (CUDA/HIP only) |
| Models | Qwen3.8 (main), PlainModel: GLM-5.3-Flash (`glm5-next`, also Maya's `glm5next` spelling), Ornith, llm-jp, Qwen3.6, Qwen3.5 | Qwen3.8; DeepSeek-V4.1 engine `src/dsv41` (CUDA/HIP, ~2.1 MB of source, 858 KB in `ds4_model.cu` alone) | GLM engine (~21K lines CUDA), DeepSeek `src/ds4` (CUDA/HIP) |
| Multi-GPU | Layer split over several GPUs, mixed makers | Layer split (CUDA) and SYCL port | CUDA |
| Experts | VRAM cache + RAM tier + SSD (direct I/O), per-layer cache, CPU AVX-512 expert lane | Same lineage, newer CUDA ideas | Hot/cold tiers |
| Status vs upstream | Synced to Strata v0.2.8 (merge `721afe51f`) | v0.2.12 | - |

Why XeStrata:

1. Its kernels are written for Intel GPUs (sub-group and XMX aware) rather than translated from CUDA. The earlier
   attempt on this host to SYCLomatic-port Maya's DeepSeek engine (`/DATA/AppData/maya-upstream/native-port/AUDIT.md`)
   stalled on cross-device pointer faults that come from dpct's one-context-per-device model.
2. It already runs GLM-5.3-Flash natively. Its `glm5_parity` test passes on XeMaya's build (28/28 checks; run on the CPU
   device, see STATUS).
3. Its PlainModel is the right base for DeepSeek-V4.1: the mHC hyper-connections, clamped SwiGLU, sigmoid-family
   routers, expert tiers, multi-GPU layer split and the PLE n-gram SSD reader that Engram needs are already there.
4. It dropped CUDA, Metal and the desktop GUI, so it is a much smaller surface.

How the other sources are used:

- **Strata**: `serve/` fixes (tool calls, request lifecycle) are ported commit by commit as 3-way patches restricted to
  `serve/`, and kept only where the serve test suite stays green. `src/dsv41` is a design reference for DeepSeek
  streaming, Engram I/O and expert rules, not code to translate.
- **llama.cpp** (`src/models/deepseek41.cpp`, `deepseek4.cpp` at master 8642541b2; local copy
  `/DATA/AppData/maya-upstream/llama.cpp-master-20261029`): the exact reference graph. XeStrata's GLM port is transcribed
  from llama.cpp's `glm5-next.cpp` the same way.
- **Project Maya**: its GGUF quantisations (GLM Maya-S/S24/XS/M/L, DeepSeek Maya-XS/S) and ideas.

Toolchain: Intel oneAPI `icpx` 2026.2 (libsycl 9) with `-ffp-model=precise` (`XEMAYA_ALLOW_ICPX`, default ON). XeStrata
removed icpx on 2026-10-15 to keep one compiler for all GPU makers; XeMaya is Intel-first and the hosts already have
oneAPI. intel/llvm DPC++ 7+ still works.

## Milestones

| # | Milestone | State |
|---|---|---|
| 1 | Foundation: repo, history, licences, icpx build, CPU-device kernel testing | done |
| 2 | Upstream Strata v0.2.8 -> v0.2.12: `serve/` correctness (tool calls, request lifecycle, Anthropic route) | tool calls done; lifecycle pending |
| 3 | Native GLM-5.3-Flash: Maya-S v2 (IQ2_XXS) on the Arc cards, output checked against llama.cpp | runs on B70+B65+B60, 16/16 tokens match (2026-11-09); tuning |
| 4 | Native DeepSeek-V4.1-Flash: PlainModel `deepseek41` block, SYCL kernels with CPU-device parity tests | runs on B70 + B65 + B60 with the XMX attention, both references 16 of 16, 207-225 / 19.5-21.7 tok/s (2026-11-10); the full model, serve next |
| 6 | Native MiniMax-M3.1-Flash: Maya-S IQ2_XXS in PlainModel's generic block, the sparse attention (MSA) kernels, `mm_host` as the reference | runs on B70 + B65 + B60, 32 of 32 on the chat-template reference (2026-11-10); the experts outside VRAM, speed, serve, vision and the drafter next (STATUS) |
| 5 | Performance and reliability: decode/prefill, expert residency, PCIe, multi-GPU scheduling, API soak | GLM measured (STATUS); tiled attention and the prompt-end checkpoint built, window 3 prepared |

### Milestone 3: GLM-5.3-Flash

1. Done: the kernels pass `glm5_parity` with icpx. The engine builds. Maya-S v2 uses only types XeStrata has SYCL and CPU
   kernels for (Q6_K, Q8_0, IQ2_XXS, IQ2_S, IQ3_XXS, F32) under arch `glm5-next`.
2. Model: Maya-S v2 IQ2_XXS (89.8 GiB) in `/mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/Maya-S-v2-IQ2_XXS/`
   (sha256-checked download). `setup.py --family glm53` lists Maya-S-v2/S24/XS/M/L, pinned. Its MTP layer is present,
   so XeStrata can draft with it.
3. Needs GPU (approval): load on B70+B65+B60, then a greedy 16-token check against llama.cpp SYCL on the same file. The
   historical reference is 68.3 / 13.6 tok/s on stock llama.cpp v0.8.0.
4. Then the serve path: OpenAI and Anthropic routes, tool calls, streaming.

### Milestone 5: GLM for coding agents (decided 2026-11-09)

Priorities from the owner: GLM attention and TTFT first, then agent usability (cached prefixes, long context, tool
calls, sustained throughput), then wider correctness, then DeepSeek. Evidence before optimization:

1. Attention: `glm5_mla_attn_tiled` (built, CPU-checked). Window 3 measures it against the original and prints the
   achieved GFLOP/s; only if it is still far from the cards' rate is an XMX (joint_matrix) version worth its cost,
   tested on the cards' own parity run since the CPU device here cannot run joint_matrix.
2. Long context: DSA selects at most 8,192 cells a token, so attention grows linearly past 8K but the indexer's
   scores grow with the square of the prompt (64K: ~4.4 T multiply-adds a layer). Window 3's profile split shows which.
3. Agent turns: the prompt-end checkpoint (built). Next if the window shows the need: checkpoints inside long prompts
   (Qwen3.8's `--prompt-cache-every`), so a client that edits early history does not read everything again.
4. Load time (53-60 s, NVMe-bound): lower priority.

### Milestone 4: DeepSeek-V4.1-Flash

Target files on disk: ISTA-DASLab GSQ-RCO IQ2_XXS (+ Coder) with Engram IQ4_NL (`/mnt/ai/models/ISTA-DASLab/`). Tensor
types are Q8_0, Q5_K, BF16, IQ1_M, IQ2_XXS and F32.

Steps (each with a CPU-device parity test against a host reference transcribed from llama.cpp):

1. Done: the `deepseek41` model spec with llama.cpp's checks (compress ratios, KV/top-k sources, candidate blocks,
   Engram), in `include/strata/artifact/model_spec.hpp`.
2. Done: the tensor manifest and its check (`include/strata/artifact/ds41_manifest.hpp`). Both real models pass.
3. Done: the kernels (`include/strata/kernels/ds41.hpp`, `src/kernels/xe/ds41.cpp`, `ds41_parity` 18/18).
4. Written (2026-11-09): `src/core/plain_ds41.cpp`, the block in PlainModel, with `src/reference/ds41_host.cpp` as its
   host oracle (16 of 16 against llama.cpp on two prompts). The block passes the 24- and 576-token forced checks on
   the OpenCL CPU device; open: the first GPU run (tools/xemaya/window5_ds41.sh), a chunked prompt path (STATUS, handoff).
5. Engram table: 110 GB IQ4_NL, read by row through the PLE reader (`src/ngram`, direct I/O), dequantised on the host.
   CPU expert lane: ggml's `vec_dot` covers IQ1_M and IQ2_XXS already (`native_expert.cpp` dispatches by type traits).
6. GPU validation (approval): greedy tokens against llama.cpp-ds41 SYCL on the same file, then serve with DSML tool
   calls (upstream `serve/dsml.py`, not yet ported) and the DeepSeek chat template (`serve/fixtures/deepseek41`).

#### Block design (plain_ds41.cpp)

Per layer `l`, for a window of T rows at positions `pos` (device `steps` as GLM's):

```
x [T][4][N]: the mHC streams.  Before layer 0: glm5_hc_init(embedding).
if engram(l):  rows = table rows of ds41_engram_hash (host, per token) -> engram_kv proj -> ds41_engram_gate(x)
mix_a = hc_attn_fn . x  (F32 GEMV; glm5_hc_pre gives post_a, comb_a; its y is not used)
cur = (l == 0) ? x[:,0] : ds41_hc_collapse(x, pre_prev);  rms_norm(attn_norm)
attn(cur):
  qr = rmsnorm(q_a . cur, q_a_norm); q = q_b . qr [T][64][512]; ds41_rope(q, base/YaRN of the layer, offset 448)
  kv = rmsnorm(kv . cur, kv_a_norm); ds41_rope(kv) -> raw ring [W + max_t][512] at pos % ring
  if ratio(l) and kv_src(l) == l:   compressor state rows (kv, gate projections) per position; when a block of
      `ratio` positions completes: ds41_compress -> comp latent; ik = rmsnorm(indexer.attn_k . comp); rope both at the
      block's first position (compressed base) -> comp cache row [comp_idx][512 + 128]
  if ratio(l):
      if topk_src(l) == l:  iq = indexer.attn_q_b . qr, rope; iw = indexer.proj . cur / sqrt(128 x 32)
          ds41_idx_scores -> (cand_src == l: ds41_block_max + ds41_topk(2048) = candidate blocks)
                           -> (cand_src < l: ds41_keep_blocks) -> ds41_topk(512) -> ids of layer l
      ds41_sink_attn(q, raw ring, window 128, comp cache of kv_src(l), ids of topk_src(l), sinks)
  else: ds41_sink_attn over the window only
  out: ds41_rope(inverse) on the output, then 8 grouped GEMVs wo_a[g] (4096 -> 1024), then wo_b (8192 -> 5120)
x = glm5_hc_post(attn_out, x, post_a, comb_a)
mix_f = hc_ffn_fn . x (glm5_hc_pre: post_f, comb_f; ds41_hc_pre_weights -> pre_f);  cur = ds41_hc_collapse(x, pre_a)
rms_norm(ffn_norm); logits = gate_inp . cur (F32); ds41_router (k 6, scale 1.5, norm); routed experts (the tiers, as
GLM's record_moe, router swapped) + shared expert (glm5_swiglu_clamp, limit 10); x = glm5_hc_post(ffn, x, post_f, comb_f)
pre_prev = pre_f
```

Head: `ds41_hc_collapse(x, pre_prev)`, rms_norm(output_norm), output. Per-sequence state: the raw ring per layer, the
compressor state per KV-source layer (`ratio` rows of kv and gate), the compressed cache per KV-source layer
(`max_cells / ratio` rows of 640 floats, FP16 later), and the last top-k ids per index-source layer within a window.
Graph recording follows GLM: positions, visible counts and block-completion flags live in device buffers the host
fills before each replay, so one graph per window size serves every step.

### Milestone 2: upstream serve fixes

Port order (one commit each, serve tests green after each): tool-call parser (#4701, #4663, #4751, #5276, #5281,
#5409, #5040), request lifecycle (#4271, #4479, #4738, #5044, #4792, #3443, answers one request late), Anthropic context
overflow wording, fatal GPU error restart. Skipped: DFlash2, replicas/shared arena, CUDA/HIP-only defaults, GUI, Metal.

### Milestone 6: MiniMax-M3.1-Flash

Reference graph: llama.cpp `src/models/minimax-m3.cpp` (master 8642541b2). The model is qwen3moe's block with leading
dense FFNs, a sigmoid router with a selection bias, an ungated shared expert, swiglu-oai, and MiniMax Sparse Attention
in the blocks that route. Its measure of correctness is the chat-template reference and `mm_host` (the model in
floats), not raw-text references: the first token's row amplifies activation rounding (STATUS).

1. Done: the model spec, the kernels (`minimax.hpp`, `minimax_parity`), the block in the windows and the prompt
   chunks, `mm_host`, the first runs on three GPUs.
2. Speed: 102 GiB of experts against 72 GiB of VRAM for them. The RAM copy for the rest, the slow-link GPU's spare
   room, a profile at load, the missed experts over PCIe; the selection's scores on the matrix engines.
3. Serve: the chat template, tool calls; then the vision encoder and the DSpark drafter (safetensors files).
