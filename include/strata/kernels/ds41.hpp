// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/kernels/ds41.hpp - DeepSeek-V4.1-Flash's (deepseek41) own pieces for the Xe engine: the
// sqrt-softplus router, the single-pass mHC read, the compressed-KV gate, Engram, YaRN RoPE on part of a head, the
// lightning indexer with its candidate blocks, a deterministic top-k, and MQA with attention sinks over the window and
// the selected compressed rows.  The arithmetic follows llama.cpp's src/models/deepseek4.cpp, deepseek41.cpp,
// src/llama-graph.cpp (build_moe_ffn) and ggml's CPU operators (rope_yarn, softplus) at master 8642541b2; MIT License,
// Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE.  The mHC mixes, post and comb are
// glm5.hpp's (the same ggml_dsv4_hc_comb); the clamped SwiGLU is glm5_swiglu_clamp.
//
// Every buffer is a device pointer unless named host; `stream` null runs synchronously.  T is the number of tokens,
// N the model width, H the attention heads, D a head's width (512), E the routed experts.
#pragma once

#include <cstdint>
#include <vector>

namespace strata::kernels {

/// The sqrt-softplus router (build_moe_ffn, gating SQRT_SOFTPLUS with exp_probs_b): p = sqrt(softplus(logits)),
/// softplus(x) = x above 20 else log(1 + exp(x)); the k experts with the largest p + bias (ties to the lower index);
/// weights p of those, divided by (their sum + 1e-20) when `norm` (DeepSeek-V4.1's model.py), times `scale`.
/// logits [T][E] (F32 accumulated), bias [E]; ids, weights [T][k].  k <= 32.
void ds41_router(const float* logits, const float* bias, int n_tokens, int n_expert, int k, bool norm, float scale,
                 int32_t* ids, float* weights, void* stream);

/// The pre-mix of the mHC streams (build_hc_mixes' `pre`): pre[t][s] = sigmoid(mix[s] scale[0] / rms(x[t]) + base[s])
/// + hc_eps, rms over the token's 4N stream values (eps `rms_eps`).  DeepSeek-V4.1 applies it one sub-layer later
/// (ds41_hc_collapse), so it is kept apart from glm5_hc_pre's read.  x [T][4][N], mixes [T][24], pre [T][4].
void ds41_hc_pre_weights(const float* x, const float* mixes, const float* scale, const float* base, float rms_eps,
                         float hc_eps, float* pre, int64_t n_embd, int64_t n_tokens, void* stream);

/// y[t][d] = sum_s pre[t][s] x[t][s][d] (build_hc_pre with given weights).  x [T][4][N], pre [T][4], y [T][N].
void ds41_hc_collapse(const float* x, const float* pre, float* y, int64_t n_embd, int64_t n_tokens, void* stream);

/// The KV compressor of one block of `ratio` positions (build_attention41): c[d] = sum_j v[j][d] softmax_j(s[j][d]),
/// then RMSNorm with weight `w` (eps `eps`).  v, s [B][ratio][D] (the block's rows of the compressor's kv and gate
/// projections, the state's restored rows first); s may be null for ratio 1 (the gate is 0).  out [B][D].
void ds41_compress(const float* v, const float* s, const float* w, float eps, float* out, int64_t n_blocks,
                   int ratio, int64_t dim, void* stream);

/// Engram's write into the streams (build_engram): kv [T][5N] = engram_kv applied to the token's gathered table rows
/// (key = its first 4N as [4][N], value = its last N); dot[s] = sum_d rmsnorm(x[t][s])[d] rmsnorm(key[s])[d]
/// gate_w[s][d] / sqrt(N) (rmsnorm without weight, eps `rms_eps`); g = sigmoid(sign(dot) sqrt(max(|dot|, 1e-6)))
/// (sign(0) = +1), times keep[t] when `keep` is given; x[t][s][d] += value[d] g, in place.  gate_w [4][N] (GGUF
/// engram_gate, ne0 = N).
void ds41_engram_gate(float* x, const float* kv, const float* gate_w, const float* keep, float rms_eps,
                      int64_t n_embd, int64_t n_tokens, void* stream);

/// Engram's n-gram hash of one token (llama_model_deepseek41::engram_hash), on the host.  `prev` holds the
/// ngram - 1 tokens before it, oldest first (prev[ngram - 2] is the one right before); `pos` the token's position
/// (a shift that reaches before position 0, or a token outside the map, is dead and so is every longer shift; a dead
/// token hashes as the pad).  mult [ngram], primes and offsets [n_heads] of module e (the spec's arrays from
/// e ngram and e n_heads).  out [n_heads] table rows.
void ds41_engram_hash(int32_t token, const int32_t* prev, int64_t pos, const std::vector<int32_t>& token_map,
                      int32_t pad_token_id, int ngram, int heads_per_ngram, const uint64_t* mult,
                      const uint64_t* primes, const uint64_t* offsets, int32_t* out);

/// YaRN RoPE parameters as ggml_rope_ext takes them (NORM type: adjacent pairs).
struct Ds41Rope {
    int n_rot = 64;            ///< rotated dimensions
    int offset = 448;          ///< the first rotated dimension (ggml_rope_set_offset: the head's NoPE part before it)
    int n_ctx_orig = 0;
    float freq_base = 10000.0f, freq_scale = 1.0f, ext_factor = 0.0f, attn_factor = 1.0f;
    float beta_fast = 32.0f, beta_slow = 1.0f;
};

/// RoPE on dims [offset, offset + n_rot) of every head row, in place: x [T][rows][dim], pos [T] (device).  `inverse`
/// rotates back (ggml_rope_ext_back: the same angles, sin negated).
void ds41_rope(float* x, const int32_t* pos, const Ds41Rope& r, bool inverse, int64_t rows, int64_t dim,
               int64_t n_tokens, void* stream);

/// The lightning indexer's scores (build_attention41, unfused path): score[t][c] = sum_h relu(q[t][h] . k[c]) w[t][h]
/// for c < n_visible[t], -inf above.  q [T][H][Di], w [T][H] (indexer_proj scaled by 1/sqrt(Di H)), keys [C][key_stride]
/// with the Di indexer values at key_offset (a compressed row: the latent, then the index key).  scores [T][score_stride].
void ds41_idx_scores(const float* q, const float* w, const float* keys, int64_t key_stride, int64_t key_offset,
                     const int32_t* n_visible, int n_heads, int dim, int64_t n_tokens, int64_t score_stride,
                     float* scores, void* stream);

/// The candidate blocks' scores (level one): bmax[t][b] = max of score[t][b bs .. b bs + bs) over the visible rows,
/// -inf for a block with none, plus 1e30 on the block of the newest visible row ((n_visible - 1) / bs), which is
/// always kept.  bmax [T][blk_stride].
void ds41_block_max(const float* scores, const int32_t* n_visible, int bs, int64_t n_tokens, int64_t score_stride,
                    int64_t blk_stride, float* bmax, void* stream);

/// Level two: the scores of rows outside the token's chosen blocks become -inf.  blocks [T][blk_cap] the block ids
/// (ds41_topk's), n_blocks [T] how many.
void ds41_keep_blocks(float* scores, const int32_t* n_visible, const int32_t* blocks, const int32_t* n_blocks,
                      int64_t blk_cap, int bs, int64_t n_tokens, int64_t score_stride, void* stream);

/// The k largest of score[t][0 .. n[t]) (ties to the lower index; -inf rows are never taken), their indices in
/// ascending order (deterministic attention), count[t] = min(k, n[t] minus its -inf rows).  ids [T][k].
void ds41_topk(const float* scores, const int32_t* n, int k, int64_t n_tokens, int64_t score_stride, int32_t* ids,
               int32_t* count, void* stream);

/// MQA with attention sinks over the window and the selected compressed rows (build_attn_mha with K = V): for head h
/// of token t, the keys are the raw rows of positions [max(0, p - window + 1), p] (row pos % ring of `raw` [ring][D])
/// then the compressed rows ids[t][0 .. count[t]) of `comp` [rows][comp_stride] (first D values); p_j =
/// softmax over the keys and the sink (exp(sinks[h]) in the denominator, no value) of q . k_j scale; out = sum p_j k_j.
/// q, out [T][H][D]; pos [T]; count may be null (window only).
void ds41_sink_attn(const float* q, const float* raw, int64_t ring, const int32_t* pos, int window, const float* comp,
                    int64_t comp_stride, const int32_t* ids, const int32_t* count, int64_t id_cap, const float* sinks,
                    float scale, int n_heads, int dim, int64_t n_tokens, float* out, void* stream);

}  // namespace strata::kernels
