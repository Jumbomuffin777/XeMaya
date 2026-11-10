// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/plain_ds41.cpp - DeepSeek-V4.1-Flash (deepseek41) in PlainModel (plain_model.hpp).
//
// The block follows llama.cpp's deepseek41 graph (src/models/deepseek41.cpp, deepseek4.cpp, src/llama-graph.cpp and
// src/llama-kv-cache-dsv4.cpp at master 8642541b2; MIT License, Copyright (c) 2023-2026 The ggml authors, see
// third_party/main/ggml/LICENSE) step for step, as src/reference/ds41_host.cpp does on the host:
//
//   streams:  four copies of the embedding; an Engram layer first adds its gated n-gram rows to them.  Before the
//             attention and before the FFN the mixes hc_fn . flat give post and comb for the write after it (glm5.hpp)
//             and pre; the row a sub-layer reads is the streams collapsed with the pre of the sub-layer before
//             (layer 0's attention reads stream 0), so pre lags one sub-layer behind
//   MQA:      qr = rms_norm(q_a(h)); q = q_b(qr) per head, kv = rms_norm(kv(h)), both rotated (YaRN on the last 64
//             dims; a compressed layer's base and scaling); kv into the layer's ring.  A KV-source layer compresses
//             each block of `ratio` positions (softmax gate per dim, RMSNorm) into a row with its index key; a top-k
//             source layer scores the visible rows with the lightning indexer (through the candidate blocks) and keeps
//             the best 512.  Attention with sinks over the window's raw rows and the selected compressed rows (K = V),
//             rotated back, then the grouped low-rank output (8 groups through o_a, then o_b)
//   FFN:      the sqrt-softplus router (k experts, weights normalized and scaled), the routed experts through the
//             engine's tiers, the shared expert; every SwiGLU clamped
//
// The per-sequence state lives in device buffers (the rings, the open blocks, the compressed caches), and the host
// stages each window's positions, visible-row counts and Engram table rows, so one recorded graph a window size serves
// every position.  The prompt runs in chunks (src/prefill/plain_prompt_ds41.cpp); the model has no MTP layer.
#include "strata/core/plain_model.hpp"
#include "plain_ds41_rope.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/runtime.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/ds41.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/swiglu_clamp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>

namespace strata::core {
namespace {

namespace k = strata::kernels;

constexpr int kHC = 4, kMix = (2 + kHC) * kHC;
constexpr int kBF16 = 30;
constexpr int kQ6K = 14, kQ80 = 8;

bool mapped(size_t bytes, void** h, void** d) {
    if (!strata::gpu::alloc_host(h, bytes)) return false;
    std::memset(*h, 0, bytes);
    return strata::gpu::device_pointer(d, *h);
}

/// A projection's GGUF type before the engine's row order (the byte count of its rows)
int file_type(int t) { return t == k::kNativeQ6KRows ? kQ6K : t == k::kNativeQ8Rows ? kQ80 : t; }

/// y [T][n_out] = w x for an F32 matrix, 8 rows a launch (f32_gemv_fp32_mmvf_multi's limit)
void f32_rows(const float* x, int64_t n_in, const float* w, float* y, int64_t n_out, int T, void* cs) {
    for (int t0 = 0; t0 < T; t0 += 8)
        k::f32_gemv_fp32_mmvf_multi(x + (size_t) t0 * (size_t) n_in, n_in, w, y + (size_t) t0 * (size_t) n_out, n_out,
                                    n_in, n_out, std::min(8, T - t0), cs);
}

}  // namespace

bool PlainModel::load_dblock(const GgufModel& m, int64_t l, DBlock& b, std::string& err) {
    const ModelSpec& s = spec_;
    const uint64_t N = (uint64_t) s.n_embd, E = (uint64_t) s.n_expert;
    const int64_t H = s.n_head, D = s.key_length, QL = s.q_lora_rank, G = s.o_group_count, OL = s.o_lora_rank;
    const int64_t ID = s.idx_key_length, IH = s.idx_n_head, FS = s.n_ff_shexp;
    const std::string p = "blk." + std::to_string(l) + ".";
    auto bad = [&](const char* what) {
        err = "plain model: block " + std::to_string(l) + "'s " + what + " have other shapes";
        return false;
    };
    if (!upload_f32(m, p + "attn_norm.weight", N, b.attn_norm, err) ||
        !upload_f32(m, p + "ffn_norm.weight", N, b.ffn_norm, err) ||
        !upload_f32(m, p + "hc_attn_fn.weight", kHC * N * kMix, b.hc_attn_fn, err) ||
        !upload_f32(m, p + "hc_attn_base.weight", kMix, b.hc_attn_base, err) ||
        !upload_f32(m, p + "hc_attn_scale.weight", 3, b.hc_attn_scale, err) ||
        !upload_f32(m, p + "hc_ffn_fn.weight", kHC * N * kMix, b.hc_ffn_fn, err) ||
        !upload_f32(m, p + "hc_ffn_base.weight", kMix, b.hc_ffn_base, err) ||
        !upload_f32(m, p + "hc_ffn_scale.weight", 3, b.hc_ffn_scale, err))
        return false;
    if (!upload(m, p + "attn_q_a.weight", b.q_a, err) || !upload(m, p + "attn_q_b.weight", b.q_b, err) ||
        !upload(m, p + "attn_kv.weight", b.kv, err) || !upload(m, p + "attn_output_a.weight", b.o_a, err) ||
        !upload(m, p + "attn_output_b.weight", b.o_b, err) ||
        !upload_f32(m, p + "attn_q_a_norm.weight", (uint64_t) QL, b.q_a_norm, err) ||
        !upload_f32(m, p + "attn_kv_a_norm.weight", (uint64_t) D, b.kv_norm, err) ||
        !upload_f32(m, p + "attn_sinks.weight", (uint64_t) H, b.sinks, err))
        return false;
    if ((uint64_t) b.q_a.n_in != N || b.q_a.n_out != QL || b.q_b.n_in != QL || b.q_b.n_out != H * D ||
        (uint64_t) b.kv.n_in != N || b.kv.n_out != D || b.o_a.n_in != H * D / G || b.o_a.n_out != OL * G ||
        b.o_b.n_in != G * OL || (uint64_t) b.o_b.n_out != N)
        return bad("attention projections");
    if (b.kv_src == l) {
        if (!upload_f32(m, p + "attn_compressor_kv.weight", N * (uint64_t) D, b.comp_kv, err) ||
            !upload_f32(m, p + "attn_compressor_norm.weight", (uint64_t) D, b.comp_norm, err) ||
            (b.ratio > 1 && !upload_f32(m, p + "attn_compressor_gate.weight", N * (uint64_t) D, b.comp_gate, err)) ||
            !upload_f32(m, p + "indexer.attn_k.weight", (uint64_t) (D * ID), b.idx_k, err) ||
            !upload_f32(m, p + "indexer.k_norm.weight", (uint64_t) ID, b.idx_k_norm, err))
            return false;
    }
    if (b.topk_src == l) {
        if (!upload_f32(m, p + "indexer.attn_q_b.weight", (uint64_t) (QL * IH * ID), b.idx_q_b, err) ||
            !upload_f32(m, p + "indexer.proj.weight", N * (uint64_t) IH, b.idx_proj, err))
            return false;
    }
    if (b.eng >= 0) {
        if (!upload(m, p + "engram_kv.weight", b.eng_kv, err) ||
            !upload_f32(m, p + "engram_gate.weight", N * kHC, b.eng_gate, err))
            return false;
        if (b.eng_kv.n_in != s.engram_n_heads * s.engram_head_dim || (uint64_t) b.eng_kv.n_out != N * (kHC + 1))
            return bad("Engram projection");
    }
    if (!upload(m, p + "ffn_gate_shexp.weight", b.ffn_gate, err) || !upload(m, p + "ffn_up_shexp.weight", b.ffn_up, err) ||
        !upload(m, p + "ffn_down_shexp.weight", b.ffn_down, err) ||
        !upload_f32(m, p + "ffn_gate_inp.weight", N * E, b.router, err) ||
        !upload_f32(m, p + "exp_probs_b.bias", E, b.router_b, err))
        return false;
    if ((uint64_t) b.ffn_gate.n_in != N || b.ffn_gate.n_out != FS || (uint64_t) b.ffn_up.n_in != N ||
        b.ffn_up.n_out != FS || b.ffn_down.n_in != FS || (uint64_t) b.ffn_down.n_out != N)
        return bad("shared expert");
    b.clamp = (size_t) l < s.swiglu_clamp_shexp.size() ? (float) s.swiglu_clamp_shexp[(size_t) l] : 0.0f;
    // the native matrix-vector kernels' formats only (the files keep these in Q8_0 and Q5_K)
    for (const W* w : {&b.q_a, &b.q_b, &b.kv, &b.o_a, &b.o_b, &b.eng_kv, &b.ffn_gate, &b.ffn_up, &b.ffn_down})
        if (w->data != nullptr && (w->type == kBF16 || w->n_in % 32 != 0)) {
            err = "plain model: block " + std::to_string(l) + " has a BF16 projection the DeepSeek-V4.1 block does not "
                  "take yet";
            return false;
        }
    return true;
}

bool PlainModel::load_ds41(std::string& err) {
    const ModelSpec& s = spec_;
    ds_ = true;
    if (opt_.kv_int8) { err = "plain model: --kv int8 is not available for DeepSeek-V4.1-Flash"; return false; }
    const int64_t N = s.n_embd, H = s.n_head, D = s.key_length, QL = s.q_lora_rank, G = s.o_group_count,
                  OL = s.o_lora_rank, ID = s.idx_key_length, IH = s.idx_n_head, FS = s.n_ff_shexp, E = s.n_expert,
                  K = s.n_expert_used, L = s.n_layer, WIN = s.sliding_window;
    // the shapes the kernels and this block are written for
    if (s.hc != kHC || H <= 0 || D <= 0 || D % 32 != 0 || N % 256 != 0 || QL % 32 != 0 || (H * D / G) % 32 != 0 ||
        (G * OL) % 32 != 0 || FS <= 0 || FS % 32 != 0 || K < 1 || K > 15 || E < K || ID <= 0 || IH <= 0 || WIN <= 0 ||
        s.n_dense_lead != 0 || (int64_t) s.compress_ratios.size() < L) {
        err = "plain model: a DeepSeek-V4.1-Flash shape this engine's kernels are not built for";
        return false;
    }
    // the prompt in chunks (plain_prompt_ds41.cpp) of at most kDsChunk tokens: the rings, the scores and the
    // attention's rows grow with the chunk.  XEMAYA_DS41_PROMPT=windows: through the decode windows.  No drafts
    {
        constexpr int64_t kDsChunk = 1024;
        const char* v = std::getenv("XEMAYA_DS41_PROMPT");
        const char* c = std::getenv("XEMAYA_DS41_CHUNK");
        const int64_t cap = c != nullptr && std::atoll(c) >= 16 ? std::atoll(c) : kDsChunk;
        opt_.prompt_chunk = v != nullptr && std::strcmp(v, "windows") == 0 ? 0 : std::min<int64_t>(opt_.prompt_chunk, cap);
        opt_.prompt_chunk_given = true;   // no longer chunk is chosen (auto_prompt_chunk)
    }
    opt_.spec = 0;
    model_mtp_ = false;
    has_mtp_ = false;
    d_ratios_.clear();
    for (int64_t l = 0; l < L; ++l) {
        const int64_t r = s.compress_ratios[(size_t) l];
        if (r > 8) { err = "plain model: a compress ratio above 8"; return false; }
        if (r > 0 && std::find(d_ratios_.begin(), d_ratios_.end(), r) == d_ratios_.end()) d_ratios_.push_back(r);
    }
    if ((int) d_ratios_.size() > kDsRatios) { err = "plain model: more than 4 distinct compress ratios"; return false; }
    // a layer split cuts anywhere: what the blocks after a cut read of the blocks before it (a KV source's compressed
    // rows, a selection, the candidate blocks) crosses with the streams (DsCarry)
    d_in_ = DsCarry();
    d_out_ = DsCarry();
    if ((!first() && !ds41_carry(lb_, d_in_, err)) || (!last() && !ds41_carry(le_, d_out_, err))) return false;
    {
        const int64_t w[kDsHSections] = {kHC * N, kHC, D, ID, std::max<int64_t>(1, s.idx_top_k), 1,
                                         std::max<int64_t>(1, s.cand_topk_blocks), 1};
        d_hoff_[0] = 0;
        for (int k = 0; k < kDsHSections; ++k) d_hoff_[k + 1] = d_hoff_[k] + w[k];
    }
    cs_ = strata::gpu::stream_create();
    if (cs_ == nullptr) { err = "plain model: stream create failed"; return false; }
    {   // the routed experts' SwiGLU limit: one for every MoE layer (a process setting, before any graph)
        float lim = 0.0f;
        bool one = true;
        for (int64_t l = 0; l < L; ++l) {
            const float v = (size_t) l < s.swiglu_clamp_exp.size() ? (float) s.swiglu_clamp_exp[(size_t) l] : 0.0f;
            if (l == 0) lim = v;
            else one = one && v == lim;
        }
        if (!one) { err = "plain model: the experts' SwiGLU limits differ between layers"; return false; }
        k::expert_swiglu_set_limit(lim);
    }

    const GgufModel& m = *model_;
    {   // the embedding stays in the mapping: a token's row is dequantized on the host
        size_t at = 0;
        const TensorInfo* t = m.find("token_embd.weight", &at);
        if (t == nullptr || t->shape.size() != 2 || t->shape[0] != (uint64_t) N || !m.in_bounds(*t, at)) {
            err = "plain model: token_embd.weight is missing or of another width";
            return false;
        }
        emb_type_ = (int) t->type;
        n_vocab_ = (int64_t) t->shape[1];
        emb_row_ = k::cpu::native_row_bytes(emb_type_, N);
        if (emb_row_ == 0) { err = "plain model: token_embd.weight's type has no host dequantizer"; return false; }
        emb_ = m.shard(at).tensor_data(*t);
    }
    if (last()) {   // the head: the last stage's
        if (!upload(m, "output.weight", head_, err) || !upload_f32(m, "output_norm.weight", (uint64_t) N, out_norm_, err))
            return false;
        if (head_.n_in != N || head_.n_out != n_vocab_ || head_.type == kBF16) {
            err = "plain model: output.weight's shape or type";
            return false;
        }
    }
    dblocks_.assign((size_t) L, {});
    d_eng_local_.clear();
    for (int64_t l = 0; l < L; ++l) {
        DBlock& b = dblocks_[(size_t) l];
        b.ratio = s.compress_ratios[(size_t) l];
        b.kv_src = b.ratio ? s.kv_src[(size_t) l] : -1;
        b.topk_src = b.ratio ? s.topk_src[(size_t) l] : -1;
        b.eng = s.engram_index(l);
        if (l < lb_ || l >= le_) continue;
        if (!load_dblock(m, l, b, err)) return false;
        if (b.eng >= 0) d_eng_local_.push_back(l);
    }
    if (!d_eng_local_.empty()) {   // the Engram tables: one tensor, read by row from the mapping
        size_t at = 0;
        const TensorInfo* t = m.find("per_layer_token_embd.weight", &at);
        if (t == nullptr || t->shape.size() != 2 || t->shape[0] != (uint64_t) s.engram_head_dim || !m.in_bounds(*t, at)) {
            err = "plain model: per_layer_token_embd.weight is missing or of another width";
            return false;
        }
        d_eng_type_ = (int) t->type;
        d_eng_row_ = k::cpu::native_row_bytes(d_eng_type_, s.engram_head_dim);
        if (d_eng_row_ == 0) { err = "plain model: the Engram table's type has no host dequantizer"; return false; }
        d_eng_tab_ = m.shard(at).tensor_data(*t);
        d_eng_rows_n_ = t->shape[1];
    }

    // the per-sequence state
    const int64_t cells = opt_.max_cells, MT = opt_.max_t;
    d_ring_ = WIN + std::max<int64_t>(MT, opt_.prompt_chunk);
    d_raw_.assign((size_t) L, nullptr);
    d_cstate_.assign((size_t) L, nullptr);
    d_comp_.assign((size_t) L, nullptr);
    d_rows_.assign((size_t) L, 0);
    d_score_stride_ = 1;
    // a KV source's compressed cache: this stage's own blocks', and its copy of the one carried over the cut before it
    auto comp_cache = [&](int64_t l) {
        d_rows_[(size_t) l] = cells / dblocks_[(size_t) l].ratio + 1;
        d_score_stride_ = std::max(d_score_stride_, d_rows_[(size_t) l]);
        float* c = nullptr;
        if (!strata::gpu::alloc_device(&c, (size_t) (d_rows_[(size_t) l] * (D + ID)) * 4)) {
            err = "plain model: no VRAM for the compressed KV caches (" + std::to_string(d_rows_[(size_t) l]) + " rows)";
            return false;
        }
        allocs_.push_back(c);
        d_comp_[(size_t) l] = c;
        return true;
    };
    if (d_in_.kv >= 0 && !comp_cache(d_in_.kv)) return false;
    for (int64_t l = lb_; l < le_; ++l) {
        const DBlock& b = dblocks_[(size_t) l];
        float* p = nullptr;
        if (!strata::gpu::alloc_device(&p, (size_t) (d_ring_ * D) * 4)) { err = "plain model: no VRAM for the KV rings"; return false; }
        allocs_.push_back(p);
        d_raw_[(size_t) l] = p;
        if (b.kv_src != l) continue;
        float* st = nullptr;
        if (!comp_cache(l)) return false;
        if (!strata::gpu::alloc_device(&st, (size_t) (std::max<int64_t>(1, b.ratio - 1) * 2 * D) * 4)) {
            err = "plain model: no VRAM for the compressor's open blocks";
            return false;
        }
        allocs_.push_back(st);
        d_cstate_[(size_t) l] = st;
    }
    const int64_t bs = std::max<int64_t>(1, s.cand_block_size);
    d_blk_stride_ = (d_score_stride_ + bs - 1) / bs;
    const int64_t cand_k = std::max<int64_t>(1, s.cand_topk_blocks), top_k = std::max<int64_t>(1, s.idx_top_k);

    // the window's buffers
    const uint64_t T = (uint64_t) MT, UN = (uint64_t) N, UK = (uint64_t) K;
    const uint64_t GI = (uint64_t) (H * D / G), RW = (uint64_t) (s.engram_n_heads * s.engram_head_dim);
    hand_w_ = d_hoff_[kDsHSections];   // a stage hands on the four streams, the lagging pre and its cut's carry
    if (!mapped(T * UN * 4, (void**) &h_emb_, (void**) &m_emb_) ||
        !mapped(T * (uint64_t) kDsMeta * 4, (void**) &h_dmeta_, (void**) &m_dmeta_) ||
        !mapped((uint64_t) cells * 4 + 64, (void**) &h_dtok_, (void**) &m_dtok_) ||
        !mapped(T * kHC * 4, (void**) &h_done_, (void**) &m_done_) ||
        (!d_eng_local_.empty() &&
         !mapped((uint64_t) d_eng_local_.size() * T * RW * 4, (void**) &h_deng_, (void**) &m_deng_)) ||
        !mapped(T * 4 + 16, (void**) &h_out_, (void**) &m_out_) || !mapped(16, (void**) &h_prob_, (void**) &m_prob_) ||
        (!first() && !mapped(T * (uint64_t) hand_w_ * 4, (void**) &h_in_, (void**) &m_in_)) ||
        (!last() && !mapped(T * (uint64_t) hand_w_ * 4, (void**) &h_hand_, (void**) &m_hand_))) {
        err = "plain model: mapped staging allocation failed";
        return false;
    }
    for (uint64_t t = 0; t < T; ++t) h_done_[t * kHC] = 1.0f;
    const size_t xq = std::max({k::native_q8_1_bytes((int) N, (int) T), k::native_q8_1_bytes((int) QL, (int) T),
                                k::native_q8_1_bytes((int) FS, (int) T), k::native_q8_1_bytes((int) (G * OL), (int) T),
                                k::native_q8_1_bytes((int) std::max<uint64_t>(RW, 32), (int) T),
                                k::native_q8_1_rows_bytes((int) GI, (int64_t) (G * T))});
    auto dev = [&](auto** p, uint64_t bytes) {
        void* d = nullptr;
        if (!strata::gpu::alloc_device(&d, (size_t) std::max<uint64_t>(bytes, 256))) return false;
        allocs_.push_back(d);
        *p = (std::remove_reference_t<decltype(*p)>) d;
        return true;
    };
    const uint64_t HD = (uint64_t) (H * D);
    if (!dev(&d_meta_, T * (uint64_t) kDsMeta * 4) || !dev(&x_, T * UN * 4) || !dev(&hn_, T * UN * 4) ||
        !dev(&mix_, T * UN * 4) || !dev(&g_xq_, xq) || !dev(&g_hs_, T * kHC * UN * 4) || !dev(&g_mixes_, T * kMix * 4) ||
        !dev(&g_post_, T * kHC * 4) || !dev(&g_comb_, T * kHC * kHC * 4) || !dev(&g_y_, T * UN * 4) ||
        !dev(&d_pre_[0], T * kHC * 4) || !dev(&d_pre_[1], T * kHC * 4) || !dev(&d_pre_[2], T * kHC * 4) ||
        // the attention
        !dev(&d_qa_, T * (uint64_t) QL * 4) || !dev(&d_qr_, T * (uint64_t) QL * 4) || !dev(&d_q_, T * HD * 4) ||
        !dev(&d_kvr_, T * (uint64_t) D * 4) || !dev(&d_kv_, T * (uint64_t) D * 4) || !dev(&d_cv_, T * (uint64_t) D * 4) ||
        !dev(&d_cg_, T * (uint64_t) D * 4) || !dev(&d_lat_, T * (uint64_t) D * 4) ||
        !dev(&d_ikr_, T * (uint64_t) ID * 4) || !dev(&d_ik_, T * (uint64_t) ID * 4) ||
        !dev(&d_iq_, T * (uint64_t) (IH * ID) * 4) || !dev(&d_iw_, T * (uint64_t) IH * 4) ||
        !dev(&d_scores_, T * (uint64_t) d_score_stride_ * 4) || !dev(&d_bmax_, T * (uint64_t) d_blk_stride_ * 4) ||
        !dev(&d_ids_, T * (uint64_t) top_k * 4) || !dev(&d_cnt_, T * 4) || !dev(&d_cand_, T * (uint64_t) cand_k * 4) ||
        !dev(&d_ncand_, T * 4) || !dev(&d_att_, T * HD * 4) || !dev(&d_attT_, T * HD * 4) ||
        !dev(&d_oaT_, T * (uint64_t) (G * OL) * 4) || !dev(&d_oa_, T * (uint64_t) (G * OL) * 4) ||
        !dev(&d_engr_, T * std::max<uint64_t>(RW, 1) * 4) || !dev(&d_engkv_, T * UN * (kHC + 1) * 4) ||
        // the FFN
        !dev(&g_ffg_, T * (uint64_t) FS * 4) || !dev(&g_ffu_, T * (uint64_t) FS * 4) || !dev(&rlog_, T * (uint64_t) E * 4) ||
        !dev(&ids_, T * UK * 4) || !dev(&w_, T * UK * 4) || !dev(&shared_, T * UN * 4) || !dev(&moe_out_, T * UN * 4) ||
        !dev(&logits_, T * (uint64_t) n_vocab_ * 4)) {
        err = "plain model: no VRAM for the window's buffers";
        return false;
    }
    xq_ = g_xq_;   // the head's q8_1 rows (head_logits)
    if (!strata::gpu::stream_sync(cs_)) { err = "plain model: the uploads failed"; return false; }
    if (!open_experts(err)) return false;
    exec_.assign((size_t) opt_.max_t + 1, nullptr);
    segs_.assign((size_t) opt_.max_t + 1, {});
    mexec_.assign((size_t) opt_.max_t + 1, nullptr);
    msegs_.assign((size_t) opt_.max_t + 1, {});
    if (root_ == this) {
        int64_t n_kv = 0, n_idx = 0;
        for (int64_t l = 0; l < L; ++l) {
            n_kv += s.compress_ratios[(size_t) l] && s.kv_src[(size_t) l] == l;
            n_idx += s.compress_ratios[(size_t) l] && s.topk_src[(size_t) l] == l;
        }
        std::fprintf(stderr, "strata plain: DeepSeek-V4.1-Flash, %lld blocks (%lld KV-source, %lld top-k source), %lld "
                             "experts, %zu Engram layers, window %lld, %lld cells; the prompt in %s of %lld\n",
                     (long long) L, (long long) n_kv, (long long) n_idx, (long long) E, s.engram_layers.size(),
                     (long long) WIN, (long long) cells, opt_.prompt_chunk > 0 ? "chunks" : "windows",
                     (long long) (opt_.prompt_chunk > 0 ? opt_.prompt_chunk : opt_.max_t));
    }
    return true;
}

bool PlainModel::ds41_carry(int64_t c, DsCarry& carry, std::string& err) const {
    const ModelSpec& s = spec_;
    carry = DsCarry();
    for (int64_t l = c; l < s.n_layer; ++l) {
        const int64_t r = s.compress_ratios[(size_t) l];
        if (r <= 0) continue;
        const int64_t kv = s.kv_src[(size_t) l];
        if (kv < c) {
            if (carry.kv >= 0 && carry.kv != kv) {
                err = "plain model: DeepSeek-V4.1-Flash cannot be split before block " + std::to_string(c) +
                      " (the blocks after it read two compressed caches made before it)";
                return false;
            }
            carry.kv = kv;
        }
        carry.ids = carry.ids || s.topk_src[(size_t) l] < c;
        carry.cand = carry.cand || (s.topk_src[(size_t) l] == l && s.cand_src >= 0 && s.cand_src < c &&
                                    s.compress_ratios[(size_t) s.cand_src] == r);
    }
    return true;
}

std::vector<PlainModel::StatePart> PlainModel::ds41_recurrent_parts() const {
    std::vector<StatePart> parts;
    if (!ds_) return parts;
    const int64_t D = spec_.key_length;
    for (int64_t l = lb_; l < le_; ++l) {
        parts.push_back({d_raw_[(size_t) l], (size_t) (d_ring_ * D) * 4});
        if (d_cstate_[(size_t) l] != nullptr)
            parts.push_back({d_cstate_[(size_t) l],
                             (size_t) (std::max<int64_t>(1, dblocks_[(size_t) l].ratio - 1) * 2 * D) * 4});
    }
    return parts;
}

bool PlainModel::reset_ds41(std::string& err) {
    bool ok = true;
    const int64_t D = spec_.key_length, ID = spec_.idx_key_length;
    for (const StatePart& p : ds41_recurrent_parts()) ok = ok && strata::gpu::memset(p.dev, 0, p.bytes);
    for (int64_t l = 0; l < le_; ++l)   // the stage's own caches and its copy of a carried one
        if (d_comp_[(size_t) l] != nullptr)
            ok = ok && strata::gpu::memset(d_comp_[(size_t) l], 0, (size_t) (d_rows_[(size_t) l] * (D + ID)) * 4);
    std::memset(h_dtok_, 0, (size_t) opt_.max_cells * 4);
    cur_ = -1;
    drafts_.clear();
    if (!ok) err = "plain model: resetting the state failed";
    return ok;
}

void PlainModel::ds41_meta(int64_t pos, int64_t T, int64_t stride, int32_t* meta) const {
    const ModelSpec& s = spec_;
    const int64_t bs = std::max<int64_t>(1, s.cand_block_size);
    const int64_t cand_ratio = s.cand_src >= 0 ? s.compress_ratios[(size_t) s.cand_src] : 0;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t p = pos + t;
        meta[t] = (int32_t) p;
        for (size_t ri = 0; ri < d_ratios_.size(); ++ri) {
            const int64_t r = d_ratios_[ri];
            meta[(1 + 2 * (int64_t) ri) * stride + t] = (int32_t) ((p + 1) / r);
            meta[(2 + 2 * (int64_t) ri) * stride + t] = (int32_t) std::max<int64_t>(0, p - r + 1);
        }
        meta[(1 + 2 * kDsRatios) * stride + t] = cand_ratio > 0 ? (int32_t) (((p + 1) / cand_ratio + bs - 1) / bs) : 0;
    }
}

bool PlainModel::ds41_engram_rows(int64_t e, int64_t p, float* dst) const {
    const ModelSpec& s = spec_;
    const int64_t NH = s.engram_n_heads, HD = s.engram_head_dim, ng = s.engram_ngram;
    int32_t rows[64], prev[16] = {};
    if (NH > 64 || ng > 16 || ng < 2) return false;
    for (int64_t i = 0; i < ng - 1; ++i) {   // oldest first; a position before 0 is not read
        const int64_t q = p - (ng - 1) + i;
        prev[i] = q >= 0 ? h_dtok_[q] : 0;
    }
    k::ds41_engram_hash(h_dtok_[p], prev, p, s.engram_token_map, (int32_t) s.engram_pad_id, (int) ng,
                        (int) s.engram_heads_per_ngram, s.engram_multipliers.data() + e * ng,
                        s.engram_vocab.data() + e * NH, s.engram_offsets.data() + e * NH, rows);
    for (int64_t c = 0; c < NH; ++c)
        if (rows[c] < 0 || (uint64_t) rows[c] >= d_eng_rows_n_ ||
            !k::cpu::native_row_to_float(d_eng_type_, d_eng_tab_ + (size_t) rows[c] * d_eng_row_, dst + c * HD, HD))
            return false;
    return true;
}

bool PlainModel::stage_ds41(const int32_t* tokens, int T, int64_t pos, std::string& err, bool emb) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, MT = opt_.max_t;
    for (int t = 0; t < T; ++t) {
        const int64_t p = pos + t;
        if (tokens[t] < 0 || tokens[t] >= n_vocab_) { err = "plain model: a token is out of the vocabulary"; return false; }
        if (emb && !embed(tokens[t], p, h_emb_ + (size_t) t * (size_t) N)) {
            err = "plain model: the embedding row could not be dequantized";
            return false;
        }
        h_dtok_[p] = tokens[t];
    }
    ds41_meta(pos, T, MT, h_dmeta_);
    // Engram: each token's n-gram hashes (the tokens before it from the sequence so far), their table rows dequantized
    const int64_t RW = s.engram_n_heads * s.engram_head_dim;
    for (size_t j = 0; j < d_eng_local_.size(); ++j)
        for (int t = 0; t < T; ++t)
            if (!ds41_engram_rows(dblocks_[(size_t) d_eng_local_[j]].eng, pos + t,
                                  h_deng_ + ((size_t) j * (size_t) MT + (size_t) t) * (size_t) RW)) {
                err = "plain model: an Engram table row could not be read";
                return false;
            }
    return true;
}

void PlainModel::record_dattn(const DBlock& b, int64_t l, int T, strata::gpu::Stream cs) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, H = s.n_head, D = s.key_length, QL = s.q_lora_rank, G = s.o_group_count,
                  OL = s.o_lora_rank, ID = s.idx_key_length, IH = s.idx_n_head, MT = opt_.max_t, GI = H * D / G;
    const float eps = (float) s.rms_eps;
    const k::Ds41Rope r = ds41_layer_rope(s, l, D), ri = ds41_layer_rope(s, l, ID);
    const int32_t* pos = d_meta_;
    const int32_t *nvis = nullptr, *bpos = nullptr;
    if (b.ratio > 0) {
        const int64_t i = std::find(d_ratios_.begin(), d_ratios_.end(), b.ratio) - d_ratios_.begin();
        nvis = d_meta_ + (1 + 2 * i) * MT;
        bpos = d_meta_ + (2 + 2 * i) * MT;
    }
    const bool kv_here = b.ratio > 0 && b.kv_src == l, idx_here = b.ratio > 0 && b.topk_src == l;
    // from the block row hn_ (its q8_1 in g_xq_): q_a, kv, the compressor's and the indexer's F32 products
    gproj(b.q_a, hn_, d_qa_, T, true, cs);
    gproj(b.kv, hn_, d_kvr_, T, true, cs);
    if (kv_here) {
        f32_rows(hn_, N, b.comp_kv, d_cv_, D, T, cs);
        if (b.ratio > 1) f32_rows(hn_, N, b.comp_gate, d_cg_, D, T, cs);
    }
    if (idx_here) {
        f32_rows(hn_, N, b.idx_proj, d_iw_, IH, T, cs);
        k::scale_inplace(d_iw_, (int64_t) T * IH, 1.0f / std::sqrt((float) (ID * IH)), cs);
    }
    k::native_qsa_rms_norm_weighted(d_kvr_, b.kv_norm, d_kv_, (int) D, T, eps, cs);
    k::native_qsa_rms_norm_weighted_q8_1(d_qa_, b.q_a_norm, d_qr_, g_xq_, (int) QL, T, eps, cs);
    gproj(b.q_b, d_qr_, d_q_, T, true, cs);
    if (idx_here) f32_rows(d_qr_, QL, b.idx_q_b, d_iq_, IH * ID, T, cs);
    dprof(1, cs);
    k::ds41_rope(d_q_, pos, r, false, H, D, T, cs);
    k::ds41_rope(d_kv_, pos, r, false, 1, D, T, cs);
    k::ds41_ring_store(d_kv_, pos, d_ring_, D, T, d_raw_[(size_t) l], cs);
    if (kv_here) {   // the blocks this window closes: their compressed rows and index keys, at the blocks' first positions
        k::ds41_compress_stream(d_cv_, b.ratio > 1 ? d_cg_ : nullptr, d_cstate_[(size_t) l], pos, b.comp_norm, eps,
                                (int) b.ratio, D, T, d_lat_, cs);
        f32_rows(d_lat_, D, b.idx_k, d_ikr_, ID, T, cs);
        k::native_qsa_rms_norm_weighted(d_ikr_, b.idx_k_norm, d_ik_, (int) ID, T, eps, cs);
        k::ds41_rope(d_lat_, bpos, r, false, 1, D, T, cs);
        k::ds41_rope(d_ik_, bpos, ri, false, 1, ID, T, cs);
        k::ds41_comp_store(d_lat_, d_ik_, pos, (int) b.ratio, D, ID, T, d_comp_[(size_t) l], d_rows_[(size_t) l], cs);
    }
    const float* comp = b.ratio > 0 ? d_comp_[(size_t) b.kv_src] : nullptr;
    if (idx_here) {   // the lightning indexer over the visible rows, through the candidate blocks
        const int64_t bs = std::max<int64_t>(1, s.cand_block_size), cand_k = std::max<int64_t>(1, s.cand_topk_blocks);
        k::ds41_rope(d_iq_, pos, ri, false, IH, ID, T, cs);
        k::ds41_idx_scores(d_iq_, d_iw_, comp, D + ID, D, nvis, (int) IH, (int) ID, T, d_score_stride_, d_scores_, cs);
        if (s.cand_src == l) {
            k::ds41_block_max(d_scores_, nvis, (int) bs, T, d_score_stride_, d_blk_stride_, d_bmax_, cs);
            k::ds41_topk(d_bmax_, d_meta_ + (1 + 2 * kDsRatios) * MT, (int) cand_k, T, d_blk_stride_, d_cand_, d_ncand_,
                         cs);
        } else if (s.cand_src >= 0 && s.cand_src < l && s.compress_ratios[(size_t) s.cand_src] == b.ratio) {
            k::ds41_keep_blocks(d_scores_, nvis, d_cand_, d_ncand_, cand_k, (int) bs, T, d_score_stride_, cs);
        }
        k::ds41_topk(d_scores_, nvis, (int) s.idx_top_k, T, d_score_stride_, d_ids_, d_cnt_, cs);
    }
    dprof(2, cs);
    if (k::ds41_attn_mode() < 2 ||
        !k::ds41_sink_attn_xmx(d_q_, d_raw_[(size_t) l], d_ring_, pos, (int) s.sliding_window, comp, D + ID,
                               b.ratio > 0 ? d_ids_ : nullptr, b.ratio > 0 ? d_cnt_ : nullptr, s.idx_top_k, b.sinks,
                               1.0f / std::sqrt((float) D), (int) H, (int) D, T, d_att_, cs))
        k::ds41_sink_attn(d_q_, d_raw_[(size_t) l], d_ring_, pos, (int) s.sliding_window, comp, D + ID,
                          b.ratio > 0 ? d_ids_ : nullptr, b.ratio > 0 ? d_cnt_ : nullptr, s.idx_top_k, b.sinks,
                          1.0f / std::sqrt((float) D), (int) H, (int) D, T, d_att_, cs);
    k::ds41_rope(d_att_, pos, r, true, H, D, T, cs);
    dprof(3, cs);
    // the grouped low-rank output: each group's rows of every token through its o_a, then o_b
    const size_t ab = k::native_mmvq_weight_bytes(file_type(b.o_a.type), (int) GI, (int) OL);
    k::glm5_transpose_abd(d_att_, d_attT_, T, G, GI, cs);
    k::native_quantize_q8_1_rows(d_attT_, g_xq_, (int) GI, G * T, cs);
    for (int64_t g = 0; g < G; ++g)
        k::native_mmvq(b.o_a.type, (const uint8_t*) b.o_a.data + (size_t) g * ab,
                       g_xq_ + k::native_q8_1_rows_bytes((int) GI, g * T), d_oaT_ + (size_t) (g * T * OL), (int) GI,
                       (int) OL, T, cs);
    k::glm5_transpose_abd(d_oaT_, d_oa_, G, T, OL, cs);
    gproj(b.o_b, d_oa_, mix_, T, false, cs);
    dprof(4, cs);
}

void PlainModel::dprof(int k, strata::gpu::Stream cs) {
    if (!dprof_on_) return;
    strata::gpu::stream_sync(cs);
    const auto now = std::chrono::steady_clock::now();
    dprof_ms_[k] += std::chrono::duration<double, std::milli>(now - dprof_t_).count();
    dprof_t_ = now;
}

bool PlainModel::record_dmoe(const DBlock& b, int64_t ml, int T, uint32_t& ring, std::vector<strata::gpu::Graph*>& segs,
                             strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, E = s.n_expert, K = s.n_expert_used;
    f32_rows(hn_, N, b.router, rlog_, E, T, cs);
    const float scale = s.expert_weights_scale > 0 ? (float) s.expert_weights_scale : 1.0f;
    k::ds41_router(rlog_, b.router_b, T, (int) E, (int) K, s.expert_weights_norm && K > 1, scale, ids_, w_, cs);
    moe_publish(ring, T, cs);
    {   // the shared expert while the host serves: gate / up from hn_'s q8_1, the clamp, down
        const int FW = b.ffn_gate.n_out;
        if (b.ffn_gate.type != b.ffn_up.type ||
            !k::native_mmvq_pair(b.ffn_gate.type, b.ffn_gate.data, b.ffn_up.data, g_xq_, g_ffg_, g_ffu_, (int) N, FW, T,
                                 cs)) {
            k::native_mmvq(b.ffn_gate.type, b.ffn_gate.data, g_xq_, g_ffg_, (int) N, FW, T, cs);
            k::native_mmvq(b.ffn_up.type, b.ffn_up.data, g_xq_, g_ffu_, (int) N, FW, T, cs);
        }
        k::glm5_swiglu_clamp(g_ffg_, g_ffu_, g_ffg_, (int64_t) T * FW, b.clamp, cs);
        gproj(b.ffn_down, g_ffg_, shared_, T, false, cs);
    }
    dprof(5, cs);
    if (!moe_experts(ml, ring, T, segs, cs, err)) return false;
    dprof(6, cs);
    k::native_moe_combine_multi(win_.parts(), w_, shared_, moe_out_, N, K, T, cs);
    return true;
}

bool PlainModel::record_window_ds41(int T, strata::gpu::Stream cs, std::string& err) {
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, MT = opt_.max_t;
    const float eps = (float) s.rms_eps, hc_eps = (float) s.hc_eps;
    const int iters = (int) s.hc_sinkhorn_iters;
    if (dprof_on_) {
        strata::gpu::stream_sync(cs);
        dprof_t_ = std::chrono::steady_clock::now();
    }
    k::copy_i32_from_mapped(d_meta_, m_dmeta_, (int64_t) kDsMeta * MT, cs);
    // pre of the sub-layer before (pp), of this layer's attention (pa) and FFN (pf)
    float *pp = d_pre_[0], *pa = d_pre_[1], *pf = d_pre_[2];
    // the hand-off's sections of this window's T rows (d_hoff_)
    auto hin = [&](int k) { return m_in_ + (size_t) d_hoff_[k] * (size_t) T; };
    auto hout = [&](int k) { return m_hand_ + (size_t) d_hoff_[k] * (size_t) T; };
    if (first()) {   // the four streams from the embeddings; layer 0's attention reads stream 0
        k::copy_from_mapped(x_, m_emb_, (int64_t) T * N, cs);
        k::glm5_hc_init(x_, g_hs_, N, T, cs);
        k::copy_from_mapped(pp, m_done_, (int64_t) T * kHC, cs);
    } else {         // as the stage before left them
        k::copy_from_mapped(g_hs_, m_in_, (int64_t) T * kHC * N, cs);
        k::copy_from_mapped(pp, hin(kDsHPre), (int64_t) T * kHC, cs);
        if (d_in_.kv >= 0) {   // the rows the window closes in the carried KV source: into this stage's copy of its cache
            const int64_t D = s.key_length, ID = s.idx_key_length, kv = d_in_.kv;
            k::copy_from_mapped(d_lat_, hin(kDsHLat), (int64_t) T * D, cs);
            k::copy_from_mapped(d_ik_, hin(kDsHIk), (int64_t) T * ID, cs);
            k::ds41_comp_store(d_lat_, d_ik_, d_meta_, (int) dblocks_[(size_t) kv].ratio, D, ID, T, d_comp_[(size_t) kv],
                               d_rows_[(size_t) kv], cs);
            if (d_out_.kv == kv) {   // and on to the next stage
                k::copy_from_mapped(hout(kDsHLat), d_lat_, (int64_t) T * D, cs);
                k::copy_from_mapped(hout(kDsHIk), d_ik_, (int64_t) T * ID, cs);
            }
        }
        if (d_in_.ids) {
            k::copy_i32_from_mapped(d_ids_, (const int32_t*) hin(kDsHIds), (int64_t) T * ds41_hw(kDsHIds), cs);
            k::copy_i32_from_mapped(d_cnt_, (const int32_t*) hin(kDsHCnt), T, cs);
        }
        if (d_in_.cand) {
            k::copy_i32_from_mapped(d_cand_, (const int32_t*) hin(kDsHCand), (int64_t) T * ds41_hw(kDsHCand), cs);
            k::copy_i32_from_mapped(d_ncand_, (const int32_t*) hin(kDsHNcand), T, cs);
        }
    }
    // the mixes of the streams as they are: post and comb for the write, pre for the read one sub-layer on
    auto hc_mixes = [&](const float* fn, const float* scale, const float* base, float* pre) {
        f32_rows(g_hs_, kHC * N, fn, g_mixes_, kMix, T, cs);
        k::glm5_hc_pre(g_hs_, g_mixes_, scale, base, eps, hc_eps, iters, g_y_, g_post_, g_comb_, N, T, cs);
        k::ds41_hc_pre_weights(g_hs_, g_mixes_, scale, base, eps, hc_eps, pre, N, T, cs);
    };
    uint32_t ring = 0;
    size_t eng_at = 0;
    for (int64_t l = lb_; l < le_; ++l) {
        const DBlock& b = dblocks_[(size_t) l];
        if (b.eng >= 0) {   // Engram: the staged table rows through engram_kv, gated into the streams
            const int64_t RW = s.engram_n_heads * s.engram_head_dim;
            k::copy_from_mapped(d_engr_, m_deng_ + eng_at * (size_t) MT * (size_t) RW, (int64_t) T * RW, cs);
            gproj(b.eng_kv, d_engr_, d_engkv_, T, false, cs);
            k::ds41_engram_gate(g_hs_, d_engkv_, b.eng_gate, nullptr, eps, N, T, cs);
            ++eng_at;
        }
        // ---- the attention
        hc_mixes(b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base, pa);
        k::ds41_hc_collapse(g_hs_, pp, g_y_, N, T, cs);
        k::native_qsa_rms_norm_weighted_q8_1(g_y_, b.attn_norm, hn_, g_xq_, (int) N, T, eps, cs);
        dprof(0, cs);
        record_dattn(b, l, T, cs);
        if (d_out_.kv == l) {   // this block's closed rows cross the cut after the stage (d_lat_, d_ik_ as stored)
            k::copy_from_mapped(hout(kDsHLat), d_lat_, (int64_t) T * s.key_length, cs);
            k::copy_from_mapped(hout(kDsHIk), d_ik_, (int64_t) T * s.idx_key_length, cs);
        }
        k::glm5_hc_post(mix_, g_hs_, g_post_, g_comb_, g_hs_, N, T, cs);
        // ---- the FFN
        hc_mixes(b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base, pf);
        k::ds41_hc_collapse(g_hs_, pa, g_y_, N, T, cs);
        k::native_qsa_rms_norm_weighted_q8_1(g_y_, b.ffn_norm, hn_, g_xq_, (int) N, T, eps, cs);
        dprof(0, cs);
        if (!record_dmoe(b, s.moe_index(l), T, ring, segs_[(size_t) T], cs, err)) return false;
        k::glm5_hc_post(moe_out_, g_hs_, g_post_, g_comb_, g_hs_, N, T, cs);
        dprof(7, cs);
        std::swap(pp, pf);
    }
    auto report = [&] {
        if (!dprof_on_) return;
        dprof(7, cs);
        if (++dprof_n_ % 64 != 0) return;
        std::fprintf(stderr, "strata plain: decode profile, blocks %lld..%lld, ms a window of %d (64 windows): mHC and "
                             "norms %.2f, attention projections %.2f, state and indexer %.2f, attention %.2f, its "
                             "output %.2f, router and shared expert %.2f, routed experts %.2f, combine and head %.2f\n",
                     (long long) lb_, (long long) le_ - 1, T, dprof_ms_[0] / 64, dprof_ms_[1] / 64, dprof_ms_[2] / 64,
                     dprof_ms_[3] / 64, dprof_ms_[4] / 64, dprof_ms_[5] / 64, dprof_ms_[6] / 64, dprof_ms_[7] / 64);
        for (double& v : dprof_ms_) v = 0;
    };
    if (!last()) {
        k::copy_from_mapped(m_hand_, g_hs_, (int64_t) T * kHC * N, cs);
        k::copy_from_mapped(hout(kDsHPre), pp, (int64_t) T * kHC, cs);
        if (d_out_.ids) {   // the last selection before the cut: made here or carried in
            k::copy_i32_from_mapped((int32_t*) hout(kDsHIds), d_ids_, (int64_t) T * ds41_hw(kDsHIds), cs);
            k::copy_i32_from_mapped((int32_t*) hout(kDsHCnt), d_cnt_, T, cs);
        }
        if (d_out_.cand) {
            k::copy_i32_from_mapped((int32_t*) hout(kDsHCand), d_cand_, (int64_t) T * ds41_hw(kDsHCand), cs);
            k::copy_i32_from_mapped((int32_t*) hout(kDsHNcand), d_ncand_, T, cs);
        }
        report();
        return true;
    }
    k::ds41_hc_collapse(g_hs_, pp, x_, N, T, cs);
    k::native_qsa_rms_norm_weighted(x_, out_norm_, hn_, (int) N, T, eps, cs);
    head_logits(hn_, T, cs);   // the tokens are picked after the graph (forward), with the run's sampler
    report();
    return true;
}

}  // namespace strata::core
