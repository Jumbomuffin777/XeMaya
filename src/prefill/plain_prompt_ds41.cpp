// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/prefill/plain_prompt_ds41.cpp - DeepSeek-V4.1-Flash's prompt in chunks of batched products (PlainModel::prompt).
//
// The block of src/core/plain_ds41.cpp (after llama.cpp's deepseek41 graph at master 8642541b2; MIT License,
// Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE), T tokens at a time: every quantized
// projection a matrix product of the chunk's rows (the int8 products where the weight's type and shape are theirs,
// else the weight dequantized to FP16), the F32 ones (the mHC mixes, the compressor, the indexer, the router) F32
// products, the state kernels, the indexer, the attention and the router the window's kernels over the chunk's rows
// (they take any number of rows), the routed experts grouped by expert (prompt_routed).  The state is left as the
// windows leave it: every layer's ring, the open blocks, the compressed caches, the sequence's tokens.
#include "strata/core/plain_model.hpp"

#include "../core/plain_ds41_rope.hpp"
#include "strata/core/gpu.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/ds41.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/prefill/prefill.hpp"
#include "plain_prompt.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::core {

namespace k = strata::kernels;

namespace {
constexpr int kHC = 4, kMix = (2 + kHC) * kHC;
int file_type(int t) { return t == k::kNativeQ6KRows ? 14 : t == k::kNativeQ8Rows ? 8 : t; }
}  // namespace

bool PlainModel::prompt_chunks_ds41(const int32_t* tokens, int64_t n, int64_t pos, int32_t* out, float* logits,
                                    std::string& err, int64_t c_begin, int64_t c_end, const float* hand_in) {
    PromptBufs& P = pb_;
    const ModelSpec& s = spec_;
    const int64_t N = s.n_embd, H = s.n_head, D = s.key_length, QL = s.q_lora_rank, G = s.o_group_count,
                  OL = s.o_lora_rank, ID = s.idx_key_length, IH = s.idx_n_head, E = s.n_expert, K = s.n_expert_used,
                  GI = H * D / G, RW = s.engram_n_heads * s.engram_head_dim;
    const float eps = (float) s.rms_eps, hc_eps = (float) s.hc_eps;
    const int iters = (int) s.hc_sinkhorn_iters;
    const int64_t bs = std::max<int64_t>(1, s.cand_block_size), cand_k = std::max<int64_t>(1, s.cand_topk_blocks);
    strata::gpu::Stream cs = cs_;
    PromptGemm& pg = prompt_gemm();
    prefill::Gemm& gm = pg.gemm;
    static const bool dense_env = [] {   // STRATA_PLAIN_DENSE_MMQ=0: every dense product from FP16
        const char* v = std::getenv("STRATA_PLAIN_DENSE_MMQ");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool dense_on = dense_env && prefill::mmq::chosen();
    // x [T][n_in] (contiguous rows) through a quantized weight into y [T][n_out]: int8 where the weight's type and
    // shape are the int8 products', else from an FP16 copy of x
    auto prod = [&](int type, const void* w, int64_t n_in, int64_t n_out, const float* x, float* y, int64_t T) {
        if (dense_on && prefill::mmq::supported(type) && n_out % 128 == 0 && n_in % 32 == 0) {
            prefill::mmq::quantize(x, nullptr, P.xq, type, n_in, n_in, T, cs);
            prefill::mmq::Product p;
            p.w = w; p.type = type; p.w_rows = n_out; p.w_cols = n_in; p.n = 1; p.xq = P.xq;
            p.bounds = P.dense_bounds; p.ids = P.identity; p.total_rows = T; p.max_rows = T;
            p.dst = y; p.ld_dst = n_out;
            pg.mmq.run(p, cs);
            return;
        }
        prefill::to_f16(x, P.x16, T * n_in, cs);
        gm.native(P.x16, type, w, y, T, n_out, n_in, n_out);
    };
    auto dense = [&](const W& w, const float* x, float* y, int64_t T) { prod(w.type, w.data, w.n_in, w.n_out, x, y, T); };
    static const bool cpu_env = [] {
        const char* v = std::getenv("STRATA_PLAIN_CPU_SHARE");
        return v == nullptr || std::strtol(v, nullptr, 10) != 0;
    }();
    const bool cpu_share = cpu_env && strata::prefill::cpu_share_setting() != 0.0 && split_.c > 0.0 && split_.p > 0.0 &&
                           P.cpu_cap > 0;
    static const bool profile = std::getenv("STRATA_PROMPT_PROFILE") != nullptr;
    // 0 the mHC, 1 the attention's projections, 2 the state and the indexer, 3 the attention, 4 its output, 5 the
    // grouping, 6 the experts, 7 the shared expert and the router, 8 Engram, 9 the chunk's staging, 10 the head, 11 the rest
    double stage_ms[12] = {};
    auto t_mark = std::chrono::steady_clock::now();
    auto mark = [&](int stage) {
        if (!profile) return;
        strata::gpu::stream_sync(cs);
        if (pg.xs != nullptr) strata::gpu::stream_sync(pg.xs);
        const auto now = std::chrono::steady_clock::now();
        stage_ms[stage] += std::chrono::duration<double, std::milli>(now - t_mark).count();
        t_mark = now;
    };
    PromptRouted rst;
    if (!pg.identity) {
        prefill::mmq::iota(P.identity, std::max(P.T * K, P.T), cs);
        pg.identity = true;
    }
    try {
        const int64_t L = n_moe_;
        std::vector<int32_t> routes(disp_.usage.empty() ? 0 : (size_t) (L * P.T * K));
        const int64_t c_last = std::min(n, c_end);
        // the hand-off's section k (d_hoff_) from the row of token c0 on, in this stage's pinned rows
        auto hsec = [&](int sec, int64_t c0) {
            return hand_host_ + d_hoff_[sec] * hand_cap_ + (c0 - c_begin) * ds41_hw(sec);
        };
        const int64_t top_k = ds41_hw(kDsHIds), cand_w = ds41_hw(kDsHCand);
        if (!first()) {   // the rows handed on, section by section, into this stage's own pinned memory
            int64_t prev_cap = hand_cap_;
            const std::vector<PlainModel*> ch = root_->chain();
            for (size_t i = 1; i < ch.size(); ++i)
                if (ch[i] == this) prev_cap = ch[i - 1]->hand_cap_;
            const bool take[kDsHSections] = {true, true, d_in_.kv >= 0, d_in_.kv >= 0, d_in_.ids, d_in_.ids, d_in_.cand,
                                             d_in_.cand};
            for (int sec = 0; sec < kDsHSections; ++sec)
                if (take[sec])
                    std::memcpy(hand_host_ + d_hoff_[sec] * hand_cap_, hand_in + d_hoff_[sec] * prev_cap,
                                (size_t) ((c_last - c_begin) * ds41_hw(sec)) * 4);
        }
        for (int64_t c0 = c_begin; c0 < c_last; c0 += P.T) {
            const int64_t T = std::min<int64_t>(P.T, c_last - c0), p0 = pos + c0;
            float* hand = hand_host_ != nullptr ? hsec(kDsHStreams, c0) : nullptr;
            float* hand_pre = hand_host_ != nullptr ? hsec(kDsHPre, c0) : nullptr;
            // the chunk's staging on the host: the tokens, the embeddings, the positions' records, Engram's rows
            for (int64_t t = 0; t < T; ++t) {
                const int32_t tok = tokens[c0 + t];
                if (tok < 0 || tok >= n_vocab_ ||
                    (first() && !k::cpu::native_row_to_float(emb_type_, emb_ + (size_t) tok * emb_row_, P.emb + t * N, N))) {
                    err = "plain prompt: a token is out of the vocabulary";
                    return false;
                }
                h_dtok_[p0 + t] = tok;
                if (first()) {
                    float* one = P.h_one + t * kHC;
                    one[0] = 1.0f, one[1] = one[2] = one[3] = 0.0f;
                }
            }
            ds41_meta(p0, T, P.T, P.h_meta);
            if (!d_eng_local_.empty()) {   // the table rows of every token and module, read on a few threads
                std::atomic<int64_t> next{0};
                std::atomic<bool> bad{false};
                const int64_t jobs = (int64_t) d_eng_local_.size() * T;
                auto work = [&] {
                    for (int64_t i = next.fetch_add(16); i < jobs && !bad; i = next.fetch_add(16))
                        for (int64_t q = i; q < std::min(jobs, i + 16); ++q) {
                            const int64_t j = q / T, t = q % T;
                            if (!ds41_engram_rows(dblocks_[(size_t) d_eng_local_[(size_t) j]].eng, p0 + t,
                                                  P.h_eng + (j * P.T + t) * RW))
                                bad = true;
                        }
                };
                const int nt = (int) std::max<int64_t>(1, std::min<int64_t>({16, (int64_t) std::thread::hardware_concurrency(), jobs / 64 + 1}));
                std::vector<std::thread> th;
                for (int i = 1; i < nt; ++i) th.emplace_back(work);
                work();
                for (auto& x : th) x.join();
                if (bad) { err = "plain prompt: an Engram table row could not be read"; return false; }
            }
            P.h_dense_bounds[0] = 0;
            P.h_dense_bounds[1] = (int32_t) T;
            // pre of the sub-layer before (pp), of this layer's attention (pa) and FFN (pf)
            float *pp = P.d_pre[0], *pa = P.d_pre[1], *pf = P.d_pre[2];
            if (!(first() ? strata::gpu::copy_async(P.x, P.emb, (size_t) (T * N) * 4, cs) &&
                                strata::gpu::copy_async(pp, P.h_one, (size_t) (T * kHC) * 4, cs)
                          : strata::gpu::copy_async(P.g_hs, hand, (size_t) (T * kHC * N) * 4, cs) &&
                                strata::gpu::copy_async(pp, hand_pre, (size_t) (T * kHC) * 4, cs)) ||
                !strata::gpu::copy_async(P.d_meta, P.h_meta, (size_t) (kDsMeta * P.T) * 4, cs) ||
                !strata::gpu::copy_async(P.dense_bounds, P.h_dense_bounds, 2 * sizeof(int32_t), cs)) {
                err = std::string("plain prompt: staging: ") + strata::gpu::last_error();
                return false;
            }
            if (first()) k::glm5_hc_init(P.x, P.g_hs, N, T, cs);
            const int32_t* mpos = P.d_meta;
            if (!first()) {   // the cut's carry (DsCarry): the closed rows into this stage's copy of the cache (they
                              // stay in the pinned rows for a next stage that reads them too), the selection, the blocks
                bool ok = true;
                if (d_in_.kv >= 0) {
                    const int64_t kv = d_in_.kv;
                    ok = strata::gpu::copy_async(P.d_lat, hsec(kDsHLat, c0), (size_t) (T * D) * 4, cs) &&
                         strata::gpu::copy_async(P.g_ik, hsec(kDsHIk, c0), (size_t) (T * ID) * 4, cs);
                    if (ok)
                        k::ds41_comp_store(P.d_lat, P.g_ik, mpos, (int) dblocks_[(size_t) kv].ratio, D, ID, T,
                                           d_comp_[(size_t) kv], d_rows_[(size_t) kv], cs);
                }
                if (d_in_.ids)
                    ok = ok && strata::gpu::copy_async(P.d_ids, hsec(kDsHIds, c0), (size_t) (T * top_k) * 4, cs) &&
                         strata::gpu::copy_async(P.d_cnt, hsec(kDsHCnt, c0), (size_t) T * 4, cs);
                if (d_in_.cand)
                    ok = ok && strata::gpu::copy_async(P.d_cand, hsec(kDsHCand, c0), (size_t) (T * cand_w) * 4, cs) &&
                         strata::gpu::copy_async(P.d_ncand, hsec(kDsHNcand, c0), (size_t) T * 4, cs);
                if (!ok) {
                    err = std::string("plain prompt: the carried rows: ") + strata::gpu::last_error();
                    return false;
                }
            }
            mark(9);
            auto hc_mixes = [&](const float* fn, const float* scale, const float* base, float* pre) {
                k::f32_gemm_rows(P.g_hs, kHC * N, fn, P.g_mixes, kMix, kHC * N, kMix, T, cs);
                k::glm5_hc_pre(P.g_hs, P.g_mixes, scale, base, eps, hc_eps, iters, P.g_y, P.g_post, P.g_comb, N, T, cs);
                k::ds41_hc_pre_weights(P.g_hs, P.g_mixes, scale, base, eps, hc_eps, pre, N, T, cs);
            };
            size_t eng_at = 0;
            for (int64_t l = lb_; l < le_; ++l) {
                const DBlock& b = dblocks_[(size_t) l];
                if (b.eng >= 0) {   // Engram: the staged table rows through engram_kv, gated into the streams
                    if (!strata::gpu::copy_async(P.d_engr, P.h_eng + (int64_t) eng_at * P.T * RW, (size_t) (T * RW) * 4, cs)) {
                        err = std::string("plain prompt: Engram's rows: ") + strata::gpu::last_error();
                        return false;
                    }
                    dense(b.eng_kv, P.d_engr, P.d_engkv, T);
                    k::ds41_engram_gate(P.g_hs, P.d_engkv, b.eng_gate, nullptr, eps, N, T, cs);
                    ++eng_at;
                    mark(8);
                }
                // ---- the attention
                hc_mixes(b.hc_attn_fn, b.hc_attn_scale, b.hc_attn_base, pa);
                k::ds41_hc_collapse(P.g_hs, pp, P.g_y, N, T, cs);
                k::native_qsa_rms_norm_weighted(P.g_y, b.attn_norm, P.hn, (int) N, (int) T, eps, cs);
                mark(0);
                const k::Ds41Rope r = ds41_layer_rope(s, l, D), ri = ds41_layer_rope(s, l, ID);
                const int32_t *nvis = nullptr, *bpos = nullptr;
                if (b.ratio > 0) {
                    const int64_t i = std::find(d_ratios_.begin(), d_ratios_.end(), b.ratio) - d_ratios_.begin();
                    nvis = P.d_meta + (1 + 2 * i) * P.T;
                    bpos = P.d_meta + (2 + 2 * i) * P.T;
                }
                const bool kv_here = b.ratio > 0 && b.kv_src == l, idx_here = b.ratio > 0 && b.topk_src == l;
                dense(b.q_a, P.hn, P.g_qa, T);
                dense(b.kv, P.hn, P.d_kvr, T);
                if (kv_here) {
                    k::f32_gemm_rows(P.hn, N, b.comp_kv, P.d_cv, D, N, D, T, cs);
                    if (b.ratio > 1) k::f32_gemm_rows(P.hn, N, b.comp_gate, P.d_cg, D, N, D, T, cs);
                }
                if (idx_here) {
                    k::f32_gemm_rows(P.hn, N, b.idx_proj, P.g_iw, IH, N, IH, T, cs);
                    k::scale_inplace(P.g_iw, T * IH, 1.0f / std::sqrt((float) (ID * IH)), cs);
                }
                k::native_qsa_rms_norm_weighted(P.d_kvr, b.kv_norm, P.d_kv, (int) D, (int) T, eps, cs);
                k::native_qsa_rms_norm_weighted(P.g_qa, b.q_a_norm, P.g_qr, (int) QL, (int) T, eps, cs);
                dense(b.q_b, P.g_qr, P.qcur, T);
                if (idx_here) k::f32_gemm_rows(P.g_qr, QL, b.idx_q_b, P.g_iq, IH * ID, QL, IH * ID, T, cs);
                mark(1);
                k::ds41_rope(P.qcur, mpos, r, false, H, D, T, cs);
                k::ds41_rope(P.d_kv, mpos, r, false, 1, D, T, cs);
                k::ds41_ring_store(P.d_kv, mpos, d_ring_, D, T, d_raw_[(size_t) l], cs);
                if (kv_here) {
                    k::ds41_compress_stream(P.d_cv, b.ratio > 1 ? P.d_cg : nullptr, d_cstate_[(size_t) l], mpos, b.comp_norm,
                                            eps, (int) b.ratio, D, T, P.d_lat, cs);
                    k::f32_gemm_rows(P.d_lat, D, b.idx_k, P.g_ig, ID, D, ID, T, cs);
                    k::native_qsa_rms_norm_weighted(P.g_ig, b.idx_k_norm, P.g_ik, (int) ID, (int) T, eps, cs);
                    k::ds41_rope(P.d_lat, bpos, r, false, 1, D, T, cs);
                    k::ds41_rope(P.g_ik, bpos, ri, false, 1, ID, T, cs);
                    k::ds41_comp_store(P.d_lat, P.g_ik, mpos, (int) b.ratio, D, ID, T, d_comp_[(size_t) l],
                                       d_rows_[(size_t) l], cs);
                    if (d_out_.kv == l &&   // the closed rows cross the cut after this stage
                        (!strata::gpu::copy_async(hsec(kDsHLat, c0), P.d_lat, (size_t) (T * D) * 4, cs) ||
                         !strata::gpu::copy_async(hsec(kDsHIk, c0), P.g_ik, (size_t) (T * ID) * 4, cs))) {
                        err = std::string("plain prompt: the compressed rows to the next GPU: ") + strata::gpu::last_error();
                        return false;
                    }
                }
                const float* comp = b.ratio > 0 ? d_comp_[(size_t) b.kv_src] : nullptr;
                if (idx_here) {
                    k::ds41_rope(P.g_iq, mpos, ri, false, IH, ID, T, cs);
                    k::ds41_idx_scores(P.g_iq, P.g_iw, comp, D + ID, D, nvis, (int) IH, (int) ID, T, d_score_stride_,
                                       P.d_scores, cs);
                    if (s.cand_src == l) {
                        k::ds41_block_max(P.d_scores, nvis, (int) bs, T, d_score_stride_, d_blk_stride_, P.d_bmax, cs);
                        k::ds41_topk(P.d_bmax, P.d_meta + (1 + 2 * kDsRatios) * P.T, (int) cand_k, T, d_blk_stride_,
                                     P.d_cand, P.d_ncand, cs);
                    } else if (s.cand_src >= 0 && s.cand_src < l && s.compress_ratios[(size_t) s.cand_src] == b.ratio) {
                        k::ds41_keep_blocks(P.d_scores, nvis, P.d_cand, P.d_ncand, cand_k, (int) bs, T, d_score_stride_, cs);
                    }
                    k::ds41_topk(P.d_scores, nvis, (int) s.idx_top_k, T, d_score_stride_, P.d_ids, P.d_cnt, cs);
                }
                mark(2);
                if (k::ds41_attn_mode() < 1 ||
                    !k::ds41_sink_attn_xmx(P.qcur, d_raw_[(size_t) l], d_ring_, mpos, (int) s.sliding_window, comp, D + ID,
                                           b.ratio > 0 ? P.d_ids : nullptr, b.ratio > 0 ? P.d_cnt : nullptr, s.idx_top_k,
                                           b.sinks, 1.0f / std::sqrt((float) D), (int) H, (int) D, T, P.attn, cs))
                    k::ds41_sink_attn(P.qcur, d_raw_[(size_t) l], d_ring_, mpos, (int) s.sliding_window, comp, D + ID,
                                      b.ratio > 0 ? P.d_ids : nullptr, b.ratio > 0 ? P.d_cnt : nullptr, s.idx_top_k,
                                      b.sinks, 1.0f / std::sqrt((float) D), (int) H, (int) D, T, P.attn, cs);
                k::ds41_rope(P.attn, mpos, r, true, H, D, T, cs);
                mark(3);
                // the grouped low-rank output: each group's rows of every token through its o_a, then o_b
                const size_t ab = k::native_mmvq_weight_bytes(file_type(b.o_a.type), (int) GI, (int) OL);
                k::glm5_transpose_abd(P.attn, P.attn32, T, G, GI, cs);
                for (int64_t g = 0; g < G; ++g)
                    prod(b.o_a.type, (const uint8_t*) b.o_a.data + (size_t) g * ab, GI, OL, P.attn32 + g * T * GI,
                         P.d_oaT + g * T * OL, T);
                k::glm5_transpose_abd(P.d_oaT, P.d_oa, G, T, OL, cs);
                dense(b.o_b, P.d_oa, P.mix, T);
                mark(4);
                k::glm5_hc_post(P.mix, P.g_hs, P.g_post, P.g_comb, P.g_hs, N, T, cs);

                // ---- the FFN
                hc_mixes(b.hc_ffn_fn, b.hc_ffn_scale, b.hc_ffn_base, pf);
                k::ds41_hc_collapse(P.g_hs, pa, P.g_y, N, T, cs);
                k::native_qsa_rms_norm_weighted(P.g_y, b.ffn_norm, P.hn, (int) N, (int) T, eps, cs);
                mark(0);
                k::f32_gemm_rows(P.hn, N, b.router, P.rlog, E, N, E, T, cs);
                const float scale = s.expert_weights_scale > 0 ? (float) s.expert_weights_scale : 1.0f;
                k::ds41_router(P.rlog, b.router_b, (int) T, (int) E, (int) K, s.expert_weights_norm && K > 1, scale, P.ids,
                               P.w, cs);
                if (!P.int8) prefill::to_f16(P.hn, P.x16, T * N, cs);   // the FP16 path's routed rows
                const int64_t ml = s.moe_index(l);
                if (!prompt_routed(ml, T, cpu_share, routes.empty() ? nullptr : routes.data() + ml * P.T * K,
                                   [&](int stage) { mark(stage == 1 ? 5 : 6); }, rst, err))
                    return false;
                {   // the shared expert: gate / up from hn, the clamp, down
                    const int64_t FW = b.ffn_gate.n_out;
                    dense(b.ffn_gate, P.hn, P.g_ffg, T);
                    dense(b.ffn_up, P.hn, P.g_ffu, T);
                    k::glm5_swiglu_clamp(P.g_ffg, P.g_ffu, P.g_ffg, T * FW, b.clamp, cs);
                    dense(b.ffn_down, P.g_ffg, P.shared, T);
                }
                k::native_moe_combine_multi(P.parts, P.w, P.shared, P.moe_out, N, K, (int) T, cs);
                mark(7);
                k::glm5_hc_post(P.moe_out, P.g_hs, P.g_post, P.g_comb, P.g_hs, N, T, cs);
                std::swap(pp, pf);
            }
            if (!last() && (!strata::gpu::copy_async(hand, P.g_hs, (size_t) (T * kHC * N) * 4, cs) ||
                            !strata::gpu::copy_async(hand_pre, pp, (size_t) (T * kHC) * 4, cs) ||
                            (d_out_.ids &&
                             (!strata::gpu::copy_async(hsec(kDsHIds, c0), P.d_ids, (size_t) (T * top_k) * 4, cs) ||
                              !strata::gpu::copy_async(hsec(kDsHCnt, c0), P.d_cnt, (size_t) T * 4, cs))) ||
                            (d_out_.cand &&
                             (!strata::gpu::copy_async(hsec(kDsHCand, c0), P.d_cand, (size_t) (T * cand_w) * 4, cs) ||
                              !strata::gpu::copy_async(hsec(kDsHNcand, c0), P.d_ncand, (size_t) T * 4, cs))))) {
                err = std::string("plain prompt: the rows to the next GPU: ") + strata::gpu::last_error();
                return false;
            }
            if (last() && c0 + T >= n) {   // the last token's logits, by the decode's head
                k::ds41_hc_collapse(P.g_hs + (T - 1) * kHC * N, pp + (T - 1) * kHC, P.x, N, 1, cs);
                k::native_qsa_rms_norm_weighted(P.x, out_norm_, P.hn, (int) N, 1, eps, cs);
                k::native_quantize_q8_1(P.hn, xq_, head_.n_in, 1, cs);
                k::native_mmvq(head_.type, head_.data, xq_, logits_, head_.n_in, head_.n_out, 1, cs);
                k::SamplerParams sp = sampler_;
                sp.counter = (uint64_t) (p0 + T - 1);
                k::sample_tokens(logits_, 1, (int) n_vocab_, nullptr, 0, sp, m_out_, cs);
            }
            // the host's staging is rewritten by the next chunk: everything that reads it is done first
            if (!strata::gpu::stream_sync(cs)) {
                err = std::string("plain prompt: ") + strata::gpu::last_error();
                return false;
            }
            mark(10);
            if (!routes.empty()) {   // the prompt's routings, as plain_prompt.cpp counts them
                for (int64_t ml : moe_local_)
                    for (int64_t i = 0; i < T * K; ++i) ++prompt_routed_[(size_t) (ml * E + routes[(size_t) (ml * P.T * K + i)])];
            }
        }
    } catch (const std::exception& e) {
        err = std::string("plain prompt: ") + e.what();
        return false;
    }
    mark(11);
    if (!strata::gpu::stream_sync(cs)) {
        err = std::string("plain prompt: ") + strata::gpu::last_error();
        return false;
    }
    if (c_end < n) return true;   // a layer split's chunk: the stage's part of the prompt goes on with the next
    windows_ += (n + opt_.max_t - 1) / opt_.max_t;
    if (profile)
        std::fprintf(stderr, "strata plain: DeepSeek prompt of %lld tokens (blocks %lld..%lld): mHC %.1f ms, attention "
                             "projections %.1f, state and indexer %.1f, attention %.1f, its output %.1f, grouping %.1f, "
                             "experts %.1f, shared expert and router %.1f, Engram %.1f, staging %.1f, head and sync %.1f, "
                             "the rest %.1f; experts not in VRAM %lld, on the CPU %lld (%lld rows)\n",
                     (long long) n, (long long) lb_, (long long) le_ - 1, stage_ms[0], stage_ms[1], stage_ms[2],
                     stage_ms[3], stage_ms[4], stage_ms[5], stage_ms[6], stage_ms[7], stage_ms[8], stage_ms[9],
                     stage_ms[10], stage_ms[11], (long long) rst.miss_n, (long long) rst.cpu_n, (long long) rst.cpu_rows);
    if (!last()) return true;
    *out = ((volatile int32_t*) h_out_)[0];
    if (logits != nullptr && !strata::gpu::copy(logits, logits_, (size_t) n_vocab_ * 4)) {
        err = "plain prompt: reading the logits failed";
        return false;
    }
    return true;
}

}  // namespace strata::core
