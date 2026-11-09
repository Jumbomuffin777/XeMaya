// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/ds41.cpp - see include/strata/kernels/ds41.hpp.
//
// Transcribed from llama.cpp master 8642541b2 (src/models/deepseek4.cpp, deepseek41.cpp, src/llama-graph.cpp's
// build_moe_ffn, ggml/src/ggml-cpu/ops.cpp's rope_yarn, ggml-impl.h's softplus); MIT License, Copyright (c) 2023-2026
// The ggml authors, see third_party/main/ggml/LICENSE.  Written for correctness first: each kernel is checked by
// ds41_parity against a host reference; the decode path's speed work comes after the block runs end to end.
#include "strata/kernels/ds41.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_ds41
#include "strata/kernels/device_arch_tag.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {
namespace {

constexpr int HC = 4;
constexpr int MIX = (2 + HC) * HC;
constexpr int WARP = 32;

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : sycl::log(1.0f + sycl::exp(x)); }

// order-preserving key of a float: a larger float has a larger key
inline uint32_t fkey(float v) {
    const uint32_t u = sycl::bit_cast<uint32_t>(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

}  // namespace

void ds41_router(const float* logits, const float* bias, int n_tokens, int n_expert, int k, bool norm, float scale,
                 int32_t* ids, float* weights, void* stream) {
    if (n_tokens <= 0) return;
    if (n_expert <= 0 || k <= 0 || k > WARP || k > n_expert)
        throw core::DeviceError("ds41_router: " + std::to_string(k) + " of " + std::to_string(n_expert) +
                                " experts (at most 32 picks)");
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * WARP, WARP), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int t = (int) it.get_group(0), lane = (int) sg.get_local_linear_id();
            const float* l = logits + (size_t) t * n_expert;
            float pv = INFINITY, mine = 0.0f;
            int pi = -1;
            for (int i = 0; i < k; ++i) {
                float bv = -INFINITY;
                int bi = n_expert;
                for (int e = lane; e < n_expert; e += WARP) {
                    const float v = sycl::sqrt(softplus(l[e])) + bias[e];
                    if ((v < pv || (v == pv && e > pi)) && (v > bv || (v == bv && e < bi))) { bv = v; bi = e; }
                }
                for (int off = WARP / 2; off > 0; off >>= 1) {
                    const float ov = sycl::permute_group_by_xor(sg, bv, off);
                    const int oi = sycl::permute_group_by_xor(sg, bi, off);
                    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                }
                if (lane == i) {
                    mine = sycl::sqrt(softplus(l[bi]));
                    ids[(size_t) t * k + i] = bi;
                }
                pv = bv;
                pi = bi;
            }
            float div = 1.0f;
            if (norm) div = sycl::reduce_over_group(sg, lane < k ? mine : 0.0f, sycl::plus<float>()) + 1e-20f;
            if (lane < k) weights[(size_t) t * k + lane] = mine / div * scale;
        });
    sync_if_needed(stream, event, "ds41_router");
}

void ds41_hc_pre_weights(const float* x, const float* mixes, const float* scale, const float* base, float rms_eps,
                         float hc_eps, float* pre, int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (n_embd <= 0) throw core::DeviceError("ds41_hc_pre_weights: n_embd " + std::to_string(n_embd));
    constexpr int WG = 256;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t t = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const float* xt = x + t * HC * N;
            float ss = 0.0f;
            for (int64_t i = lid; i < HC * N; i += WG) ss += xt[i] * xt[i];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            if (lid < HC) {
                const float inv = 1.0f / sycl::sqrt(ss / (float) (HC * N) + rms_eps);
                pre[t * HC + lid] = sigmoid(mixes[t * MIX + lid] * inv * scale[0] + base[lid]) + hc_eps;
            }
        });
    sync_if_needed(stream, event, "ds41_hc_pre_weights");
}

void ds41_hc_collapse(const float* x, const float* pre, float* y, int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0 || n_embd <= 0) return;
    const int64_t N = n_embd, total = n_tokens * n_embd;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) total), [=](sycl::id<1> i) {
        const int64_t t = (int64_t) i[0] / N, d = (int64_t) i[0] % N;
        const float* xt = x + t * HC * N + d;
        const float* p = pre + t * HC;
        y[i[0]] = p[0] * xt[0] + p[1] * xt[N] + p[2] * xt[2 * N] + p[3] * xt[3 * N];
    });
    sync_if_needed(stream, event, "ds41_hc_collapse");
}

void ds41_compress(const float* v, const float* s, const float* w, float eps, float* out, int64_t n_blocks,
                   int ratio, int64_t dim, void* stream) {
    if (n_blocks <= 0) return;
    if (ratio < 1 || dim <= 0 || (ratio > 1 && !s))
        throw core::DeviceError("ds41_compress: ratio " + std::to_string(ratio) + " (a gate is needed above 1)");
    constexpr int WG = 256;
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> c_l(sycl::range<1>((size_t) dim), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_blocks * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t b = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const float* vb = v + b * ratio * dim;
            const float* sb = s ? s + b * ratio * dim : nullptr;
            float ss = 0.0f;
            for (int64_t d = lid; d < dim; d += WG) {
                float c;
                if (!sb) {
                    c = 0.0f;   // ratio 1 without a gate: softmax of one 0 is 1
                    for (int j = 0; j < ratio; ++j) c += vb[j * dim + d] / (float) ratio;
                    if (ratio == 1) c = vb[d];
                } else {
                    float m = -INFINITY;
                    for (int j = 0; j < ratio; ++j) m = sycl::fmax(m, sb[j * dim + d]);
                    float sum = 0.0f, acc = 0.0f;
                    for (int j = 0; j < ratio; ++j) {
                        const float e = sycl::exp(sb[j * dim + d] - m);
                        sum += e;
                        acc += vb[j * dim + d] * e;
                    }
                    c = acc / sum;
                }
                c_l[d] = c;
                ss += c * c;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = 1.0f / sycl::sqrt(ss / (float) dim + eps);
            for (int64_t d = lid; d < dim; d += WG) out[b * dim + d] = c_l[d] * inv * w[d];
        });
    });
    sync_if_needed(stream, event, "ds41_compress");
}

void ds41_ring_store(const float* kv, const int32_t* pos, int64_t ring, int64_t dim, int64_t n_tokens, float* raw,
                     void* stream) {
    if (n_tokens <= 0) return;
    if (ring < n_tokens || dim <= 0)
        throw core::DeviceError("ds41_ring_store: a ring of " + std::to_string(ring) + " rows for " +
                                std::to_string(n_tokens) + " tokens");
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * dim)), [=](sycl::id<1> i) {
        const int64_t t = (int64_t) i[0] / dim, d = (int64_t) i[0] % dim;
        raw[((int64_t) pos[t] % ring) * dim + d] = kv[t * dim + d];
    });
    sync_if_needed(stream, event, "ds41_ring_store");
}

void ds41_block_pos(const int32_t* pos, int ratio, int32_t* block_pos, int32_t* n_visible, int64_t n_tokens,
                    void* stream) {
    if (n_tokens <= 0) return;
    if (ratio < 1) throw core::DeviceError("ds41_block_pos: ratio " + std::to_string(ratio));
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) n_tokens), [=](sycl::id<1> i) {
        const int32_t p = pos[i[0]];
        block_pos[i[0]] = p - ratio + 1 > 0 ? p - ratio + 1 : 0;
        n_visible[i[0]] = (p + 1) / ratio;
    });
    sync_if_needed(stream, event, "ds41_block_pos");
}

void ds41_compress_stream(const float* v, const float* s, float* state, const int32_t* pos, const float* w, float eps,
                          int ratio, int64_t dim, int64_t n_tokens, float* out, void* stream) {
    if (n_tokens <= 0) return;
    if (ratio < 1 || ratio > 8 || dim <= 0 || (ratio > 1 && (!s || !state)))
        throw core::DeviceError("ds41_compress_stream: ratio " + std::to_string(ratio) +
                                " (1 to 8; a gate and a state above 1)");
    constexpr int WG = 256;
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> c_l(sycl::range<1>((size_t) dim), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t t = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const bool done = (pos[t] + 1) % ratio == 0;
            float ss = 0.0f;
            for (int64_t d = lid; d < dim; d += WG) {
                float c = 0.0f;
                if (done && ratio == 1) {
                    c = v[t * dim + d];
                } else if (done) {
                    // the block's row j: `back` rows before this token, in the window or in the state
                    float m = -INFINITY;
                    for (int j = 0; j < ratio; ++j) {
                        const int64_t back = ratio - 1 - j;
                        m = sycl::fmax(m, back <= t ? s[(t - back) * dim + d] : state[(j * 2 + 1) * dim + d]);
                    }
                    float sum = 0.0f, acc = 0.0f;
                    for (int j = 0; j < ratio; ++j) {
                        const int64_t back = ratio - 1 - j;
                        const float sv = back <= t ? s[(t - back) * dim + d] : state[(j * 2 + 1) * dim + d];
                        const float vv = back <= t ? v[(t - back) * dim + d] : state[(j * 2) * dim + d];
                        const float e = sycl::exp(sv - m);
                        sum += e;
                        acc += vv * e;
                    }
                    c = acc / sum;
                }
                c_l[d] = c;
                ss += c * c;
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float inv = done ? 1.0f / sycl::sqrt(ss / (float) dim + eps) : 0.0f;
            for (int64_t d = lid; d < dim; d += WG) out[t * dim + d] = c_l[d] * inv * w[d];
        });
    });
    sync_if_needed(stream, event, "ds41_compress_stream");
    if (ratio == 1) return;
    // the block left open after the window: its rows of this window into the state (the earlier ones are there)
    const auto keep = queue_for(stream).parallel_for(sycl::range<1>((size_t) ((ratio - 1) * dim)), [=](sycl::id<1> i) {
        const int64_t j = (int64_t) i[0] / dim, d = (int64_t) i[0] % dim;
        const int64_t open = (pos[n_tokens - 1] + 1) % ratio;   // positions of the open block so far
        if (j >= open) return;
        const int64_t back = open - 1 - j;                       // rows before the window's last
        if (back >= n_tokens) return;
        state[(j * 2) * dim + d] = v[(n_tokens - 1 - back) * dim + d];
        state[(j * 2 + 1) * dim + d] = s[(n_tokens - 1 - back) * dim + d];
    });
    sync_if_needed(stream, keep, "ds41_compress_stream (state)");
}

void ds41_comp_store(const float* lat, const float* ik, const int32_t* pos, int ratio, int64_t dim, int64_t idx_dim,
                     int64_t n_tokens, float* comp, int64_t rows, void* stream) {
    if (n_tokens <= 0) return;
    if (ratio < 1 || dim <= 0 || idx_dim < 0) throw core::DeviceError("ds41_comp_store: ratio " + std::to_string(ratio));
    const int64_t stride = dim + idx_dim;
    const auto event = queue_for(stream).parallel_for(sycl::range<1>((size_t) (n_tokens * stride)), [=](sycl::id<1> i) {
        const int64_t t = (int64_t) i[0] / stride, d = (int64_t) i[0] % stride;
        const int64_t p = pos[t];
        if ((p + 1) % ratio != 0) return;
        const int64_t row = (p + 1) / ratio - 1;
        if (row >= rows) return;
        comp[row * stride + d] = d < dim ? lat[t * dim + d] : ik[t * idx_dim + (d - dim)];
    });
    sync_if_needed(stream, event, "ds41_comp_store");
}

void ds41_engram_gate(float* x, const float* kv, const float* gate_w, const float* keep, float rms_eps,
                      int64_t n_embd, int64_t n_tokens, void* stream) {
    if (n_tokens <= 0) return;
    if (n_embd <= 0) throw core::DeviceError("ds41_engram_gate: n_embd " + std::to_string(n_embd));
    constexpr int WG = 256;
    const int64_t N = n_embd;
    const auto event = queue_for(stream).parallel_for(
        sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t t = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            float* xt = x + t * HC * N;
            const float* key = kv + t * (HC + 1) * N;
            const float* val = key + HC * N;
            float xx[HC], kk[HC], xkg[HC];
            for (int s = 0; s < HC; ++s) xx[s] = kk[s] = xkg[s] = 0.0f;
            for (int64_t d = lid; d < N; d += WG)
                for (int s = 0; s < HC; ++s) {
                    const float a = xt[s * N + d], b = key[s * N + d];
                    xx[s] += a * a;
                    kk[s] += b * b;
                    xkg[s] += a * b * gate_w[s * N + d];
                }
            float g[HC];
            const float kp = keep ? keep[t] : 1.0f;
            for (int s = 0; s < HC; ++s) {
                const float sx = sycl::reduce_over_group(it.get_group(), xx[s], sycl::plus<float>());
                const float sk = sycl::reduce_over_group(it.get_group(), kk[s], sycl::plus<float>());
                const float sg = sycl::reduce_over_group(it.get_group(), xkg[s], sycl::plus<float>());
                const float ix = 1.0f / sycl::sqrt(sx / (float) N + rms_eps);
                const float ik = 1.0f / sycl::sqrt(sk / (float) N + rms_eps);
                const float dot = sg * ix * ik / sycl::sqrt((float) N);
                const float mag = sycl::sqrt(sycl::fmax(sycl::fabs(dot), 1e-6f));
                g[s] = sigmoid(dot < 0.0f ? -mag : mag) * kp;
            }
            sycl::group_barrier(it.get_group());   // every read of x before its write
            for (int64_t d = lid; d < N; d += WG)
                for (int s = 0; s < HC; ++s) xt[s * N + d] += val[d] * g[s];
        });
    sync_if_needed(stream, event, "ds41_engram_gate");
}

void ds41_engram_hash(int32_t token, const int32_t* prev, int64_t pos, const std::vector<int32_t>& token_map,
                      int32_t pad_token_id, int ngram, int heads_per_ngram, const uint64_t* mult,
                      const uint64_t* primes, const uint64_t* offsets, int32_t* out) {
    constexpr int64_t DEAD = -1;
    const int n_prev = ngram - 1;
    const int64_t pad = token_map.at((size_t) pad_token_id);
    const auto cid = [&](int32_t t) -> int64_t {
        return t < 0 || (size_t) t >= token_map.size() ? DEAD : (int64_t) token_map[(size_t) t];
    };
    uint64_t toks[16];
    if (ngram < 2 || ngram > 16) throw core::DeviceError("ds41_engram_hash: n-gram size " + std::to_string(ngram));
    int64_t c = cid(token);
    bool blocked = c == DEAD;
    toks[0] = (uint64_t) (blocked ? pad : c);
    for (int s = 1; s < ngram; ++s) {
        c = pos - s < 0 ? DEAD : cid(prev[n_prev - s]);
        blocked = blocked || c == DEAD;
        toks[s] = (uint64_t) (blocked ? pad : c);
    }
    uint64_t rolling = toks[0] * mult[0];
    for (int o = 1; o < ngram; ++o) {
        rolling ^= toks[o] * mult[o];
        for (int h = 0; h < heads_per_ngram; ++h) {
            const int col = (o - 1) * heads_per_ngram + h;
            out[col] = (int32_t) (rolling % primes[col] + offsets[col]);
        }
    }
}

void ds41_rope(float* x, const int32_t* pos, const Ds41Rope& r, bool inverse, int64_t rows, int64_t dim,
               int64_t n_tokens, void* stream) {
    if (n_tokens <= 0 || rows <= 0) return;
    if (r.n_rot <= 0 || (r.n_rot & 1) || r.offset < 0 || r.offset + r.n_rot > dim)
        throw core::DeviceError("ds41_rope: " + std::to_string(r.n_rot) + " dims at " + std::to_string(r.offset) +
                                " of " + std::to_string(dim));
    // ggml_rope_yarn_corr_dims, on the host in float as ggml
    float corr[2] = {0.0f, 0.0f};
    if (r.ext_factor != 0.0f) {
        const auto corr_dim = [&](float n_rot) {
            return (float) r.n_rot * std::log((float) r.n_ctx_orig / (n_rot * 2.0f * (float) M_PI)) /
                   (2.0f * std::log(r.freq_base));
        };
        corr[0] = std::fmax(0.0f, std::floor(corr_dim(r.beta_fast)));
        corr[1] = std::fmin((float) (r.n_rot - 1), std::ceil(corr_dim(r.beta_slow)));
    }
    const float theta_scale = std::pow(r.freq_base, -2.0f / (float) r.n_rot);
    const float c0 = corr[0], c1 = corr[1], fs = r.freq_scale, ef = r.ext_factor, af = r.attn_factor;
    const int half = r.n_rot / 2, off = r.offset;
    const float sin_sign = inverse ? -1.0f : 1.0f;
    const auto event = queue_for(stream).parallel_for(
        sycl::range<1>((size_t) (n_tokens * rows * half)), [=](sycl::id<1> id) {
            const int64_t i = (int64_t) id[0];
            const int p2 = (int) (i % half);
            const int64_t row = i / half;              // token t's head row
            const int64_t t = row / rows;
            // ggml_rope_cache_init: theta = p theta_scale^i, by repeated products in float as ggml
            float theta_extrap = (float) pos[t];
            for (int k = 0; k < p2; ++k) theta_extrap *= theta_scale;
            const int i0 = 2 * p2;
            const float theta_interp = fs * theta_extrap;
            float theta = theta_interp, mscale = af;
            if (ef != 0.0f) {
                const float y = ((float) (i0 / 2) - c0) / sycl::fmax(0.001f, c1 - c0);
                const float ramp = (1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y))) * ef;
                theta = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
                mscale *= 1.0f + 0.1f * sycl::log(1.0f / fs);
            }
            const float ct = sycl::cos(theta) * mscale, st = sycl::sin(theta) * mscale * sin_sign;
            float* v = x + row * dim + off + i0;
            const float x0 = v[0], x1 = v[1];
            v[0] = x0 * ct - x1 * st;
            v[1] = x0 * st + x1 * ct;
        });
    sync_if_needed(stream, event, "ds41_rope");
}

void ds41_idx_scores(const float* q, const float* w, const float* keys, int64_t key_stride, int64_t key_offset,
                     const int32_t* n_visible, int n_heads, int dim, int64_t n_tokens, int64_t score_stride,
                     float* scores, void* stream) {
    if (n_tokens <= 0 || score_stride <= 0) return;
    if (n_heads <= 0 || dim <= 0) throw core::DeviceError("ds41_idx_scores: heads and width");
    constexpr int WG = 128;
    const int64_t groups = (score_stride + WG - 1) / WG;
    const int64_t QD = (int64_t) n_heads * dim;
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> q_l(sycl::range<1>((size_t) QD), h);
        sycl::local_accessor<float, 1> w_l(sycl::range<1>((size_t) n_heads), h);
        h.parallel_for(sycl::nd_range<1>((size_t) (n_tokens * groups * WG), WG), [=](sycl::nd_item<1> it) {
            const int64_t g = (int64_t) it.get_group(0);
            const int64_t t = g / groups;
            const int lid = (int) it.get_local_id(0);
            for (int64_t i = lid; i < QD; i += WG) q_l[i] = q[t * QD + i];
            for (int i = lid; i < n_heads; i += WG) w_l[i] = w[t * n_heads + i];
            sycl::group_barrier(it.get_group());
            const int64_t c = (g % groups) * WG + lid;
            if (c >= score_stride) return;
            if (c >= n_visible[t]) { scores[t * score_stride + c] = -INFINITY; return; }
            const float* k = keys + c * key_stride + key_offset;
            float acc = 0.0f;
            for (int hh = 0; hh < n_heads; ++hh) {
                float dot = 0.0f;
                for (int d = 0; d < dim; ++d) dot += q_l[hh * dim + d] * k[d];
                acc += sycl::fmax(dot, 0.0f) * w_l[hh];
            }
            scores[t * score_stride + c] = acc;
        });
    });
    sync_if_needed(stream, event, "ds41_idx_scores");
}

void ds41_block_max(const float* scores, const int32_t* n_visible, int bs, int64_t n_tokens, int64_t score_stride,
                    int64_t blk_stride, float* bmax, void* stream) {
    if (n_tokens <= 0 || blk_stride <= 0) return;
    if (bs <= 0) throw core::DeviceError("ds41_block_max: block size " + std::to_string(bs));
    const auto event = queue_for(stream).parallel_for(
        sycl::range<1>((size_t) (n_tokens * blk_stride)), [=](sycl::id<1> id) {
            const int64_t t = (int64_t) id[0] / blk_stride, b = (int64_t) id[0] % blk_stride;
            const int64_t nv = n_visible[t];
            float m = -INFINITY;
            for (int64_t c = b * bs; c < sycl::min<int64_t>(nv, (b + 1) * bs); ++c) m = sycl::fmax(m, scores[t * score_stride + c]);
            if (nv > 0 && b == (nv - 1) / bs) m += 1e30f;
            bmax[t * blk_stride + b] = m;
        });
    sync_if_needed(stream, event, "ds41_block_max");
}

void ds41_keep_blocks(float* scores, const int32_t* n_visible, const int32_t* blocks, const int32_t* n_blocks,
                      int64_t blk_cap, int bs, int64_t n_tokens, int64_t score_stride, void* stream) {
    if (n_tokens <= 0 || score_stride <= 0) return;
    if (bs <= 0) throw core::DeviceError("ds41_keep_blocks: block size " + std::to_string(bs));
    const auto event = queue_for(stream).parallel_for(
        sycl::range<1>((size_t) (n_tokens * score_stride)), [=](sycl::id<1> id) {
            const int64_t t = (int64_t) id[0] / score_stride, c = (int64_t) id[0] % score_stride;
            if (c >= n_visible[t]) return;
            const int32_t b = (int32_t) (c / bs);
            const int32_t* bl = blocks + t * blk_cap;
            int lo = 0, hi = n_blocks[t];          // the ids are ascending (ds41_topk)
            while (lo < hi) {
                const int mid = (lo + hi) / 2;
                if (bl[mid] < b) lo = mid + 1; else hi = mid;
            }
            if (!(lo < n_blocks[t] && bl[lo] == b)) scores[t * score_stride + c] = -INFINITY;
        });
    sync_if_needed(stream, event, "ds41_keep_blocks");
}

void ds41_topk(const float* scores, const int32_t* n, int k, int64_t n_tokens, int64_t score_stride, int32_t* ids,
               int32_t* count, void* stream) {
    if (n_tokens <= 0) return;
    if (k <= 0) throw core::DeviceError("ds41_topk: k " + std::to_string(k));
    constexpr int WG = 256, BINS = 256;
    const uint32_t neg_inf = fkey(-INFINITY);
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> hist(sycl::range<1>(BINS), h);
        sycl::local_accessor<uint32_t, 1> shared(sycl::range<1>(4), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * WG, WG), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int64_t t = (int64_t) it.get_group(0);
            const int lid = (int) it.get_local_id(0);
            const float* sc = scores + t * score_stride;
            const int64_t nt = sycl::min<int64_t>(n[t], score_stride);
            // the rows that can be taken: every one but -inf (and NaN)
            uint32_t valid = 0;
            for (int64_t c = lid; c < nt; c += WG) valid += fkey(sc[c]) > neg_inf && !sycl::isnan(sc[c]);
            valid = sycl::reduce_over_group(grp, valid, sycl::plus<uint32_t>());
            const uint32_t kk = sycl::min<uint32_t>((uint32_t) k, valid);
            if (lid == 0) count[t] = (int32_t) kk;
            if (kk == 0) return;
            // radix select of the kk-th largest key, 8 bits at a time
            uint32_t prefix = 0, mask = 0, remaining = kk;
            for (int shift = 24; shift >= 0; shift -= 8) {
                for (int b = lid; b < BINS; b += WG) hist[b] = 0;
                sycl::group_barrier(it.get_group());
                for (int64_t c = lid; c < nt; c += WG) {
                    const float v = sc[c];
                    const uint32_t key = fkey(v);
                    if (key <= neg_inf || sycl::isnan(v) || (key & mask) != prefix) continue;
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::local_space>(hist[(key >> shift) & 0xFF])
                        .fetch_add(1u);
                }
                sycl::group_barrier(it.get_group());
                if (lid == 0) {
                    uint32_t above = 0;
                    int b = BINS - 1;
                    for (; b > 0; --b) {
                        if (above + hist[b] >= remaining) break;
                        above += hist[b];
                    }
                    shared[0] = (uint32_t) b;
                    shared[1] = above;
                }
                sycl::group_barrier(it.get_group());
                prefix |= shared[0] << shift;
                mask |= 0xFFu << shift;
                remaining -= shared[1];
                sycl::group_barrier(it.get_group());
            }
            // keys above the threshold all go in; `remaining` of the equal ones, lowest index first; ascending order
            uint32_t run_sel = 0, run_eq = 0;
            for (int64_t base = 0; base < nt; base += WG) {
                const int64_t c = base + lid;
                bool gt = false, eq = false;
                if (c < nt) {
                    const float v = sc[c];
                    const uint32_t key = fkey(v);
                    const bool ok = key > neg_inf && !sycl::isnan(v);
                    gt = ok && key > prefix;
                    eq = ok && key == prefix;
                }
                const uint32_t eq_before = run_eq + sycl::exclusive_scan_over_group(grp, (uint32_t) eq, sycl::plus<uint32_t>());
                const bool sel = gt || (eq && eq_before < remaining);
                const uint32_t at = run_sel + sycl::exclusive_scan_over_group(grp, (uint32_t) sel, sycl::plus<uint32_t>());
                if (sel) ids[t * k + at] = (int32_t) c;
                run_sel += sycl::reduce_over_group(grp, (uint32_t) sel, sycl::plus<uint32_t>());
                run_eq += sycl::reduce_over_group(grp, (uint32_t) eq, sycl::plus<uint32_t>());
            }
        });
    });
    sync_if_needed(stream, event, "ds41_topk");
}

void ds41_sink_attn(const float* q, const float* raw, int64_t ring, const int32_t* pos, int window, const float* comp,
                    int64_t comp_stride, const int32_t* ids, const int32_t* count, int64_t id_cap, const float* sinks,
                    float scale, int n_heads, int dim, int64_t n_tokens, float* out, void* stream) {
    if (n_tokens <= 0) return;
    if (n_heads <= 0 || dim <= 0 || window <= 0 || ring < window)
        throw core::DeviceError("ds41_sink_attn: heads, width, a window of " + std::to_string(window) +
                                " in a ring of " + std::to_string(ring));
    constexpr int WG = 256;
    const int64_t max_keys = window + (count ? id_cap : 0);
    const auto event = queue_for(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> q_l(sycl::range<1>((size_t) dim), h);
        sycl::local_accessor<float, 1> p_l(sycl::range<1>((size_t) std::max<int64_t>(max_keys, 1)), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * n_heads * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP)]] {
            const auto grp = it.get_group();
            const sycl::sub_group sg = it.get_sub_group();
            const int64_t th = (int64_t) it.get_group(0);
            const int64_t t = th / n_heads;
            const int hh = (int) (th % n_heads);
            const int lid = (int) it.get_local_id(0);
            const int lane = (int) sg.get_local_linear_id(), sgi = (int) sg.get_group_linear_id();
            const int n_sg = (int) sg.get_group_linear_range();
            const float* qt = q + th * dim;
            for (int d = lid; d < dim; d += WG) q_l[d] = qt[d];
            sycl::group_barrier(it.get_group());
            const int64_t p = pos[t];
            const int64_t first = sycl::max<int64_t>(0, p - window + 1);
            const int64_t nw = p - first + 1;
            const int64_t nc = count ? count[t] : 0;
            const int64_t nk = nw + nc;
            const auto key_of = [&](int64_t j) -> const float* {
                return j < nw ? raw + ((first + j) % ring) * dim : comp + (int64_t) ids[t * id_cap + (j - nw)] * comp_stride;
            };
            for (int64_t j = sgi; j < nk; j += n_sg) {
                const float* kj = key_of(j);
                float dot = 0.0f;
                for (int d = lane; d < dim; d += WARP) dot += q_l[d] * kj[d];
                dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
                if (lane == 0) p_l[j] = dot * scale;
            }
            sycl::group_barrier(it.get_group());
            const float sink = sinks[hh];
            float m = sink;
            for (int64_t j = lid; j < nk; j += WG) m = sycl::fmax(m, p_l[j]);
            m = sycl::reduce_over_group(grp, m, sycl::maximum<float>());
            float sum = 0.0f;
            for (int64_t j = lid; j < nk; j += WG) {
                const float e = sycl::exp(p_l[j] - m);
                p_l[j] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(grp, sum, sycl::plus<float>()) + sycl::exp(sink - m);
            sycl::group_barrier(it.get_group());
            const float inv = 1.0f / sum;
            for (int d = lid; d < dim; d += WG) {
                float acc = 0.0f;
                for (int64_t j = 0; j < nk; ++j) acc += p_l[j] * key_of(j)[d];
                out[th * dim + d] = acc * inv;
            }
        });
    });
    sync_if_needed(stream, event, "ds41_sink_attn");
}

}  // namespace strata::kernels
