// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/reference/ds41_host.cpp - DeepSeek-V4.1-Flash (deepseek41) on the host, in F32: the whole model's forward pass
// over one sequence, as a deterministic reference for the engine's block (PLAN, Milestone 4).  No GPU, no SYCL.
//
//   ds41_host MODEL.gguf IDS OUT.f32 [--from P] [--threads N] [--dump]
//
// IDS: comma-separated token ids (positions 0 .. T-1).  OUT.f32: the logits [T - P][n_vocab] of positions P .. T-1
// (default P = T - 1), raw little-endian floats, which tools/xemaya/compare_ref.py --forced compares with a llama.cpp
// reference.  --dump prints each layer's sums of the named intermediates (llama.cpp's cb() names, for
// llama-eval-callback).
//
// The arithmetic follows llama.cpp's src/models/deepseek41.cpp (graph, build_attention41, build_engram, engram_hash),
// deepseek4.cpp (the mHC mixes), src/llama-graph.cpp (build_moe_ffn, build_ffn's clamped SwiGLU),
// src/llama-kv-cache-dsv4.cpp (which compressed rows a position sees) and ggml's CPU rope_yarn, at master 8642541b2;
// MIT License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE.
//
// A layer's dense weights are dequantized once (every token goes through the layer together), a routed expert when a
// token first takes it; both are dropped after the layer.  What it does not do: image positions, several sequences.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_spec.hpp"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using strata::GgufModel;
using strata::ModelSpec;
using strata::TensorInfo;

int g_threads = 16;
bool g_dump = false;

template <class F> void parallel_for(int64_t n, F f) {
    const int nt = (int) std::max<int64_t>(1, std::min<int64_t>(g_threads, n));
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    auto work = [&] {
        for (;;) {
            const int64_t i = next.fetch_add(1);
            if (i >= n) return;
            f(i);
        }
    };
    for (int i = 1; i < nt; ++i) th.emplace_back(work);
    work();
    for (auto& t : th) t.join();
}

// 16 partial sums: vectorizes without reordering a float sum behind the compiler's back
float dot(const float* a, const float* b, int64_t n) {
    float acc[16] = {0};
    int64_t i = 0;
    for (; i + 16 <= n; i += 16)
        for (int j = 0; j < 16; ++j) acc[j] += a[i + j] * b[i + j];
    float s = 0;
    for (int j = 0; j < 16; ++j) s += acc[j];
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

struct Mat {
    std::vector<float> w;   // [n_out][n_in]
    int64_t n_in = 0, n_out = 0;
};

const TensorInfo& tensor(const GgufModel& m, const std::string& name, size_t& shard) {
    const TensorInfo* t = m.find(name, &shard);
    if (t == nullptr) throw std::runtime_error("no tensor " + name);
    if (!m.in_bounds(*t, shard)) throw std::runtime_error(name + " is out of the file's bounds");
    return *t;
}

void to_float(uint32_t type, const uint8_t* src, float* dst, int64_t n) {
    if (type == GGML_TYPE_F32) { std::memcpy(dst, src, (size_t) n * 4); return; }
    const ggml_type_traits* tr = ggml_get_type_traits((ggml_type) type);
    if (tr == nullptr || tr->to_float == nullptr) throw std::runtime_error("no dequantizer for a tensor type");
    tr->to_float(src, dst, n);
}

/// `rows` rows of `n_in` values from row `first` of a tensor whose dim 0 is n_in
void rows_of(const GgufModel& m, const std::string& name, int64_t first, int64_t rows, int64_t n_in, float* dst) {
    size_t sh = 0;
    const TensorInfo& t = tensor(m, name, sh);
    if ((int64_t) t.shape[0] != n_in || (first + rows) * n_in > (int64_t) t.elements())
        throw std::runtime_error(name + ": rows out of range");
    const size_t rb = ggml_row_size((ggml_type) t.type, n_in);
    const uint8_t* base = m.shard(sh).tensor_data(t) + (size_t) first * rb;
    parallel_for(rows, [&](int64_t r) { to_float(t.type, base + (size_t) r * rb, dst + r * n_in, n_in); });
}

Mat mat(const GgufModel& m, const std::string& name, int64_t n_in, int64_t n_out) {
    size_t sh = 0;
    const TensorInfo& t = tensor(m, name, sh);
    if ((int64_t) t.elements() != n_in * n_out || (int64_t) t.shape[0] != n_in)
        throw std::runtime_error(name + " has another shape");
    Mat a;
    a.n_in = n_in, a.n_out = n_out;
    a.w.resize((size_t) (n_in * n_out));
    rows_of(m, name, 0, n_out, n_in, a.w.data());
    return a;
}

std::vector<float> vec(const GgufModel& m, const std::string& name, int64_t n) {
    size_t sh = 0;
    const TensorInfo& t = tensor(m, name, sh);
    if ((int64_t) t.elements() != n) throw std::runtime_error(name + " has another length");
    std::vector<float> v((size_t) n);
    to_float(t.type, m.shard(sh).tensor_data(t), v.data(), n);
    return v;
}

/// y [T][n_out] = w x, x [T][n_in]
std::vector<float> mul(const Mat& a, const float* x, int64_t T) {
    std::vector<float> y((size_t) (T * a.n_out));
    parallel_for(a.n_out, [&](int64_t r) {
        const float* w = a.w.data() + r * a.n_in;
        for (int64_t t = 0; t < T; ++t) y[(size_t) (t * a.n_out + r)] = dot(w, x + t * a.n_in, a.n_in);
    });
    return y;
}

/// RMSNorm of rows of n, with weight w (null: none)
void rms_norm(float* x, int64_t rows, int64_t n, const float* w, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        float* v = x + r * n;
        double ss = 0;
        for (int64_t i = 0; i < n; ++i) ss += (double) v[i] * v[i];
        const float inv = 1.0f / std::sqrt((float) (ss / (double) n) + eps);
        for (int64_t i = 0; i < n; ++i) v[i] = v[i] * inv * (w ? w[i] : 1.0f);
    }
}

double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

struct Rope {
    int n_rot = 64, offset = 448, n_ctx_orig = 0;
    float freq_base = 10000.0f, freq_scale = 1.0f, ext_factor = 0.0f, attn_factor = 1.0f, beta_fast = 0, beta_slow = 0;
};

/// ggml's rope_yarn (NORM pairs) on dims [offset, offset + n_rot) of `rows` rows of `dim` at position `pos`
void rope(float* x, int32_t pos, const Rope& r, bool inverse, int64_t rows, int64_t dim) {
    float corr[2] = {0, 0};
    if (r.ext_factor != 0.0f) {
        const auto cd = [&](float n_rot) {
            return (float) r.n_rot * logf((float) r.n_ctx_orig / (n_rot * 2 * 3.14159265358979323846f /* pi written out: MSVC's <cmath> has no M_PI without a define */)) / (2 * logf(r.freq_base));
        };
        corr[0] = std::max(0.0f, floorf(cd(r.beta_fast)));
        corr[1] = std::min((float) (r.n_rot - 1), ceilf(cd(r.beta_slow)));
    }
    const float theta_scale = powf(r.freq_base, -2.0f / r.n_rot);
    for (int64_t row = 0; row < rows; ++row) {
        float theta_extrap = (float) pos;
        for (int i0 = 0; i0 < r.n_rot; i0 += 2) {
            const float theta_interp = r.freq_scale * theta_extrap;
            float theta = theta_interp, mscale = r.attn_factor;
            if (r.ext_factor != 0.0f) {
                const float y = (i0 / 2 - corr[0]) / std::max(0.001f, corr[1] - corr[0]);
                const float ramp = (1 - std::min(1.0f, std::max(0.0f, y))) * r.ext_factor;
                theta = theta_interp * (1 - ramp) + theta_extrap * ramp;
                mscale *= 1.0f + 0.1f * logf(1.0f / r.freq_scale);
            }
            const float ct = cosf(theta) * mscale, st = sinf(theta) * mscale * (inverse ? -1.0f : 1.0f);
            float* v = x + row * dim + r.offset + i0;
            const float x0 = v[0], x1 = v[1];
            v[0] = x0 * ct - x1 * st;
            v[1] = x0 * st + x1 * ct;
            theta_extrap *= theta_scale;
        }
    }
}

constexpr int HC = 4, MIX = (2 + HC) * HC;

struct Mixes {
    std::vector<float> pre, post, comb;   // [T][4], [T][4], [T][4 src][4 dst] as c[dst + 4 src]
};

/// build_hc_mixes: the mixes fn . flat(x), scaled by 1 / rms(x), into pre, post and the Sinkhorn comb
Mixes hc_mixes(const Mat& fn, const std::vector<float>& scale, const std::vector<float>& base, const float* x, int64_t N,
               int64_t T, float rms_eps, float hc_eps, int iters) {
    const std::vector<float> mixes = mul(fn, x, T);
    Mixes o;
    o.pre.resize((size_t) T * HC), o.post.resize((size_t) T * HC), o.comb.resize((size_t) T * HC * HC);
    for (int64_t t = 0; t < T; ++t) {
        double ss = 0;
        for (int64_t i = 0; i < HC * N; ++i) ss += (double) x[t * HC * N + i] * x[t * HC * N + i];
        const double inv = 1.0 / std::sqrt(ss / (double) (HC * N) + rms_eps);
        double m[MIX], c[HC * HC];
        for (int i = 0; i < MIX; ++i) m[i] = mixes[(size_t) (t * MIX + i)] * inv;
        for (int s = 0; s < HC; ++s) {
            o.pre[(size_t) (t * HC + s)] = (float) (sigm(m[s] * scale[0] + base[(size_t) s]) + hc_eps);
            o.post[(size_t) (t * HC + s)] = (float) (2.0 * sigm(m[HC + s] * scale[1] + base[(size_t) (HC + s)]));
        }
        for (int src = 0; src < HC; ++src) {
            double mx = -HUGE_VAL, sum = 0;
            for (int dst = 0; dst < HC; ++dst)
                mx = std::max(mx, m[2 * HC + dst + HC * src] * scale[2] + base[(size_t) (2 * HC + dst + HC * src)]);
            for (int dst = 0; dst < HC; ++dst) {
                c[dst + HC * src] = std::exp(m[2 * HC + dst + HC * src] * scale[2] + base[(size_t) (2 * HC + dst + HC * src)] - mx);
                sum += c[dst + HC * src];
            }
            for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] = c[dst + HC * src] / sum + hc_eps;
        }
        auto cols = [&] {
            for (int dst = 0; dst < HC; ++dst) {
                double sum = hc_eps;
                for (int src = 0; src < HC; ++src) sum += c[dst + HC * src];
                for (int src = 0; src < HC; ++src) c[dst + HC * src] /= sum;
            }
        };
        auto rows = [&] {
            for (int src = 0; src < HC; ++src) {
                double sum = hc_eps;
                for (int dst = 0; dst < HC; ++dst) sum += c[dst + HC * src];
                for (int dst = 0; dst < HC; ++dst) c[dst + HC * src] /= sum;
            }
        };
        cols();
        for (int i = 1; i < iters; ++i) { rows(); cols(); }
        for (int i = 0; i < HC * HC; ++i) o.comb[(size_t) (t * HC * HC + i)] = (float) c[i];
    }
    return o;
}

/// build_hc_pre: y[t] = sum_s pre[t][s] x[t][s]
std::vector<float> collapse(const float* x, const float* pre, int64_t N, int64_t T) {
    std::vector<float> y((size_t) (T * N));
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) {
            float v = 0;
            for (int s = 0; s < HC; ++s) v += x[(t * HC + s) * N + n] * pre[t * HC + s];
            y[(size_t) (t * N + n)] = v;
        }
    return y;
}

/// build_hc_post: x[t][d] = f[t] post[t][d] + sum_s x[t][s] comb[t][s][d]
void hc_post(std::vector<float>& x, const float* f, const Mixes& m, int64_t N, int64_t T) {
    std::vector<float> out(x.size());
    for (int64_t t = 0; t < T; ++t)
        for (int d = 0; d < HC; ++d)
            for (int64_t n = 0; n < N; ++n) {
                float v = f[t * N + n] * m.post[(size_t) (t * HC + d)];
                for (int s = 0; s < HC; ++s) v += x[(size_t) ((t * HC + s) * N + n)] * m.comb[(size_t) ((t * HC + s) * HC + d)];
                out[(size_t) ((t * HC + d) * N + n)] = v;
            }
    x.swap(out);
}

double total(const std::vector<float>& v) {
    double s = 0;
    for (float f : v) s += f;
    return s;
}
void dump(const char* name, int64_t l, const std::vector<float>& v) {
    if (g_dump) std::printf("dump %-14s l=%-2lld n=%-9zu sum=%.6f\n", name, (long long) l, v.size(), total(v));
}

/// The k largest of s[0 .. n) that are not -inf, ties to the lower index, in ascending index order
std::vector<int32_t> topk(const float* s, int64_t n, int64_t k) {
    std::vector<int32_t> id;
    for (int64_t i = 0; i < n; ++i)
        if (s[i] != -INFINITY) id.push_back((int32_t) i);
    if ((int64_t) id.size() > k) {
        std::stable_sort(id.begin(), id.end(), [&](int32_t a, int32_t b) { return s[a] > s[b]; });
        id.resize((size_t) k);
        std::sort(id.begin(), id.end());
    }
    return id;
}

struct Model {
    const GgufModel& m;
    ModelSpec s;
    int64_t N = 0, H = 0, D = 0, T = 0, n_vocab = 0;
    std::vector<int32_t> ids;
    std::vector<std::vector<float>> comp;          // per KV-source layer: rows [n][D + ID] (latent, index key)
    std::vector<std::vector<std::vector<int32_t>>> sel;   // per top-k source layer, per token: the selected rows
    std::vector<std::vector<int32_t>> cand;               // the candidate source layer: its blocks per token
    bool have_cand = false;

    Rope rope_of(int64_t l, int64_t dim) const {
        Rope r;
        r.n_rot = (int) (s.rope_dims > 0 ? s.rope_dims : 64);
        r.offset = (int) (dim - r.n_rot);
        const bool c = s.compress_ratios[(size_t) l] != 0;
        const bool yarn = s.rope_scaling_type == "yarn";
        r.freq_base = (float) (c ? s.compress_rope_base : s.rope_base);
        r.freq_scale = c && yarn && s.rope_scaling_factor > 0 ? (float) (1.0 / s.rope_scaling_factor) : 1.0f;
        r.ext_factor = c && yarn ? 1.0f : 0.0f;
        r.attn_factor = r.ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / r.freq_scale));
        r.beta_fast = c ? 32.0f : 0.0f;
        r.beta_slow = c ? 1.0f : 0.0f;
        r.n_ctx_orig = c ? (int) s.rope_scaling_orig_ctx : 0;
        return r;
    }

    std::vector<float> attention(int64_t l, const std::vector<float>& cur) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const float eps = (float) s.rms_eps;
        const int64_t QL = s.q_lora_rank, ID = s.idx_key_length, IH = s.idx_n_head, W = s.sliding_window;
        const int64_t ratio = s.compress_ratios[(size_t) l];
        const Rope r = rope_of(l, D), ri = rope_of(l, ID);

        std::vector<float> qr = mul(mat(m, p + "attn_q_a.weight", N, QL), cur.data(), T);
        rms_norm(qr.data(), T, QL, vec(m, p + "attn_q_a_norm.weight", QL).data(), eps);
        dump("qr_norm", l, qr);
        std::vector<float> q = mul(mat(m, p + "attn_q_b.weight", QL, H * D), qr.data(), T);
        std::vector<float> kv = mul(mat(m, p + "attn_kv.weight", N, D), cur.data(), T);
        rms_norm(kv.data(), T, D, vec(m, p + "attn_kv_a_norm.weight", D).data(), eps);
        for (int64_t t = 0; t < T; ++t) {
            rope(q.data() + t * H * D, (int32_t) t, r, false, H, D);
            rope(kv.data() + t * D, (int32_t) t, r, false, 1, D);
        }
        dump("q", l, q);
        dump("kv", l, kv);

        const int64_t src = ratio ? s.kv_src[(size_t) l] : -1, tsrc = ratio ? s.topk_src[(size_t) l] : -1;
        if (ratio && src == l) {   // this layer writes the compressed rows
            const std::vector<float> v = mul(mat(m, p + "attn_compressor_kv.weight", N, D), cur.data(), T);
            std::vector<float> g;
            if (ratio > 1) g = mul(mat(m, p + "attn_compressor_gate.weight", N, D), cur.data(), T);
            const int64_t nb = T / ratio;
            std::vector<float> lat((size_t) (nb * D));
            for (int64_t b = 0; b < nb; ++b)
                for (int64_t d = 0; d < D; ++d) {
                    if (ratio == 1) { lat[(size_t) (b * D + d)] = v[(size_t) (b * D + d)]; continue; }
                    float mx = -INFINITY, sum = 0, acc = 0;
                    for (int64_t j = 0; j < ratio; ++j) mx = std::max(mx, g[(size_t) ((b * ratio + j) * D + d)]);
                    for (int64_t j = 0; j < ratio; ++j) sum += std::exp(g[(size_t) ((b * ratio + j) * D + d)] - mx);
                    for (int64_t j = 0; j < ratio; ++j)
                        acc += v[(size_t) ((b * ratio + j) * D + d)] * (std::exp(g[(size_t) ((b * ratio + j) * D + d)] - mx) / sum);
                    lat[(size_t) (b * D + d)] = acc;
                }
            rms_norm(lat.data(), nb, D, vec(m, p + "attn_compressor_norm.weight", D).data(), eps);
            dump("comp_latent", l, lat);
            std::vector<float> ik = mul(mat(m, p + "indexer.attn_k.weight", D, ID), lat.data(), nb);
            rms_norm(ik.data(), nb, ID, vec(m, p + "indexer.k_norm.weight", ID).data(), eps);
            std::vector<float>& rows = comp[(size_t) l];
            rows.assign((size_t) (nb * (D + ID)), 0.0f);
            for (int64_t b = 0; b < nb; ++b) {   // a latent stands at the first position of its block
                rope(lat.data() + b * D, (int32_t) (b * ratio), r, false, 1, D);
                rope(ik.data() + b * ID, (int32_t) (b * ratio), ri, false, 1, ID);
                std::memcpy(rows.data() + b * (D + ID), lat.data() + b * D, (size_t) D * 4);
                std::memcpy(rows.data() + b * (D + ID) + D, ik.data() + b * ID, (size_t) ID * 4);
            }
        }
        if (ratio && tsrc == l) {   // this layer selects: the lightning indexer over the visible rows
            const std::vector<float>& rows = comp[(size_t) src];
            const int64_t n_comp = T / ratio, K = std::min<int64_t>(n_comp, s.idx_top_k);
            auto& mine = sel[(size_t) l];
            mine.assign((size_t) T, {});
            if (n_comp > 0) {
                std::vector<float> iq = mul(mat(m, p + "indexer.attn_q_b.weight", QL, IH * ID), qr.data(), T);
                std::vector<float> iw = mul(mat(m, p + "indexer.proj.weight", N, IH), cur.data(), T);
                const float wscale = 1.0f / std::sqrt((float) (ID * IH));
                const int64_t bs = s.cand_block_size, n_blk = s.cand_src >= 0 ? (n_comp + bs - 1) / bs : 0;
                const bool pick = s.cand_src == l;
                const bool use = s.cand_src >= 0 && s.cand_src < l && have_cand &&
                                 s.compress_ratios[(size_t) s.cand_src] == ratio;
                if (pick) cand.assign((size_t) T, {});
                parallel_for(T, [&](int64_t t) {
                    const int64_t vis = (t + 1) / ratio;
                    if (vis == 0) return;
                    rope(iq.data() + t * IH * ID, (int32_t) t, ri, false, IH, ID);
                    std::vector<float> sc((size_t) n_comp, -INFINITY);
                    for (int64_t c = 0; c < vis; ++c) {
                        float a = 0;
                        for (int64_t h = 0; h < IH; ++h) {
                            const float d = dot(iq.data() + (t * IH + h) * ID, rows.data() + c * (D + ID) + D, ID);
                            a += std::max(0.0f, d) * (iw[(size_t) (t * IH + h)] * wscale);
                        }
                        sc[(size_t) c] = a;
                    }
                    if (pick) {   // level one: the blocks' best rows, the newest block always kept
                        std::vector<float> bmax((size_t) n_blk, -INFINITY);
                        for (int64_t c = 0; c < vis; ++c) bmax[(size_t) (c / bs)] = std::max(bmax[(size_t) (c / bs)], sc[(size_t) c]);
                        bmax[(size_t) ((vis - 1) / bs)] += 1e30f;
                        cand[(size_t) t] = topk(bmax.data(), n_blk, std::min<int64_t>(s.cand_topk_blocks, n_blk));
                    } else if (use) {   // level two: only rows of the candidate blocks
                        std::vector<char> in((size_t) n_blk, 0);
                        for (int32_t b : cand[(size_t) t]) in[(size_t) b] = 1;
                        for (int64_t c = 0; c < vis; ++c)
                            if (!in[(size_t) (c / bs)]) sc[(size_t) c] = -INFINITY;
                    }
                    mine[(size_t) t] = topk(sc.data(), n_comp, K);
                });
                if (pick) have_cand = true;
            }
        }

        // MQA with sinks over the window and the selected compressed rows (K = V)
        const std::vector<float> sinks = vec(m, p + "attn_sinks.weight", H);
        const float scale = 1.0f / std::sqrt((float) D);
        std::vector<float> out((size_t) (T * H * D), 0.0f);
        parallel_for(T * H, [&](int64_t th) {
            const int64_t t = th / H, h = th % H;
            const float* qv = q.data() + (t * H + h) * D;
            std::vector<const float*> keys;
            for (int64_t j = std::max<int64_t>(0, t - W + 1); j <= t; ++j) keys.push_back(kv.data() + j * D);
            if (ratio && !sel[(size_t) tsrc].empty())
                for (int32_t c : sel[(size_t) tsrc][(size_t) t]) keys.push_back(comp[(size_t) src].data() + (int64_t) c * (D + s.idx_key_length));
            std::vector<float> a(keys.size());
            float mx = sinks[(size_t) h];
            for (size_t j = 0; j < keys.size(); ++j) a[j] = dot(qv, keys[j], D) * scale, mx = std::max(mx, a[j]);
            double den = std::exp((double) sinks[(size_t) h] - mx);
            for (size_t j = 0; j < keys.size(); ++j) den += std::exp((double) a[j] - mx);
            float* o = out.data() + (t * H + h) * D;
            for (size_t j = 0; j < keys.size(); ++j) {
                const float pj = (float) (std::exp((double) a[j] - mx) / den);
                for (int64_t d = 0; d < D; ++d) o[d] += pj * keys[j][d];
            }
        });
        dump("attn_comp", l, out);
        for (int64_t t = 0; t < T; ++t) rope(out.data() + t * H * D, (int32_t) t, r, true, H, D);
        // the grouped low-rank output
        const int64_t G = s.o_group_count, OL = s.o_lora_rank, GI = H * D / G;
        std::vector<float> oa((size_t) (T * G * OL));
        {
            size_t sh = 0;
            const TensorInfo& t = tensor(m, p + "attn_output_a.weight", sh);
            if ((int64_t) t.elements() != GI * OL * G) throw std::runtime_error("attn_output_a has another shape");
            for (int64_t g = 0; g < G; ++g) {
                Mat a;
                a.n_in = GI, a.n_out = OL;
                a.w.resize((size_t) (GI * OL));
                rows_of(m, p + "attn_output_a.weight", g * OL, OL, GI, a.w.data());
                parallel_for(OL, [&](int64_t rr) {
                    for (int64_t tt = 0; tt < T; ++tt)
                        oa[(size_t) ((tt * G + g) * OL + rr)] = dot(a.w.data() + rr * GI, out.data() + tt * H * D + g * GI, GI);
                });
            }
        }
        return mul(mat(m, p + "attn_output_b.weight", G * OL, N), oa.data(), T);
    }

    std::vector<float> ffn(int64_t l, const std::vector<float>& cur) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const int64_t E = s.n_expert, K = s.n_expert_used, F = s.n_ff_exp, FS = s.n_ff_exp * std::max<int64_t>(1, s.n_expert_shared);
        const float lim_e = (size_t) l < s.swiglu_clamp_exp.size() ? (float) s.swiglu_clamp_exp[(size_t) l] : 0.0f;
        const float lim_s = (size_t) l < s.swiglu_clamp_shexp.size() ? (float) s.swiglu_clamp_shexp[(size_t) l] : 0.0f;
        auto swiglu = [](float g, float u, float lim) {
            if (lim > 1e-6f) g = std::min(g, lim), u = std::max(-lim, std::min(lim, u));
            return g / (1.0f + std::exp(-g)) * u;
        };
        std::vector<float> out((size_t) (T * N), 0.0f);
        {   // the shared expert
            const std::vector<float> g = mul(mat(m, p + "ffn_gate_shexp.weight", N, FS), cur.data(), T);
            const std::vector<float> u = mul(mat(m, p + "ffn_up_shexp.weight", N, FS), cur.data(), T);
            std::vector<float> h((size_t) (T * FS));
            for (size_t i = 0; i < h.size(); ++i) h[i] = swiglu(g[i], u[i], lim_s);
            out = mul(mat(m, p + "ffn_down_shexp.weight", FS, N), h.data(), T);
            dump("ffn_shexp", l, out);
        }
        // the sqrt-softplus router
        const std::vector<float> logits = mul(mat(m, p + "ffn_gate_inp.weight", N, E), cur.data(), T);
        const std::vector<float> bias = vec(m, p + "exp_probs_b.bias", E);
        std::vector<int32_t> ids((size_t) (T * K));
        std::vector<float> w((size_t) (T * K));
        const bool norm = s.expert_weights_norm && K > 1;
        for (int64_t t = 0; t < T; ++t) {
            std::vector<float> pr((size_t) E), sc((size_t) E);
            for (int64_t e = 0; e < E; ++e) {
                const float x = logits[(size_t) (t * E + e)];
                pr[(size_t) e] = std::sqrt(x > 20.0f ? x : std::log1p(std::exp(x)));
                sc[(size_t) e] = pr[(size_t) e] + bias[(size_t) e];
            }
            std::vector<int32_t> order((size_t) E);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return sc[(size_t) a] > sc[(size_t) b]; });
            float sum = 0;
            for (int64_t i = 0; i < K; ++i) sum += pr[(size_t) order[(size_t) i]];
            for (int64_t i = 0; i < K; ++i) {
                ids[(size_t) (t * K + i)] = order[(size_t) i];
                const float pv = pr[(size_t) order[(size_t) i]];
                w[(size_t) (t * K + i)] = (norm ? pv / (sum + 1e-20f) : pv) * (float) s.expert_weights_scale;
            }
        }
        // the routed experts: by expert, every token that took it
        std::map<int32_t, std::vector<std::pair<int64_t, float>>> by;
        for (int64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < K; ++i) by[ids[(size_t) (t * K + i)]].push_back({t, w[(size_t) (t * K + i)]});
        std::vector<int32_t> used;
        for (const auto& kvp : by) used.push_back(kvp.first);
        std::mutex mu;
        const int outer = g_threads;
        g_threads = 1;   // one expert a thread
        std::atomic<int64_t> next{0};
        std::vector<std::thread> th;
        auto work = [&] {
            std::vector<float> wg((size_t) (F * N)), wu((size_t) (F * N)), wd((size_t) (N * F)), h((size_t) F), y((size_t) N);
            for (;;) {
                const int64_t i = next.fetch_add(1);
                if (i >= (int64_t) used.size()) return;
                const int32_t e = used[(size_t) i];
                rows_of(m, p + "ffn_gate_exps.weight", (int64_t) e * F, F, N, wg.data());
                rows_of(m, p + "ffn_up_exps.weight", (int64_t) e * F, F, N, wu.data());
                rows_of(m, p + "ffn_down_exps.weight", (int64_t) e * N, N, F, wd.data());
                for (const auto& [t, wt] : by[e]) {
                    const float* x = cur.data() + t * N;
                    for (int64_t f = 0; f < F; ++f)
                        h[(size_t) f] = swiglu(dot(wg.data() + f * N, x, N), dot(wu.data() + f * N, x, N), lim_e);
                    for (int64_t n = 0; n < N; ++n) y[(size_t) n] = dot(wd.data() + n * F, h.data(), F) * wt;
                    std::lock_guard<std::mutex> lk(mu);
                    for (int64_t n = 0; n < N; ++n) out[(size_t) (t * N + n)] += y[(size_t) n];
                }
            }
        };
        for (int i = 1; i < outer; ++i) th.emplace_back(work);
        work();
        for (auto& t : th) t.join();
        g_threads = outer;
        return out;
    }

    /// Engram's write into the streams of layer l's input
    void engram(int64_t l, int64_t e, std::vector<float>& x) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const int64_t NH = s.engram_n_heads, HD = s.engram_head_dim, ng = s.engram_ngram, hp = s.engram_heads_per_ngram;
        const uint64_t* mult = s.engram_multipliers.data() + e * ng;
        const uint64_t* primes = s.engram_vocab.data() + e * NH;
        const uint64_t* offs = s.engram_offsets.data() + e * NH;
        const auto& map = s.engram_token_map;
        constexpr int64_t DEAD = -1;
        const int64_t pad = map.at((size_t) s.engram_pad_id);
        const auto cid = [&](int32_t t) -> int64_t { return t < 0 || (size_t) t >= map.size() ? DEAD : (int64_t) map[(size_t) t]; };
        std::vector<float> rows((size_t) (T * NH * HD));
        for (int64_t t = 0; t < T; ++t) {
            uint64_t toks[16];
            if (ng < 2 || ng > 16) throw std::runtime_error("Engram n-gram size");
            int64_t c = cid(ids[(size_t) t]);
            bool blocked = c == DEAD;
            toks[0] = (uint64_t) (blocked ? pad : c);
            for (int64_t k = 1; k < ng; ++k) {
                c = t - k < 0 ? DEAD : cid(ids[(size_t) (t - k)]);
                blocked = blocked || c == DEAD;
                toks[k] = (uint64_t) (blocked ? pad : c);
            }
            uint64_t rolling = toks[0] * mult[0];
            for (int64_t o = 1; o < ng; ++o) {
                rolling ^= toks[o] * mult[o];
                for (int64_t h = 0; h < hp; ++h) {
                    const int64_t col = (o - 1) * hp + h;
                    const int64_t row = (int64_t) (rolling % primes[col] + offs[col]);
                    const int outer = g_threads;
                    g_threads = 1;
                    rows_of(m, "per_layer_token_embd.weight", row, 1, HD, rows.data() + (t * NH + col) * HD);
                    g_threads = outer;
                }
            }
        }
        dump("engram_rows", l, rows);
        const std::vector<float> kv = mul(mat(m, p + "engram_kv.weight", NH * HD, N * (HC + 1)), rows.data(), T);
        const std::vector<float> gate = vec(m, p + "engram_gate.weight", N * HC);
        const float eps = (float) s.rms_eps;
        for (int64_t t = 0; t < T; ++t) {
            const float* value = kv.data() + t * N * (HC + 1) + HC * N;
            for (int st = 0; st < HC; ++st) {
                float* xs = x.data() + (t * HC + st) * N;
                const float* key = kv.data() + t * N * (HC + 1) + st * N;
                double sx = 0, sk = 0;
                for (int64_t n = 0; n < N; ++n) sx += (double) xs[n] * xs[n], sk += (double) key[n] * key[n];
                const float ix = 1.0f / std::sqrt((float) (sx / (double) N) + eps), ik = 1.0f / std::sqrt((float) (sk / (double) N) + eps);
                double d = 0;
                for (int64_t n = 0; n < N; ++n) d += (double) (xs[n] * ix) * (double) (key[n] * ik) * gate[(size_t) (st * N + n)];
                const float dv = (float) d / std::sqrt((float) N);
                const float mag = std::sqrt(std::max(std::fabs(dv), 1e-6f));
                const float g = (float) sigm(dv < 0 ? -mag : mag);
                for (int64_t n = 0; n < N; ++n) xs[n] += value[n] * g;
            }
        }
    }

    std::vector<float> forward(int64_t from) {
        const int64_t L = s.n_layer;
        const float eps = (float) s.rms_eps, hc_eps = (float) s.hc_eps;
        const int iters = (int) s.hc_sinkhorn_iters;
        comp.assign((size_t) L, {});
        sel.assign((size_t) L, {});
        std::vector<float> x((size_t) (T * HC * N));
        for (int64_t t = 0; t < T; ++t) {
            rows_of(m, "token_embd.weight", ids[(size_t) t], 1, N, x.data() + t * HC * N);
            for (int st = 1; st < HC; ++st) std::memcpy(x.data() + (t * HC + st) * N, x.data() + t * HC * N, (size_t) N * 4);
        }
        dump("hc_init", -1, x);
        std::vector<float> pre_prev;
        for (int64_t l = 0; l < L; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const int64_t e = s.engram_index(l);
            if (e >= 0) {
                engram(l, e, x);
                dump("engram_out", l, x);
            }
            const Mixes am = hc_mixes(mat(m, p + "hc_attn_fn.weight", HC * N, MIX), vec(m, p + "hc_attn_scale.weight", 3),
                                      vec(m, p + "hc_attn_base.weight", MIX), x.data(), N, T, eps, hc_eps, iters);
            std::vector<float> cur;
            if (pre_prev.empty()) {
                cur.resize((size_t) (T * N));
                for (int64_t t = 0; t < T; ++t) std::memcpy(cur.data() + t * N, x.data() + t * HC * N, (size_t) N * 4);
            } else {
                cur = collapse(x.data(), pre_prev.data(), N, T);
            }
            dump("hc_attn_pre", l, cur);
            rms_norm(cur.data(), T, N, vec(m, p + "attn_norm.weight", N).data(), eps);
            const std::vector<float> a = attention(l, cur);
            dump("attn_out", l, a);
            hc_post(x, a.data(), am, N, T);
            dump("hc_attn_post", l, x);
            const Mixes fm = hc_mixes(mat(m, p + "hc_ffn_fn.weight", HC * N, MIX), vec(m, p + "hc_ffn_scale.weight", 3),
                                      vec(m, p + "hc_ffn_base.weight", MIX), x.data(), N, T, eps, hc_eps, iters);
            cur = collapse(x.data(), am.pre.data(), N, T);
            dump("hc_ffn_pre", l, cur);
            rms_norm(cur.data(), T, N, vec(m, p + "ffn_norm.weight", N).data(), eps);
            const std::vector<float> f = ffn(l, cur);
            dump("ffn_out", l, f);
            hc_post(x, f.data(), fm, N, T);
            dump("l_out", l, x);
            pre_prev = fm.pre;
            std::fprintf(stderr, "ds41_host: layer %lld done\n", (long long) l);
        }
        const int64_t R = T - from;
        std::vector<float> cur = collapse(x.data() + from * HC * N, pre_prev.data() + from * HC, N, R);
        rms_norm(cur.data(), R, N, vec(m, "output_norm.weight", N).data(), eps);
        dump("result_norm", -1, cur);
        return mul(mat(m, "output.weight", N, n_vocab), cur.data(), R);
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::fprintf(stderr, "usage: ds41_host MODEL.gguf IDS OUT.f32 [--from P] [--threads N] [--dump]\n");
            return 2;
        }
        int64_t from = -1;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--from" && i + 1 < argc) from = std::atoll(argv[++i]);
            else if (a == "--threads" && i + 1 < argc) g_threads = std::atoi(argv[++i]);
            else if (a == "--dump") g_dump = true;
            else throw std::runtime_error("unknown argument " + a);
        }
        const GgufModel gm = GgufModel::open(argv[1]);
        Model md{gm, {}};
        std::string err;
        if (!strata::read_model_spec(gm.meta(), md.s, err)) throw std::runtime_error(err);
        if (md.s.arch != strata::ModelArch::DeepSeek41) throw std::runtime_error("not a deepseek41 model");
        {
            std::ifstream f(argv[2]);
            std::stringstream ss;
            ss << f.rdbuf();
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (tok.find_first_of("0123456789") != std::string::npos) md.ids.push_back((int32_t) std::atol(tok.c_str()));
        }
        md.T = (int64_t) md.ids.size();
        md.N = md.s.n_embd, md.H = md.s.n_head, md.D = md.s.key_length;
        size_t sh = 0;
        md.n_vocab = (int64_t) tensor(gm, "token_embd.weight", sh).shape[1];
        if (md.T == 0) throw std::runtime_error("no token ids");
        if (from < 0) from = md.T - 1;
        if (from >= md.T) throw std::runtime_error("--from is past the sequence");
        const ModelSpec& s = md.s;
        std::fprintf(stderr, "ds41_host: %lld tokens, %lld layers, width %lld, %lld heads of %lld, %lld experts (%lld used), "
                             "window %lld, top-k %lld, rope %s factor %g orig %g base %g / %g, rot %lld, Engram layers %zu, "
                             "%d threads\n", (long long) md.T, (long long) s.n_layer, (long long) md.N, (long long) md.H,
                     (long long) md.D, (long long) s.n_expert, (long long) s.n_expert_used, (long long) s.sliding_window,
                     (long long) s.idx_top_k, s.rope_scaling_type.c_str(), s.rope_scaling_factor, s.rope_scaling_orig_ctx,
                     s.rope_base, s.compress_rope_base, (long long) s.rope_dims, s.engram_layers.size(), g_threads);
        const std::vector<float> logits = md.forward(from);
        std::ofstream o(argv[3], std::ios::binary);
        o.write((const char*) logits.data(), (std::streamsize) (logits.size() * 4));
        const int64_t R = md.T - from;
        for (int64_t r = 0; r < R; ++r) {
            const float* v = logits.data() + r * md.n_vocab;
            const int64_t best = std::max_element(v, v + md.n_vocab) - v;
            std::printf("pos %lld: argmax %lld (%.4f)\n", (long long) (from + r), (long long) best, v[best]);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ds41_host: %s\n", e.what());
        return 1;
    }
}
