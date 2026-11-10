// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/reference/mm_host.cpp - MiniMax-M3 (minimax-m3) on the host, in F32: the whole model's forward pass over one
// sequence, as a deterministic reference for the engine's block.  No GPU, no SYCL.
//
//   mm_host MODEL.gguf IDS OUT.f32 [--from P] [--threads N] [--dump] [--dense] [--ggml-acts]
//
// IDS: comma-separated token ids (positions 0 .. T-1).  OUT.f32: the logits [T - P][n_vocab] of positions P .. T-1
// (default P = T - 1), raw little-endian floats, which tools/xemaya/compare_ref.py --forced compares with a llama.cpp
// reference.  --dump prints the named intermediates of each layer in the engine's XEMAYA_DUMP_LAYERS form (llama.cpp's
// cb() names; tools/xemaya/compare_evalcb.py compares them with llama-eval-callback).  --dense: every block attends
// every cell (no block selection).  --ggml-acts: each product's activations rounded to int8 as ggml's CPU backend
// rounds them for a quantized weight (ggml_acts below): llama.cpp's CPU answers, not the model's in exact floats.
//
// The arithmetic follows llama.cpp's src/models/minimax-m3.cpp (the graph and its sparse attention),
// src/llama-graph.cpp (build_moe_ffn, build_ffn's swiglu-oai), src/llama-kv-cache-msa.cpp (the positions a token
// sees) and ggml's CPU rope (NEOX pairs) and swiglu_oai, at master 8642541b2; MIT License, Copyright (c) 2023-2026
// The ggml authors, see third_party/main/ggml/LICENSE.  K, V and the indexer's keys are rounded to FP16, as
// llama.cpp's caches hold them.
//
// A layer's dense weights are dequantized once (every token goes through the layer together), a routed expert for the
// tokens that take it; both are dropped after the layer.  What it does not do: several sequences, pictures.
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
    uint32_t type = 0;      // the tensor's GGUF type
};

bool g_ggml_acts = false;   // --ggml-acts

/// --ggml-acts: x as ggml's CPU backend hands it to a product with a weight of `type` (ggml_compute_forward_mul_mat
/// converts the activations to the weight's vec_dot_type): int8 with a scale a block of 32 (Q8_0, for Q8_0 weights:
/// quantize_row_q8_0, the scale kept in FP16) or of 256 (Q8_K, for the K- and I-quants: quantize_row_q8_K); F32, F16
/// and BF16 weights take the floats.  The values back as floats: the integer dot products are exact in float here.
void ggml_acts(uint32_t type, const float* x, float* y, int64_t n) {
    const bool q80 = type == GGML_TYPE_Q8_0;
    const bool q8k = type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
                     type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_XS ||
                     type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S ||
                     type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ4_XS;
    if (!g_ggml_acts || !(q80 || q8k)) {
        if (y != x) std::memcpy(y, x, (size_t) n * 4);
        return;
    }
    const int64_t B = q80 ? 32 : 256;
    for (int64_t b = 0; b < n; b += B) {
        const int64_t e = std::min(n, b + B);
        if (q80) {
            float amax = 0;
            for (int64_t i = b; i < e; ++i) amax = std::max(amax, std::fabs(x[i]));
            const float d = amax / 127.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
            const float dh = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
            for (int64_t i = b; i < e; ++i) y[i] = std::nearbyint(x[i] * id) * dh;
        } else {
            float mx = 0, amax = 0;
            for (int64_t i = b; i < e; ++i)
                if (std::fabs(x[i]) > amax) amax = std::fabs(x[i]), mx = x[i];
            if (amax == 0.0f) {
                for (int64_t i = b; i < e; ++i) y[i] = 0;
                continue;
            }
            const float iscale = -127.0f / mx;
            for (int64_t i = b; i < e; ++i) y[i] = std::min(127.0f, std::nearbyint(iscale * x[i])) / iscale;
        }
    }
}

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
    a.type = t.type;
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
    std::vector<float> xq;
    if (g_ggml_acts) {
        xq.resize((size_t) (T * a.n_in));
        for (int64_t t = 0; t < T; ++t) ggml_acts(a.type, x + t * a.n_in, xq.data() + t * a.n_in, a.n_in);
        x = xq.data();
    }
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


bool g_dense = false;
int64_t g_layers = -1;   // --layers: stop after this many (with --dump)

float f16r(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }

/// ggml's rope (NEOX pairs: i with i + n_rot / 2) on the first n_rot dims of `rows` rows of `dim` at position `pos`
void rope_neox(float* x, int32_t pos, int n_rot, float freq_base, int64_t rows, int64_t dim) {
    const float theta_scale = powf(freq_base, -2.0f / (float) n_rot);
    for (int64_t row = 0; row < rows; ++row) {
        float theta = (float) pos;
        float* v = x + row * dim;
        for (int ic = 0; ic < n_rot / 2; ++ic) {
            const float ct = cosf(theta), st = sinf(theta);
            const float x0 = v[ic], x1 = v[ic + n_rot / 2];
            v[ic] = x0 * ct - x1 * st;
            v[ic + n_rot / 2] = x0 * st + x1 * ct;
            theta *= theta_scale;
        }
    }
}

/// ggml_compute_forward_swiglu_oai_f32 with llama.cpp's constants
float swiglu_oai(float g, float u) {
    const float alpha = 1.702f, limit = 7.0f;
    const float x = std::min(g, limit), y = std::clamp(u, -limit, limit);
    return x / (1.0f + expf(alpha * (-x))) * (y + 1.0f);
}

void dump(const char* name, int64_t l, const std::vector<float>& v, int64_t rows) {
    if (!g_dump) return;
    const int64_t width = (int64_t) v.size() / rows;
    double sum = 0;
    for (float f : v) sum += f;
    std::printf("dump %s-%lld = {%lld, %lld} sum = %.6f vals", name, (long long) l, (long long) width, (long long) rows, sum);
    for (int64_t r = 0; r < rows; ++r) {
        if (rows > 6 && r >= 3 && r < rows - 3) continue;
        for (int64_t c = 0; c < width; ++c)
            if (width <= 6 || c < 3 || c >= width - 3) std::printf(" %.4f", v[(size_t) (r * width + c)]);
    }
    std::printf("\n");
}

struct Model {
    const GgufModel& m;
    ModelSpec s;
    std::vector<int32_t> ids;
    int64_t T = 0, N = 0, n_vocab = 0;

    /// One block's attention over the T normed rows `cur`
    std::vector<float> attention(int64_t l, const std::vector<float>& cur) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const int64_t NH = s.n_head, NKV = s.n_head_kv[(size_t) l], HD = s.key_length, G = NH / NKV;
        const float eps = (float) s.rms_eps;
        std::vector<float> q = mul(mat(m, p + "attn_q.weight", N, NH * HD), cur.data(), T);
        std::vector<float> k = mul(mat(m, p + "attn_k.weight", N, NKV * HD), cur.data(), T);
        std::vector<float> v = mul(mat(m, p + "attn_v.weight", N, NKV * HD), cur.data(), T);
        rms_norm(q.data(), T * NH, HD, vec(m, p + "attn_q_norm.weight", HD).data(), eps);
        rms_norm(k.data(), T * NKV, HD, vec(m, p + "attn_k_norm.weight", HD).data(), eps);
        for (int64_t t = 0; t < T; ++t) {
            rope_neox(q.data() + t * NH * HD, (int32_t) t, (int) s.rope_dims, (float) s.rope_base, NH, HD);
            rope_neox(k.data() + t * NKV * HD, (int32_t) t, (int) s.rope_dims, (float) s.rope_base, NKV, HD);
        }
        dump("Qcur", l, q, T);
        dump("Kcur", l, k, T);
        dump("Vcur", l, v, T);
        for (auto& f : k) f = f16r(f);   // the caches are FP16
        for (auto& f : v) f = f16r(f);
        // the sparse attention's blocks: sel[t][h], empty where the token attends every cell
        const bool sparse = !g_dense && l >= s.n_dense_lead;
        const int64_t blk = s.idx_block_size, top = s.idx_top_k, loc = s.idx_local_blocks, ID = s.idx_key_length;
        std::vector<std::vector<int32_t>> sel((size_t) (T * NKV));
        if (sparse && (T - 1) / blk + 1 > top) {
            std::vector<float> iq = mul(mat(m, p + "indexer.q_proj.weight", N, NKV * ID), cur.data(), T);
            std::vector<float> ik = mul(mat(m, p + "indexer.k_proj.weight", N, ID), cur.data(), T);
            rms_norm(iq.data(), T * NKV, ID, vec(m, p + "indexer.q_norm.weight", ID).data(), eps);
            rms_norm(ik.data(), T, ID, vec(m, p + "indexer.k_norm.weight", ID).data(), eps);
            for (int64_t t = 0; t < T; ++t) {
                rope_neox(iq.data() + t * NKV * ID, (int32_t) t, (int) s.rope_dims, (float) s.rope_base, NKV, ID);
                rope_neox(ik.data() + t * ID, (int32_t) t, (int) s.rope_dims, (float) s.rope_base, 1, ID);
            }
            for (auto& f : ik) f = f16r(f);
            parallel_for(T * NKV, [&](int64_t r) {
                const int64_t t = r / NKV, vis = t / blk + 1;
                if (vis <= top) return;
                std::vector<float> bs((size_t) vis, -INFINITY);
                for (int64_t c = 0; c <= t; ++c)
                    bs[(size_t) (c / blk)] = std::max(bs[(size_t) (c / blk)], dot(iq.data() + r * ID, ik.data() + c * ID, ID));
                for (int64_t i = 0; i < loc && vis - 1 - i >= 0; ++i) bs[(size_t) (vis - 1 - i)] += 1e30f;
                std::vector<int32_t> order((size_t) vis);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return bs[(size_t) a] > bs[(size_t) b]; });
                order.resize((size_t) top);
                std::sort(order.begin(), order.end());
                sel[(size_t) r] = order;
            });
        }
        std::vector<float> o((size_t) (T * NH * HD));
        const float scale = 1.0f / std::sqrt((float) HD);
        parallel_for(T * NH, [&](int64_t r) {
            const int64_t t = r / NH, h = r % NH, hk = h / G;
            std::vector<int64_t> cells;
            const auto& b = sel[(size_t) (t * NKV + hk)];
            if (b.empty())
                for (int64_t c = 0; c <= t; ++c) cells.push_back(c);
            else
                for (int32_t bi : b)
                    for (int64_t c = (int64_t) bi * blk; c < (int64_t) (bi + 1) * blk && c <= t; ++c) cells.push_back(c);
            std::vector<double> w(cells.size());
            double mx = -HUGE_VAL, sum = 0;
            for (size_t j = 0; j < cells.size(); ++j) {
                w[j] = (double) dot(q.data() + r * HD, k.data() + (cells[j] * NKV + hk) * HD, HD) * scale;
                mx = std::max(mx, w[j]);
            }
            for (auto& x : w) sum += (x = std::exp(x - mx));
            float* out = o.data() + r * HD;
            for (int64_t d = 0; d < HD; ++d) {
                double acc = 0;
                for (size_t j = 0; j < cells.size(); ++j) acc += w[j] * v[(size_t) ((cells[j] * NKV + hk) * HD + d)];
                out[d] = (float) (acc / sum);
            }
        });
        dump("kqv_out", l, o, T);
        return mul(mat(m, p + "attn_output.weight", NH * HD, N), o.data(), T);
    }

    /// gate / up, swiglu-oai, down over T rows
    std::vector<float> swiglu_ffn(const std::string& gate, const std::string& up, const std::string& down, int64_t F,
                                  const std::vector<float>& cur, int64_t l) {
        std::vector<float> g = mul(mat(m, gate, N, F), cur.data(), T);
        const std::vector<float> u = mul(mat(m, up, N, F), cur.data(), T);
        dump("ffn_gate", l, g, T);
        dump("ffn_up", l, u, T);
        for (size_t i = 0; i < g.size(); ++i) g[i] = swiglu_oai(g[i], u[i]);
        dump("ffn_swiglu_oai", l, g, T);
        return mul(mat(m, down, F, N), g.data(), T);
    }

    std::vector<float> ffn(int64_t l, const std::vector<float>& cur) {
        const std::string p = "blk." + std::to_string(l) + ".";
        if (l < s.n_dense_lead)
            return swiglu_ffn(p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight", s.n_ff, cur, l);
        const int64_t E = s.n_expert, K = s.n_expert_used, F = s.n_ff_exp;
        // build_moe_ffn: sigmoid probabilities, the selection by probability + bias, the weights the probabilities
        const std::vector<float> logits = mul(mat(m, p + "ffn_gate_inp.weight", N, E), cur.data(), T);
        const std::vector<float> bias = vec(m, p + "exp_probs_b.bias", E);
        std::vector<std::vector<std::pair<int64_t, float>>> takers((size_t) E);   // expert -> (token, weight)
        std::vector<float> wdump((size_t) (T * K));
        for (int64_t t = 0; t < T; ++t) {
            std::vector<float> pr((size_t) E), sp((size_t) E);
            for (int64_t e = 0; e < E; ++e) {
                pr[(size_t) e] = 1.0f / (1.0f + expf(-logits[(size_t) (t * E + e)]));
                sp[(size_t) e] = pr[(size_t) e] + bias[(size_t) e];
            }
            std::vector<int32_t> order((size_t) E);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return sp[(size_t) a] > sp[(size_t) b]; });
            float sum = 0;
            for (int64_t i = 0; i < K; ++i) sum += pr[(size_t) order[(size_t) i]];
            sum = std::max(sum, 6.103515625e-5f);
            for (int64_t i = 0; i < K; ++i) {
                float w = pr[(size_t) order[(size_t) i]];
                if (s.expert_weights_norm) w /= sum;
                if (s.expert_weights_scale != 0.0 && s.expert_weights_scale != 1.0) w *= (float) s.expert_weights_scale;
                takers[(size_t) order[(size_t) i]].push_back({t, w});
                wdump[(size_t) (t * K + i)] = w;
            }
        }
        dump("ffn_moe_weights_scaled", l, wdump, T);
        std::vector<float> out((size_t) (T * N), 0.0f);
        std::vector<float> wg((size_t) (F * N)), wu((size_t) (F * N)), wd((size_t) (N * F));
        for (int64_t e = 0; e < E; ++e) {
            const auto& tk = takers[(size_t) e];
            if (tk.empty()) continue;
            size_t esh = 0;
            const uint32_t tg = tensor(m, p + "ffn_gate_exps.weight", esh).type,
                           tu = tensor(m, p + "ffn_up_exps.weight", esh).type,
                           td = tensor(m, p + "ffn_down_exps.weight", esh).type;
            rows_of(m, p + "ffn_gate_exps.weight", e * F, F, N, wg.data());
            rows_of(m, p + "ffn_up_exps.weight", e * F, F, N, wu.data());
            rows_of(m, p + "ffn_down_exps.weight", e * N, N, F, wd.data());
            for (const auto& [t, w] : tk) {
                std::vector<float> xg((size_t) N), xu((size_t) N);
                ggml_acts(tg, cur.data() + t * N, xg.data(), N);
                ggml_acts(tu, cur.data() + t * N, xu.data(), N);
                std::vector<float> h((size_t) F);
                parallel_for(F, [&](int64_t f) {
                    h[(size_t) f] = swiglu_oai(dot(wg.data() + f * N, xg.data(), N), dot(wu.data() + f * N, xu.data(), N));
                });
                ggml_acts(td, h.data(), h.data(), F);
                const float wt = w;
                const int64_t tt = t;
                parallel_for(N, [&](int64_t n) { out[(size_t) (tt * N + n)] += dot(wd.data() + n * F, h.data(), F) * wt; });
            }
        }
        dump("ffn_moe_out", l, out, T);
        const std::vector<float> sh = swiglu_ffn(p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight",
                                                 p + "ffn_down_shexp.weight", s.n_ff_shexp, cur, l);
        dump("ffn_shexp", l, sh, T);
        for (size_t i = 0; i < out.size(); ++i) out[i] += sh[i];
        return out;
    }

    std::vector<float> forward(int64_t from) {
        const float eps = (float) s.rms_eps;
        std::vector<float> x((size_t) (T * N));
        for (int64_t t = 0; t < T; ++t) {
            if (ids[(size_t) t] < 0 || ids[(size_t) t] >= n_vocab) throw std::runtime_error("a token is out of the vocabulary");
            rows_of(m, "token_embd.weight", ids[(size_t) t], 1, N, x.data() + t * N);
        }
        for (int64_t l = 0; l < s.n_layer; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            std::vector<float> cur = x;
            rms_norm(cur.data(), T, N, vec(m, p + "attn_norm.weight", N).data(), eps);
            const std::vector<float> a = attention(l, cur);
            for (size_t i = 0; i < x.size(); ++i) x[i] += a[i];
            dump("ffn_inp", l, x, T);
            cur = x;
            rms_norm(cur.data(), T, N, vec(m, p + "ffn_norm.weight", N).data(), eps);
            dump("ffn_norm", l, cur, T);
            const std::vector<float> f = ffn(l, cur);
            dump("ffn_out", l, f, T);
            for (size_t i = 0; i < x.size(); ++i) x[i] += f[i];
            dump("l_out", l, x, T);
            std::fprintf(stderr, "mm_host: layer %lld done\n", (long long) l);
            if (l + 1 == g_layers) std::exit(0);
        }
        const int64_t R = T - from;
        std::vector<float> cur(x.begin() + from * N, x.end());
        rms_norm(cur.data(), R, N, vec(m, "output_norm.weight", N).data(), eps);
        return mul(mat(m, "output.weight", N, n_vocab), cur.data(), R);
    }
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) {
            std::fprintf(stderr, "usage: mm_host MODEL.gguf IDS OUT.f32 [--from P] [--threads N] [--dump] [--dense] [--ggml-acts]\n");
            return 2;
        }
        int64_t from = -1;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--from" && i + 1 < argc) from = std::atoll(argv[++i]);
            else if (a == "--threads" && i + 1 < argc) g_threads = std::atoi(argv[++i]);
            else if (a == "--dump") g_dump = true;
            else if (a == "--dense") g_dense = true;
            else if (a == "--layers" && i + 1 < argc) g_layers = std::atoll(argv[++i]);
            else if (a == "--ggml-acts") g_ggml_acts = true;
            else throw std::runtime_error("unknown argument " + a);
        }
        const GgufModel gm = GgufModel::open(argv[1]);
        Model md{gm, {}, {}};
        std::string err;
        if (!strata::read_model_spec(gm.meta(), md.s, err)) throw std::runtime_error(err);
        if (md.s.arch != strata::ModelArch::MiniMaxM3) throw std::runtime_error("not a minimax-m3 model");
        {
            std::ifstream f(argv[2]);
            std::stringstream ss;
            ss << f.rdbuf();
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (tok.find_first_of("0123456789") != std::string::npos) md.ids.push_back((int32_t) std::atol(tok.c_str()));
        }
        md.T = (int64_t) md.ids.size();
        md.N = md.s.n_embd;
        size_t sh = 0;
        md.n_vocab = (int64_t) tensor(gm, "token_embd.weight", sh).shape[1];
        if (md.T == 0) throw std::runtime_error("no token ids");
        if (from < 0) from = md.T - 1;
        if (from >= md.T) throw std::runtime_error("--from is past the sequence");
        const ModelSpec& s = md.s;
        std::fprintf(stderr, "mm_host: %lld tokens, %lld layers (%lld dense), width %lld, %lld heads of %lld (%lld K/V), "
                             "%lld experts (%lld used), blocks of %lld, top %lld, %lld local, rope base %g, rot %lld, "
                             "%d threads\n", (long long) md.T, (long long) s.n_layer, (long long) s.n_dense_lead,
                     (long long) md.N, (long long) s.n_head, (long long) s.key_length, (long long) s.n_head_kv[0],
                     (long long) s.n_expert, (long long) s.n_expert_used, (long long) s.idx_block_size,
                     (long long) s.idx_top_k, (long long) s.idx_local_blocks, s.rope_base, (long long) s.rope_dims,
                     g_threads);
        const std::vector<float> logits = md.forward(from);
        std::ofstream o(argv[3], std::ios::binary);
        o.write((const char*) logits.data(), (std::streamsize) (logits.size() * 4));
        const int64_t R = md.T - from;
        for (int64_t r = 0; r < R; ++r) {
            const float* v = logits.data() + r * md.n_vocab;
            const int64_t best = std::max_element(v, v + md.n_vocab) - v;
            std::fprintf(stderr, "pos %lld: argmax %lld (%.4f)\n", (long long) (from + r), (long long) best, v[best]);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mm_host: %s\n", e.what());
        return 1;
    }
}
