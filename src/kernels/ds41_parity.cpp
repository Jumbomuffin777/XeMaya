// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/ds41_parity.cpp - DeepSeek-V4.1-Flash's kernels (ds41.hpp) against host references written from
// llama.cpp master 8642541b2's graph (src/models/deepseek4.cpp, deepseek41.cpp, build_moe_ffn) and ggml's CPU
// operators (MIT License, Copyright (c) 2023-2026 The ggml authors).  The references are in double where ggml sums in
// float, so the tolerances are those of float rounding; the selections (router, top-k, blocks) must agree exactly.
//
// Runs on any SYCL device the engine accepts; on a host whose GPUs are leased:
//   ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 ds41_parity
#include "strata/kernels/ds41.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

namespace K = strata::kernels;
using strata::parity::copy_out;

constexpr int HC = 4, MIX = 24;
std::mt19937& rng() {
    static std::mt19937 r(4141);   // fixed: a failure repeats
    return r;
}
std::vector<float> randn(size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng());
    return v;
}
double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

int compare(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    double worst = 0;
    long long bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double err = std::fabs((double) got[i] - want[i]) / std::max(1.0, std::fabs(want[i]));
        worst = std::max(worst, err);
        if (!(err <= tol)) ++bad;
    }
    std::printf("  %-40s worst %.2e  %s\n", what, worst, bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}
int exact(const char* what, long long wrong) {
    std::printf("  %-40s %lld wrong  %s\n", what, wrong, wrong ? "*** FAIL ***" : "ok");
    return wrong ? 1 : 0;
}

// ---- router
int test_router() {
    strata::parity::Device dev;
    const int T = 5, E = 384, k = 6;
    auto logits = randn((size_t) T * E, 2.0f);
    auto bias = randn(E, 0.05f);
    for (int e = 100; e < 112; ++e) logits[(size_t) 2 * E + e] = 3.0f, bias[e] = 0.0f;   // a 12-way tie in token 2
    std::vector<int32_t> ids((size_t) T * k);
    std::vector<float> w((size_t) T * k);
    int32_t* di = dev.alloc<int32_t>(ids.size());
    float* dw = dev.alloc<float>(w.size());
    K::ds41_router(dev.upload(logits), dev.upload(bias), T, E, k, true, 1.5f, di, dw, nullptr);
    copy_out(ids.data(), di, ids.size());
    copy_out(w.data(), dw, w.size());
    long long wrong = 0;
    std::vector<double> rw((size_t) T * k);
    for (int t = 0; t < T; ++t) {
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::vector<double> p(E), s(E);
        for (int e = 0; e < E; ++e) {
            const float l = logits[(size_t) t * E + e];
            p[e] = std::sqrt((double) (l > 20.0f ? l : std::log1p(std::exp((double) l))));
            s[e] = (double) (float) p[e] + bias[e];
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s[a] > s[b]; });
        double sum = 0;
        for (int i = 0; i < k; ++i) sum += p[order[i]];
        for (int i = 0; i < k; ++i) {
            wrong += ids[(size_t) t * k + i] != order[i];
            rw[(size_t) t * k + i] = p[order[i]] / (sum + 1e-20) * 1.5;
        }
    }
    return exact("router ids (384 -> 6, a 12-way tie)", wrong) + compare("router weights (sum + 1e-20, x1.5)", w, rw, 1e-5);
}

// ---- mHC pre weights and collapse
int test_hc() {
    strata::parity::Device dev;
    const int64_t T = 3, N = 512;
    auto x = randn((size_t) T * HC * N), mixes = randn((size_t) T * MIX, 4.0f);
    const float scale[3] = {0.7f, 1.3f, 0.9f};
    auto base = randn(MIX, 0.5f);
    std::vector<float> pre((size_t) T * HC), y((size_t) T * N);
    float* dpre = dev.alloc<float>(pre.size());
    float* dy = dev.alloc<float>(y.size());
    float* dx = dev.upload(x);
    K::ds41_hc_pre_weights(dx, dev.upload(mixes), dev.upload(std::vector<float>(scale, scale + 3)), dev.upload(base),
                           1e-20f, 1e-6f, dpre, N, T, nullptr);
    copy_out(pre.data(), dpre, pre.size());
    std::vector<double> rpre(pre.size());
    for (int64_t t = 0; t < T; ++t) {
        double ss = 0;
        for (int64_t i = 0; i < HC * N; ++i) ss += (double) x[(size_t) (t * HC * N + i)] * x[(size_t) (t * HC * N + i)];
        const double inv = 1.0 / std::sqrt(ss / (double) (HC * N) + 1e-20);
        for (int s = 0; s < HC; ++s) rpre[(size_t) (t * HC + s)] = sigm(mixes[(size_t) (t * MIX + s)] * inv * scale[0] + base[s]) + 1e-6;
    }
    int bad = compare("hc pre weights", pre, rpre, 1e-6);
    // collapse with OTHER weights (the single-pass mHC applies the previous sub-layer's)
    auto other = randn((size_t) T * HC);
    K::ds41_hc_collapse(dx, dev.upload(other), dy, N, T, nullptr);
    copy_out(y.data(), dy, y.size());
    std::vector<double> ry(y.size());
    for (int64_t t = 0; t < T; ++t)
        for (int64_t d = 0; d < N; ++d) {
            double a = 0;
            for (int s = 0; s < HC; ++s) a += (double) other[(size_t) (t * HC + s)] * x[(size_t) ((t * HC + s) * N + d)];
            ry[(size_t) (t * N + d)] = a;
        }
    return bad + compare("hc collapse (given pre)", y, ry, 1e-5);
}

// ---- compressor
int test_compress() {
    strata::parity::Device dev;
    const int64_t B = 4, D = 512;
    int bad = 0;
    for (int ratio : {2, 1}) {
        auto v = randn((size_t) B * ratio * D), s = randn((size_t) B * ratio * D, 2.0f), w = randn(D, 0.3f);
        std::vector<float> out((size_t) B * D);
        float* dout = dev.alloc<float>(out.size());
        K::ds41_compress(dev.upload(v), ratio > 1 ? dev.upload(s) : nullptr, dev.upload(w), 1e-20f, dout, B, ratio, D,
                         nullptr);
        copy_out(out.data(), dout, out.size());
        std::vector<double> r(out.size());
        for (int64_t b = 0; b < B; ++b) {
            std::vector<double> c(D);
            double ss = 0;
            for (int64_t d = 0; d < D; ++d) {
                if (ratio == 1) c[d] = v[(size_t) (b * D + d)];
                else {
                    double m = -HUGE_VAL, sum = 0, acc = 0;
                    for (int j = 0; j < ratio; ++j) m = std::max(m, (double) s[(size_t) ((b * ratio + j) * D + d)]);
                    for (int j = 0; j < ratio; ++j) {
                        const double e = std::exp(s[(size_t) ((b * ratio + j) * D + d)] - m);
                        sum += e;
                        acc += e * v[(size_t) ((b * ratio + j) * D + d)];
                    }
                    c[d] = acc / sum;
                }
                ss += c[d] * c[d];
            }
            const double inv = 1.0 / std::sqrt(ss / (double) D + 1e-20);
            for (int64_t d = 0; d < D; ++d) r[(size_t) (b * D + d)] = c[d] * inv * w[d];
        }
        bad += compare(ratio == 2 ? "compress (ratio 2, softmax gate, RMSNorm)" : "compress (ratio 1, no gate)", out, r, 2e-5);
    }
    return bad;
}

// ---- Engram
int test_engram() {
    strata::parity::Device dev;
    const int64_t T = 3, N = 512;
    auto x = randn((size_t) T * HC * N), kv = randn((size_t) T * (HC + 1) * N), gw = randn((size_t) HC * N, 0.5f);
    std::vector<float> keep = {1.0f, 0.0f, 1.0f};
    for (int64_t d = 0; d < N; ++d) kv[(size_t) d] = x[(size_t) d] * -0.001f;   // a dot near 0 in token 0, stream 0
    float* dx = dev.upload(x);
    K::ds41_engram_gate(dx, dev.upload(kv), dev.upload(gw), dev.upload(keep), 1e-20f, N, T, nullptr);
    std::vector<float> out(x.size());
    copy_out(out.data(), dx, out.size());
    std::vector<double> r(out.size());
    for (int64_t t = 0; t < T; ++t)
        for (int s = 0; s < HC; ++s) {
            const float* xs = &x[(size_t) ((t * HC + s) * N)];
            const float* ks = &kv[(size_t) (t * (HC + 1) * N + s * N)];
            const float* val = &kv[(size_t) (t * (HC + 1) * N + HC * N)];
            double xx = 0, kk = 0, g = 0;
            for (int64_t d = 0; d < N; ++d) xx += (double) xs[d] * xs[d], kk += (double) ks[d] * ks[d];
            const double ix = 1.0 / std::sqrt(xx / N + 1e-20), ik = 1.0 / std::sqrt(kk / N + 1e-20);
            for (int64_t d = 0; d < N; ++d) g += xs[d] * ix * ks[d] * ik * gw[(size_t) (s * N + d)];
            const double dot = g / std::sqrt((double) N);
            const double mag = std::sqrt(std::max(std::fabs(dot), 1e-6));
            const double gate = sigm(dot < 0 ? -mag : mag) * keep[(size_t) t];
            for (int64_t d = 0; d < N; ++d) r[(size_t) ((t * HC + s) * N + d)] = xs[d] + val[d] * gate;
        }
    return compare("engram gate (keep 1, 0, 1)", out, r, 1e-5);
}

int test_engram_hash() {
    // a small module: n-gram 4, 2 heads per n-gram -> 6 columns
    const int ngram = 4, hpn = 2, cols = (ngram - 1) * hpn;
    std::vector<int32_t> map(50);
    for (int i = 0; i < 50; ++i) map[(size_t) i] = (i * 7) % 31;
    const int32_t pad_id = 2;
    const uint64_t mult[4] = {76632096046245ull, 4839876093313ull, 35959672319349ull, 73987337458391ull};
    const uint64_t primes[6] = {97, 101, 103, 107, 109, 113};
    const uint64_t offs[6] = {0, 97, 198, 301, 408, 517};
    long long wrong = 0;
    // reference written directly from llama_model_deepseek41::engram_hash
    const auto ref = [&](int32_t tok, const int32_t* prev, int64_t pos, int32_t* out) {
        const auto cid = [&](int32_t t) -> int64_t { return t < 0 || t >= 50 ? -1 : map[(size_t) t]; };
        uint64_t toks[4];
        int64_t c = cid(tok);
        bool blocked = c == -1;
        toks[0] = (uint64_t) (blocked ? map[pad_id] : c);
        for (int s = 1; s < ngram; ++s) {
            c = pos - s < 0 ? -1 : cid(prev[ngram - 1 - s]);
            blocked = blocked || c == -1;
            toks[s] = (uint64_t) (blocked ? map[pad_id] : c);
        }
        uint64_t roll = toks[0] * mult[0];
        for (int o = 1; o < ngram; ++o) {
            roll ^= toks[o] * mult[o];
            for (int h = 0; h < hpn; ++h) out[(o - 1) * hpn + h] = (int32_t) (roll % primes[(o - 1) * hpn + h] + offs[(o - 1) * hpn + h]);
        }
    };
    struct Case { int32_t tok; int32_t prev[3]; int64_t pos; };
    const Case cases[] = {{5, {1, 2, 3}, 10}, {5, {1, 2, 3}, 1}, {5, {1, 77, 3}, 10}, {99, {1, 2, 3}, 10}, {0, {0, 0, 0}, 0}};
    for (const Case& c : cases) {
        int32_t got[cols], want[cols];
        K::ds41_engram_hash(c.tok, c.prev, c.pos, map, pad_id, ngram, hpn, mult, primes, offs, got);
        ref(c.tok, c.prev, c.pos, want);
        for (int i = 0; i < cols; ++i) {
            wrong += got[i] != want[i];
            wrong += got[i] < (int32_t) offs[i] || got[i] >= (int32_t) (offs[i] + primes[i]);   // inside its head's rows
        }
    }
    return exact("engram hash (start, dead token, pad)", wrong);
}

// ---- RoPE: a float reference that follows ggml's rope_yarn and its cache init step by step
void ref_rope(std::vector<float>& x, const std::vector<int32_t>& pos, const K::Ds41Rope& r, bool inverse, int64_t rows,
              int64_t dim) {
    float corr[2] = {0, 0};
    if (r.ext_factor != 0.0f) {
        const auto cd = [&](float n_rot) {
            return (float) r.n_rot * logf((float) r.n_ctx_orig / (n_rot * 2 * (float) M_PI)) / (2 * logf(r.freq_base));
        };
        corr[0] = std::max(0.0f, floorf(cd(r.beta_fast)));
        corr[1] = std::min((float) (r.n_rot - 1), ceilf(cd(r.beta_slow)));
    }
    const float theta_scale = powf(r.freq_base, -2.0f / r.n_rot);
    for (size_t t = 0; t < pos.size(); ++t)
        for (int64_t row = 0; row < rows; ++row) {
            float theta_extrap = (float) pos[t];
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
                float* v = &x[(size_t) (((int64_t) t * rows + row) * dim + r.offset + i0)];
                const float x0 = v[0], x1 = v[1];
                v[0] = x0 * ct - x1 * st;
                v[1] = x0 * st + x1 * ct;
                theta_extrap *= theta_scale;
            }
        }
}

int test_rope() {
    strata::parity::Device dev;
    const int64_t T = 4, H = 3, D = 512;
    const std::vector<int32_t> pos = {0, 7, 1000, 4095};
    int bad = 0;
    for (int yarn = 0; yarn < 2; ++yarn) {
        K::Ds41Rope r;
        if (yarn) {   // the compressed layers: base 160000, YaRN factor 16 over 65536 positions
            r.freq_base = 160000.0f;
            r.freq_scale = 1.0f / 16.0f;
            r.ext_factor = 1.0f;
            r.n_ctx_orig = 65536;
            r.attn_factor = 1.0f / (1.0f + 0.1f * logf(16.0f));   // dsv4_rope_attn_factor
        }
        auto x = randn((size_t) (T * H * D));
        float* dx = dev.upload(x);
        K::ds41_rope(dx, dev.upload(pos), r, false, H, D, T, nullptr);
        std::vector<float> got(x.size());
        copy_out(got.data(), dx, got.size());
        auto want = x;
        ref_rope(want, pos, r, false, H, D);
        bad += compare(yarn ? "rope (YaRN, base 160000, dims 448..511)" : "rope (base 10000, dims 448..511)", got,
                       std::vector<double>(want.begin(), want.end()), 2e-4);
        K::ds41_rope(dx, dev.upload(pos), r, true, H, D, T, nullptr);
        copy_out(got.data(), dx, got.size());
        bad += compare(yarn ? "rope then back (YaRN) = input" : "rope then back = input", got,
                       std::vector<double>(x.begin(), x.end()), 1e-5);
    }
    return bad;
}

// ---- indexer, blocks, top-k
int test_indexer() {
    strata::parity::Device dev;
    const int T = 3, H = 32, Di = 128, C = 700, stride = 640, off = 512, S = 704, BS = 8;
    auto q = randn((size_t) T * H * Di), w = randn((size_t) T * H, 0.1f), keys = randn((size_t) C * stride);
    const std::vector<int32_t> nv = {700, 333, 1};
    std::vector<float> sc((size_t) T * S);
    float* dsc = dev.alloc<float>(sc.size());
    int32_t* dnv = dev.upload(nv);
    K::ds41_idx_scores(dev.upload(q), dev.upload(w), dev.upload(keys), stride, off, dnv, H, Di, T, S, dsc, nullptr);
    copy_out(sc.data(), dsc, sc.size());
    std::vector<double> r(sc.size());
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < S; ++c) {
            if (c >= nv[(size_t) t]) { r[(size_t) (t * S + c)] = -HUGE_VAL; continue; }
            double acc = 0;
            for (int h = 0; h < H; ++h) {
                double dot = 0;
                for (int d = 0; d < Di; ++d) dot += (double) q[(size_t) ((t * H + h) * Di + d)] * keys[(size_t) (c * stride + off + d)];
                acc += std::max(dot, 0.0) * w[(size_t) (t * H + h)];
            }
            r[(size_t) (t * S + c)] = acc;
        }
    long long inf_wrong = 0;
    std::vector<float> fin;
    std::vector<double> rfin;
    for (size_t i = 0; i < sc.size(); ++i) {
        if (std::isinf(r[i])) inf_wrong += !(std::isinf(sc[i]) && sc[i] < 0);
        else fin.push_back(sc[i]), rfin.push_back(r[i]);
    }
    int bad = compare("indexer scores (32 heads, relu, weighted)", fin, rfin, 1e-4) + exact("indexer: rows past the visible are -inf", inf_wrong);

    // top-k on the kernel's own scores, with ties planted
    for (int c = 10; c < 30; ++c) sc[(size_t) c] = 5.0f;
    strata::parity::copy_in(dsc, sc.data(), sc.size());
    const int k = 512;
    std::vector<int32_t> ids((size_t) T * k), cnt(T);
    int32_t* dids = dev.alloc<int32_t>(ids.size());
    int32_t* dcnt = dev.alloc<int32_t>(cnt.size());
    K::ds41_topk(dsc, dnv, k, T, S, dids, dcnt, nullptr);
    copy_out(ids.data(), dids, ids.size());
    copy_out(cnt.data(), dcnt, cnt.size());
    long long wrong = 0;
    for (int t = 0; t < T; ++t) {
        std::vector<int> order;
        for (int c = 0; c < nv[(size_t) t]; ++c) if (!std::isinf(sc[(size_t) (t * S + c)])) order.push_back(c);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sc[(size_t) (t * S + a)] > sc[(size_t) (t * S + b)]; });
        order.resize(std::min<size_t>(order.size(), (size_t) k));
        std::sort(order.begin(), order.end());
        wrong += cnt[(size_t) t] != (int32_t) order.size();
        for (size_t i = 0; i < order.size() && i < (size_t) cnt[(size_t) t]; ++i) wrong += ids[(size_t) t * k + i] != order[i];
    }
    bad += exact("top-512 (ties to lower index, ascending)", wrong);

    // candidate blocks: block max with the newest block pinned, top 16 blocks, then keep only those
    const int NB = (S + BS - 1) / BS, KB = 16;
    std::vector<float> bm((size_t) T * NB);
    float* dbm = dev.alloc<float>(bm.size());
    K::ds41_block_max(dsc, dnv, BS, T, S, NB, dbm, nullptr);
    copy_out(bm.data(), dbm, bm.size());
    long long bwrong = 0;
    for (int t = 0; t < T; ++t)
        for (int b = 0; b < NB; ++b) {
            float m = -INFINITY;
            for (int c = b * BS; c < std::min(nv[(size_t) t], (b + 1) * BS); ++c) m = std::max(m, sc[(size_t) (t * S + c)]);
            if (nv[(size_t) t] > 0 && b == (nv[(size_t) t] - 1) / BS) m += 1e30f;
            bwrong += !(bm[(size_t) (t * NB + b)] == m || (std::isinf(m) && std::isinf(bm[(size_t) (t * NB + b)])));
        }
    bad += exact("candidate block max (+1e30 newest block)", bwrong);
    std::vector<int32_t> nb_all(T, NB);
    int32_t* dbids = dev.alloc<int32_t>((size_t) T * KB);
    int32_t* dbcnt = dev.alloc<int32_t>(T);
    K::ds41_topk(dbm, dev.upload(nb_all), KB, T, NB, dbids, dbcnt, nullptr);
    std::vector<int32_t> bids((size_t) T * KB), bcnt(T);
    copy_out(bids.data(), dbids, bids.size());
    copy_out(bcnt.data(), dbcnt, bcnt.size());
    K::ds41_keep_blocks(dsc, dnv, dbids, dbcnt, KB, BS, T, S, nullptr);
    std::vector<float> kept(sc.size());
    copy_out(kept.data(), dsc, kept.size());
    long long kwrong = 0;
    for (int t = 0; t < T; ++t) {
        const int newest = (nv[(size_t) t] - 1) / BS;
        bool has_newest = false;
        for (int i = 0; i < bcnt[(size_t) t]; ++i) has_newest |= bids[(size_t) (t * KB + i)] == newest;
        kwrong += !has_newest;
        for (int c = 0; c < nv[(size_t) t]; ++c) {
            const bool in = std::find(&bids[(size_t) (t * KB)], &bids[(size_t) (t * KB)] + bcnt[(size_t) t], c / BS) !=
                            &bids[(size_t) (t * KB)] + bcnt[(size_t) t];
            const float want = in ? sc[(size_t) (t * S + c)] : -INFINITY;
            kwrong += !(kept[(size_t) (t * S + c)] == want || (std::isinf(want) && std::isinf(kept[(size_t) (t * S + c)])));
        }
    }
    return bad + exact("keep the top 16 blocks (newest always)", kwrong);
}

// ---- MQA with sinks over the window ring and the selected compressed rows
int test_sink_attn() {
    strata::parity::Device dev;
    const int T = 3, H = 8, D = 512, W = 128, R = 160, NC = 300, STRIDE = 640, CAP = 64;
    auto q = randn((size_t) T * H * D, 0.05f), raw = randn((size_t) R * D), comp = randn((size_t) NC * STRIDE);
    auto sinks = randn(H, 1.0f);
    const std::vector<int32_t> pos = {0, 50, 1000};
    std::vector<int32_t> ids((size_t) T * CAP), cnt = {0, 10, CAP};
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < CAP; ++i) ids[(size_t) (t * CAP + i)] = (i * 37 + t) % NC;
    const float scale = 1.0f / std::sqrt((float) D);
    std::vector<float> out((size_t) T * H * D);
    float* dout = dev.alloc<float>(out.size());
    K::ds41_sink_attn(dev.upload(q), dev.upload(raw), R, dev.upload(pos), W, dev.upload(comp), STRIDE, dev.upload(ids),
                      dev.upload(cnt), CAP, dev.upload(sinks), scale, H, D, T, dout, nullptr);
    copy_out(out.data(), dout, out.size());
    std::vector<double> r(out.size());
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            std::vector<const float*> keys;
            for (int p = std::max(0, pos[(size_t) t] - W + 1); p <= pos[(size_t) t]; ++p) keys.push_back(&raw[(size_t) ((p % R) * D)]);
            for (int i = 0; i < cnt[(size_t) t]; ++i) keys.push_back(&comp[(size_t) (ids[(size_t) (t * CAP + i)] * STRIDE)]);
            std::vector<double> l(keys.size());
            double m = sinks[(size_t) h];
            for (size_t j = 0; j < keys.size(); ++j) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += (double) q[(size_t) ((t * H + h) * D + d)] * keys[j][d];
                l[j] = dot * scale;
                m = std::max(m, l[j]);
            }
            double sum = std::exp(sinks[(size_t) h] - m);
            for (auto& v : l) sum += (v = std::exp(v - m));
            for (int d = 0; d < D; ++d) {
                double acc = 0;
                for (size_t j = 0; j < keys.size(); ++j) acc += l[j] * keys[j][d];
                r[(size_t) ((t * H + h) * D + d)] = acc / sum;
            }
        }
    return compare("MQA + sinks (window ring, selected rows)", out, r, 1e-5);
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: ds41_parity [--selftest]\n");
            return 2;
        }
    std::printf("ds41_parity: DeepSeek-V4.1-Flash's kernels against host references\n");
    const int bad = test_router() + test_hc() + test_compress() + test_engram() + test_engram_hash() + test_rope() +
                    test_indexer() + test_sink_attn();
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
