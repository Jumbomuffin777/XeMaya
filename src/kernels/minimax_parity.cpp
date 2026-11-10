// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/minimax_parity.cpp - MiniMax-M3's sparse attention kernels (minimax.hpp) and swiglu-oai against host
// references in double precision, written from llama.cpp's graph (src/models/minimax-m3.cpp, master 8642541b2).
//
// Runs on any SYCL device the engine accepts; on a host whose GPUs are leased:
//   ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 minimax_parity
// The matrix-engine attention is skipped where the device has no FP16 matrix engine (the OpenCL CPU device).
#include "strata/kernels/minimax.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/swiglu_clamp.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

namespace K = strata::kernels;
using strata::parity::copy_out;

std::mt19937& rng() {
    static std::mt19937 r(3131);   // fixed: a failure repeats
    return r;
}
std::vector<float> randn(size_t n, float s = 1.0f) {
    std::normal_distribution<float> d(0.0f, s);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng());
    return v;
}
double h16(uint16_t x) { return (double) K::f32_from_f16(x); }

int compare(const char* what, const std::vector<float>& got, const std::vector<double>& want, double tol) {
    double worst = 0;
    long long bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double err = std::fabs((double) got[i] - want[i]) / std::max(1.0, std::fabs(want[i]));
        worst = std::max(worst, err);
        if (!(err <= tol)) ++bad;
    }
    std::printf("  %-58s worst %.2e  %s\n", what, worst, bad ? "*** FAIL ***" : "ok");
    return bad ? 1 : 0;
}
int exact(const char* what, long long wrong) {
    std::printf("  %-58s %lld wrong  %s\n", what, wrong, wrong ? "*** FAIL ***" : "ok");
    return wrong ? 1 : 0;
}

// ---- swiglu-oai: the dense kernel, and the pair swiglu_clamp hands the experts' silu(gate) * up
int test_swiglu() {
    strata::parity::Device dev;
    const size_t n = 4096;
    auto g = randn(n, 6.0f), u = randn(n, 6.0f);
    std::vector<float> out(n), pair(n);
    float* d = dev.alloc<float>(n);
    K::glm5_swiglu_clamp(dev.upload(g), dev.upload(u), d, (int64_t) n, -K::kSwigluOaiLimit, nullptr);
    copy_out(out.data(), d, n);
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) {   // ggml_compute_forward_swiglu_oai_f32, alpha 1.702, limit 7
        const double x = std::min((double) g[i], 7.0), y = std::clamp((double) u[i], -7.0, 7.0);
        r[i] = x / (1.0 + std::exp(1.702 * -x)) * (y + 1.0);
        float gg = g[i], uu = u[i];
        K::swiglu_clamp(gg, uu, -K::kSwigluOaiLimit);
        pair[i] = K::swiglu_host(gg, uu, 0.0f);
    }
    return compare("swiglu-oai (the dense FFN's kernel)", out, r, 2e-6) +
           compare("swiglu-oai (the experts' pair)", pair, r, 2e-6);
}

struct Fixture {
    K::QsaShapes s;
    int T = 0, blk = 0, top_k = 0, local = 0, ID = 0;
    int64_t cells = 0, max_blocks = 0;
    std::vector<int32_t> pos, step, table;
    std::vector<float> q, iq;
    std::vector<uint16_t> kp, vp, idx;
    // the cell's row in a pool
    size_t row(int64_t c, int hk) const {
        return (size_t) ((((int64_t) table[(size_t) (c / s.page_size)] * s.n_head_kv + hk) * s.page_size +
                          c % s.page_size) * s.head_dim);
    }
};

Fixture fixture(int blk, int top_k, int local, const std::vector<int32_t>& pos) {
    Fixture f;
    f.s = K::qsa_real_shapes();
    f.s.n_head = 64; f.s.n_head_kv = 4; f.s.head_dim = 128; f.s.n_rot = 64;
    f.T = (int) pos.size(); f.blk = blk; f.top_k = top_k; f.local = local; f.ID = 128;
    f.pos = pos;
    const int64_t top = *std::max_element(pos.begin(), pos.end()) + 1;
    const int64_t pages = (top + f.s.page_size - 1) / f.s.page_size;
    f.cells = pages * f.s.page_size;
    f.max_blocks = (f.cells + blk - 1) / blk;
    f.table.resize((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) f.table[(size_t) i] = (int32_t) (pages - 1 - i);   // not the identity
    f.step.assign((size_t) f.T * K::kStepCount, 0);
    for (int t = 0; t < f.T; ++t) K::qsa_step_fill(&f.step[(size_t) t * K::kStepCount], pos[(size_t) t], f.s);
    f.q = randn((size_t) f.T * f.s.n_head * f.s.head_dim, 2.0f);
    f.iq = randn((size_t) f.T * f.s.n_head_kv * f.ID);
    auto half_rows = [&](size_t n, float sd) {
        auto v = randn(n, sd);
        std::vector<uint16_t> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = K::f16_from_f32(v[i]);
        return h;
    };
    const size_t pool = (size_t) (f.cells * f.s.n_head_kv * f.s.head_dim);
    f.kp = half_rows(pool, 1.0f);
    f.vp = half_rows(pool, 1.0f);
    f.idx = half_rows((size_t) (f.cells * f.ID), 1.0f);
    return f;
}

// llama.cpp's selection for a query at position p and K/V head h (decode form: the scores of the visible positions,
// a block's maximum, + 1e30 on the local blocks, the top K = min(top_k, blocks) of them); the blocks ascending
std::vector<int32_t> ref_select(const Fixture& f, int t, int h) {
    const int64_t p = f.pos[(size_t) t], vis = p / f.blk + 1;
    std::vector<int32_t> out;
    if (vis <= f.top_k) {
        for (int64_t b = 0; b < vis; ++b) out.push_back((int32_t) b);
        return out;
    }
    std::vector<double> bs((size_t) vis, -INFINITY);
    for (int64_t c = 0; c <= p; ++c) {
        double d = 0;
        for (int i = 0; i < f.ID; ++i)
            d += (double) f.iq[(size_t) ((t * f.s.n_head_kv + h) * f.ID + i)] * h16(f.idx[(size_t) (c * f.ID + i)]);
        bs[(size_t) (c / f.blk)] = std::max(bs[(size_t) (c / f.blk)], d);
    }
    for (int l = 0; l < f.local && vis - 1 - l >= 0; ++l) bs[(size_t) (vis - 1 - l)] = 1e30 + bs[(size_t) (vis - 1 - l)];
    std::vector<int32_t> order((size_t) vis);
    for (int64_t b = 0; b < vis; ++b) order[(size_t) b] = (int32_t) b;
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return bs[(size_t) a] > bs[(size_t) b]; });
    out.assign(order.begin(), order.begin() + f.top_k);
    std::sort(out.begin(), out.end());
    return out;
}

// the attention of (t, h) over `blocks` (null: every cell up to its own)
void ref_attn(const Fixture& f, int t, int h, const std::vector<int32_t>* blocks, double* out) {
    const int HD = (int) f.s.head_dim, hk = h / (int) (f.s.n_head / f.s.n_head_kv);
    const int64_t p = f.pos[(size_t) t];
    std::vector<int64_t> cells;
    if (blocks == nullptr)
        for (int64_t c = 0; c <= p; ++c) cells.push_back(c);
    else
        for (int32_t b : *blocks)
            for (int64_t c = (int64_t) b * f.blk; c < (int64_t) (b + 1) * f.blk && c <= p; ++c) cells.push_back(c);
    std::vector<double> l(cells.size());
    double m = -INFINITY;
    const float* q = &f.q[(size_t) ((t * f.s.n_head + h) * HD)];
    for (size_t j = 0; j < cells.size(); ++j) {
        double d = 0;
        const uint16_t* kr = &f.kp[f.row(cells[j], hk)];
        for (int i = 0; i < HD; ++i) d += (double) q[i] * h16(kr[i]);
        l[j] = d / std::sqrt((double) HD);
        m = std::max(m, l[j]);
    }
    double sum = 0;
    for (auto& v : l) sum += (v = std::exp(v - m));
    for (int i = 0; i < HD; ++i) {
        double acc = 0;
        for (size_t j = 0; j < cells.size(); ++j) acc += l[j] * h16(f.vp[f.row(cells[j], hk) + (size_t) i]);
        out[i] = acc / sum;
    }
}

int test_msa(const char* name, int blk, int top_k, int local, const std::vector<int32_t>& pos) {
    strata::parity::Device dev;
    const Fixture f = fixture(blk, top_k, local, pos);
    const int T = f.T, NKV = (int) f.s.n_head_kv, NH = (int) f.s.n_head, HD = (int) f.s.head_dim;
    std::printf(" %s: %d tokens up to position %d, blocks of %d, top %d, %d local\n", name, T,
                *std::max_element(pos.begin(), pos.end()), blk, top_k, local);
    // the cache through mm_idx_append: the keys of every cell, appended in pieces
    uint16_t* dcache = dev.alloc<uint16_t>(f.idx.size());
    {
        std::vector<float> ik(f.idx.size());
        for (size_t i = 0; i < ik.size(); ++i) ik[i] = K::f32_from_f16(f.idx[i]);
        std::vector<int32_t> st((size_t) f.cells * K::kStepCount);
        for (int64_t c = 0; c < f.cells; ++c) K::qsa_step_fill(&st[(size_t) c * K::kStepCount], c, f.s);
        K::mm_idx_append(dcache, dev.upload(st), K::kStepCount, dev.upload(ik), f.ID, (int) f.cells, nullptr);
    }
    std::vector<uint16_t> cache(f.idx.size());
    copy_out(cache.data(), dcache, cache.size());
    long long wrong = 0;
    for (size_t i = 0; i < cache.size(); ++i) wrong += cache[i] != f.idx[i];
    int bad = exact("the indexer's keys in the cache", wrong);

    const int32_t* dstep = dev.upload(f.step);
    int32_t *dsel = dev.alloc<int32_t>((size_t) T * NKV * top_k), *dcnt = dev.alloc<int32_t>((size_t) T * NKV);
    float* dscores = dev.alloc<float>((size_t) (T * NKV * f.max_blocks));
    K::mm_msa_select(dev.upload(f.iq), dcache, dstep, K::kStepCount, T, NKV, f.ID, blk, top_k, local, f.max_blocks,
                     dscores, dsel, dcnt, nullptr);
    std::vector<int32_t> sel((size_t) T * NKV * top_k), cnt((size_t) T * NKV);
    copy_out(sel.data(), dsel, sel.size());
    copy_out(cnt.data(), dcnt, cnt.size());
    std::vector<std::vector<int32_t>> want((size_t) T * NKV);
    wrong = 0;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NKV; ++h) {
            auto& w = want[(size_t) (t * NKV + h)];
            w = ref_select(f, t, h);
            const int32_t* got = &sel[(size_t) ((t * NKV + h) * top_k)];
            if (cnt[(size_t) (t * NKV + h)] != (int32_t) w.size()) { ++wrong; continue; }
            for (size_t i = 0; i < w.size(); ++i) wrong += got[i] != w[i];
        }
    bad += exact("the blocks selected", wrong);

    std::vector<double> r_sparse((size_t) T * NH * HD), r_dense(r_sparse.size());
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < NH; ++h) {
            ref_attn(f, t, h, &want[(size_t) (t * NKV + h / (NH / NKV))], &r_sparse[(size_t) ((t * NH + h) * HD)]);
            ref_attn(f, t, h, nullptr, &r_dense[(size_t) ((t * NH + h) * HD)]);
        }
    const float* dq = dev.upload(f.q);
    const uint16_t *dk = dev.upload(f.kp), *dv = dev.upload(f.vp);
    const int32_t* dtab = dev.upload(f.table);
    float* dout = dev.alloc<float>(r_sparse.size());
    std::vector<float> out(r_sparse.size());
    for (int splits : {1, 16}) {
        float* scratch = dev.alloc<float>((size_t) T * K::mm_msa_attn_scratch_floats(splits, f.s) + 16);
        for (bool sparse : {true, false}) {
            K::mm_msa_attn(dq, dk, dv, dtab, dstep, K::kStepCount, sparse ? dsel : nullptr, sparse ? dcnt : nullptr,
                           top_k, blk, f.s, splits, scratch, dout, T, nullptr);
            copy_out(out.data(), dout, out.size());
            const std::string what = std::string(sparse ? "the selected blocks' attention" : "every cell's attention") +
                                     (splits > 1 ? ", in 16 parts" : "");
            bad += compare(what.c_str(), out, sparse ? r_sparse : r_dense, 2e-5);
        }
    }
    for (bool sparse : {true, false}) {
        if (!K::mm_msa_attn_xmx(dq, dk, dv, dtab, dstep, K::kStepCount, sparse ? dsel : nullptr,
                                sparse ? dcnt : nullptr, top_k, blk, f.s, dout, T, nullptr)) {
            std::printf("  %-58s skipped (no FP16 matrix engine)\n", "the attention on the matrix engines");
            break;
        }
        copy_out(out.data(), dout, out.size());
        bad += compare(sparse ? "the selected blocks' attention on the matrix engines"
                              : "every cell's attention on the matrix engines", out, sparse ? r_sparse : r_dense, 2e-5);
    }
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: minimax_parity [--selftest]\n");
            return 2;
        }
    std::printf("minimax_parity: MiniMax-M3's kernels against host references\n");
    int bad = test_swiglu();
    // the model's shape (blocks of 128, 16 of them, 1 local): tokens that see fewer blocks than they take, exactly
    // as many, and more; a block's first and last cell
    bad += test_msa("the model's blocks", 128, 16, 1, {0, 5, 127, 128, 2047, 2048, 2049, 2500, 3071, 3072, 3333});
    // small blocks: many more blocks than are taken, two local
    bad += test_msa("small blocks", 16, 5, 2, {0, 15, 16, 79, 80, 81, 200, 255, 256, 700, 1023});
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
