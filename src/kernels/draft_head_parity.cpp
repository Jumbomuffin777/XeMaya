// SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/draft_head_parity.cpp - the draft head's coarse pass (draft_head.hpp) against an FP64 host reference
// (synthetic, no model): the 2-bit rows' products and the largest of them.
//
//   draft_head_parity [--selftest]
//   draft_head_parity --bench ROWS N_IN   the time a call of each part over ROWS rows of N_IN values (no check)
#include "strata/kernels/draft_head.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
namespace par = strata::parity;

struct Head {
    int64_t rows = 0;
    int n_in = 0;
    std::vector<uint8_t> packed;
    std::vector<float> x;
};

Head make_head(int64_t rows, int n_in, uint32_t seed) {
    Head h;
    h.rows = rows;
    h.n_in = n_in;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const std::size_t rb = k::int2_row_bytes(n_in);
    h.packed.resize((std::size_t) rows * rb);
    std::vector<float> f((std::size_t) n_in);
    for (int64_t r = 0; r < rows; ++r) {
        for (auto& v : f) v = nd(rng) * 0.05f;
        k::int2_pack_rows(f.data(), 1, n_in, h.packed.data() + (std::size_t) r * rb);
    }
    h.x.resize((std::size_t) n_in);
    for (auto& v : h.x) v = nd(rng) * 3.0f;
    return h;
}

// the packed row's product in FP64, from the codes and the groups' minimum and step as the device reads them
double ref_row(const Head& h, int64_t r) {
    const std::size_t rb = k::int2_row_bytes(h.n_in);
    const uint8_t* row = h.packed.data() + (std::size_t) r * rb;
    double acc = 0.0;
    for (int g = 0; g < h.n_in / k::kInt2Group; ++g) {
        uint16_t mb, db;
        std::memcpy(&mb, row + h.n_in / 4 + (std::size_t) g * 4, 2);
        std::memcpy(&db, row + h.n_in / 4 + (std::size_t) g * 4 + 2, 2);
        const auto half = [](uint16_t b) {
            const int e = (b >> 10) & 31, m = b & 1023;
            const double v = e == 0 ? std::ldexp((double) m, -24) : std::ldexp((double) (m | 1024), e - 25);
            return (b & 0x8000) ? -v : v;
        };
        const double m = half(mb), d = half(db);
        for (int i = 0; i < k::kInt2Group; ++i) {
            const int at = g * k::kInt2Group + i;
            const int c = (row[at / 4] >> (2 * (at % 4))) & 3;
            acc += (m + d * c) * (double) h.x[(std::size_t) at];
        }
    }
    return acc;
}

// the bit-plane form: the device's products against FP64 over the activations rounded as int2_x_planes rounds them
// (tight), and against the unrounded products (the rounding's own cost, a bound on it)
int check_planes(int64_t rows, int n_in) {
    par::Device dev;
    const Head h = make_head(rows, n_in, 31u + (uint32_t) rows);
    std::vector<uint8_t> planes = h.packed;
    k::int2_rows_to_planes(planes.data(), rows, n_in);
    const uint8_t* d_w = dev.upload(planes);
    const float* d_x = dev.upload(h.x);
    void* d_xp = dev.alloc<uint8_t>(k::int2_x_planes_bytes(n_in));
    float* d_y = dev.alloc<float>((std::size_t) rows + 16);
    k::int2_x_planes(d_x, d_xp, n_in, par::stream());
    k::int2_planes_gemv(d_w, d_xp, d_y, n_in, rows, par::stream());
    std::vector<float> y((std::size_t) rows);
    par::copy_out(y.data(), d_y, y.size());
    Head hq = h;   // the activations as the device rounds them
    const float top = (float) ((1 << k::kInt2XBits) - 1);
    for (int sp = 0; sp < n_in / 32; ++sp) {
        float* xs = hq.x.data() + (std::size_t) sp * 32;
        const auto [lo, hi] = std::minmax_element(xs, xs + 32);
        const float l = *lo, step = (*hi - l) / top;
        for (int i = 0; i < 32; ++i) {
            const float u = step > 0.0f ? std::clamp(std::rint((xs[i] - l) / step), 0.0f, top) : 0.0f;
            xs[i] = (float) ((double) l + (double) step * (double) u);
        }
    }
    double scale = 0.0, tight = 0.0, loose = 0.0;
    const std::size_t rb = k::int2_row_bytes(n_in);
    for (int64_t r = 0; r < rows; ++r) {
        const double exact = ref_row(h, r);
        // m * (the unrounded sum - the rounded sum) is what the device adds to the rounded product
        double fix = 0.0;
        const uint8_t* row = h.packed.data() + (std::size_t) r * rb;
        for (int g = 0; g < n_in / k::kInt2Group; ++g) {
            uint16_t mb;
            std::memcpy(&mb, row + n_in / 4 + (std::size_t) g * 4, 2);
            const int e = (mb >> 10) & 31, mm = mb & 1023;
            double m = e == 0 ? std::ldexp((double) mm, -24) : std::ldexp((double) (mm | 1024), e - 25);
            if (mb & 0x8000) m = -m;
            for (int i = 0; i < k::kInt2Group; ++i) {
                const std::size_t at = (std::size_t) g * k::kInt2Group + (std::size_t) i;
                fix += m * ((double) h.x[at] - (double) hq.x[at]);
            }
        }
        const double rounded = ref_row(hq, r) + fix;
        scale = std::max(scale, std::fabs(exact));
        tight = std::max(tight, std::fabs(rounded - (double) y[(std::size_t) r]));
        loose = std::max(loose, std::fabs(exact - (double) y[(std::size_t) r]));
    }
    // the rounding's bound is counted from 32 rows: fewer products give no scale to hold it against
    const int fails = (tight <= 2e-5 * scale ? 0 : 1) + (rows < 32 || loose <= 5e-2 * scale ? 0 : 1);
    std::printf("%s %lld rows of %d in bit planes: within %.1e of the scale with the activations rounded alike, %.1e "
                "of the unrounded products\n", fails ? "FAIL" : "ok  ", (long long) rows, n_in,
                scale > 0 ? tight / scale : 0.0, scale > 0 ? loose / scale : 0.0);
    return fails;
}

int check(int64_t rows, int n_in, int kk) {
    par::Device dev;
    const Head h = make_head(rows, n_in, 7u + (uint32_t) rows);
    const uint8_t* d_w = dev.upload(h.packed);
    const float* d_x = dev.upload(h.x);
    float* d_y = dev.alloc<float>((std::size_t) rows + 16);
    int32_t* d_ids = dev.alloc<int32_t>((std::size_t) kk);
    void* scratch = dev.alloc<uint8_t>(k::topk_scratch_bytes(rows, kk));
    k::int2_gemv(d_w, d_x, d_y, n_in, rows, par::stream());
    k::topk(d_y, rows, kk, d_ids, scratch, par::stream());
    std::vector<float> y((std::size_t) rows);
    std::vector<int32_t> ids((std::size_t) kk);
    par::copy_out(y.data(), d_y, y.size());
    par::copy_out(ids.data(), d_ids, ids.size());
    double scale = 0.0, worst = 0.0;
    std::vector<double> ref((std::size_t) rows);
    for (int64_t r = 0; r < rows; ++r) {
        ref[(std::size_t) r] = ref_row(h, r);
        scale = std::max(scale, std::fabs(ref[(std::size_t) r]));
        worst = std::max(worst, std::fabs(ref[(std::size_t) r] - (double) y[(std::size_t) r]));
    }
    int fails = worst <= 2e-5 * scale ? 0 : 1;
    // the chosen rows are the largest of the device's own products
    std::vector<float> sorted = y;
    std::sort(sorted.begin(), sorted.end(), std::greater<float>());
    const int want = (int) std::min<int64_t>(kk, rows);
    const float bound = sorted[(std::size_t) want - 1];
    int got = 0;
    for (const int32_t id : ids)
        if (id >= 0 && id < rows && y[(std::size_t) id] >= bound) ++got;
    if (got < want) ++fails;
    std::printf("%s %lld rows of %d, top %d: products within %.1e of the scale, %d of %d chosen\n",
                fails ? "FAIL" : "ok  ", (long long) rows, n_in, kk, scale > 0 ? worst / scale : 0.0, got, want);
    return fails;
}

int bench(int64_t rows, int n_in) {
    par::Device dev;
    const Head h = make_head(rows, n_in, 99);
    const int kk = 16;
    const uint8_t* d_w = dev.upload(h.packed);
    const float* d_x = dev.upload(h.x);
    float* d_y = dev.alloc<float>((std::size_t) rows + 16);
    int32_t* d_ids = dev.alloc<int32_t>((std::size_t) kk);
    float* d_vals = dev.alloc<float>((std::size_t) kk);
    const int32_t* d_pos = dev.upload(std::vector<int32_t>{0});
    void* scratch = dev.alloc<uint8_t>(k::topk_scratch_bytes(rows, kk));
    const auto timed = [&](auto f) {
        for (int i = 0; i < 50; ++i) f();
        par::sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 300; ++i) f();
        par::sync();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 300;
    };
    void* s = par::stream();
    std::printf("draft_head --bench: %lld rows of %d (%.1f MiB at 2 bits)\n", (long long) rows, n_in,
                (double) h.packed.size() / 1048576.0);
    std::printf("  the 2-bit products:        %.1f us a call\n", timed([&] { k::int2_gemv(d_w, d_x, d_y, n_in, rows, s); }));
    std::printf("  the top %d:                %.1f us a call\n", kk, timed([&] { k::topk(d_y, rows, kk, d_ids, scratch, s); }));
    std::printf("  pad (not taken), scatter:  %.1f us a call\n", timed([&] {
        k::int2_pad_from(d_pos, 49152, d_y, rows, s);
        k::scatter_values(d_vals, d_ids, kk, d_y, s);
    }));
    std::printf("  all of them:               %.1f us a call\n", timed([&] {
        k::int2_gemv(d_w, d_x, d_y, n_in, rows, s);
        k::topk(d_y, rows, kk, d_ids, scratch, s);
        k::int2_pad_from(d_pos, 49152, d_y, rows, s);
        k::scatter_values(d_vals, d_ids, kk, d_y, s);
    }));
    {
        std::vector<uint8_t> planes = h.packed;
        k::int2_rows_to_planes(planes.data(), rows, n_in);
        const uint8_t* d_wp = dev.upload(planes);
        void* d_xp = dev.alloc<uint8_t>(k::int2_x_planes_bytes(n_in));
        std::printf("  the activations' planes:   %.1f us a call\n", timed([&] { k::int2_x_planes(d_x, d_xp, n_in, s); }));
        std::printf("  the products in planes:    %.1f us a call\n", timed([&] {
            k::int2_planes_gemv(d_wp, d_xp, d_y, n_in, rows, s);
        }));
        std::printf("  planes, top %d, scatter:   %.1f us a call\n", kk, timed([&] {
            k::int2_x_planes(d_x, d_xp, n_in, s);
            k::int2_planes_gemv(d_wp, d_xp, d_y, n_in, rows, s);
            k::topk(d_y, rows, kk, d_ids, scratch, s);
            k::int2_pad_from(d_pos, 49152, d_y, rows, s);
            k::scatter_values(d_vals, d_ids, kk, d_y, s);
        }));
    }
    // the head's own rows (Q6_K in the engine's row order where the shape allows it; random bytes: the time only)
    if (!k::native_q6_k_rows_ok(n_in)) return 0;
    const std::size_t rb = (std::size_t) n_in / 256 * 210;
    std::vector<uint8_t> head((std::size_t) rows * rb);
    std::mt19937 rng(5);
    for (auto& b : head) b = (uint8_t) (rng() & 31);
    const uint8_t* d_head = dev.upload(head);
    uint8_t* d_cand = dev.alloc<uint8_t>((std::size_t) kk * rb);
    void* d_xq = dev.alloc<uint8_t>(k::native_q8_1_bytes(n_in, 1));
    std::vector<int32_t> pick((std::size_t) kk);
    for (int i = 0; i < kk; ++i) pick[(std::size_t) i] = (int32_t) ((int64_t) i * (rows - 1) / kk);
    par::copy_in(d_ids, pick.data(), pick.size());
    std::printf("  quantize the activations:  %.1f us a call\n", timed([&] { k::native_quantize_q8_1(d_x, d_xq, n_in, 1, s); }));
    std::printf("  gather the %d rows:        %.1f us a call\n", kk, timed([&] {
        k::gather_rows(d_head, (int64_t) rb, d_ids, kk, d_cand, s);
    }));
    std::printf("  their own products:        %.1f us a call\n", timed([&] {
        k::native_mmvq(k::kNativeQ6KRows, d_cand, d_xq, d_vals, n_in, kk, 1, s);
    }));
    std::printf("  the whole head's products: %.1f us a call\n", timed([&] {
        k::native_mmvq(k::kNativeQ6KRows, d_head, d_xq, d_y, n_in, (int) rows, 1, s);
    }));
    std::printf("  the draft head as a step:  %.1f us a call\n", timed([&] {
        k::native_quantize_q8_1(d_x, d_xq, n_in, 1, s);
        k::int2_gemv(d_w, d_x, d_y, n_in, rows, s);
        k::topk(d_y, rows, kk, d_ids, scratch, s);
        k::gather_rows(d_head, (int64_t) rb, d_ids, kk, d_cand, s);
        k::native_mmvq(k::kNativeQ6KRows, d_cand, d_xq, d_vals, n_in, kk, 1, s);
        k::int2_pad_from(d_pos, 49152, d_y, rows, s);
        k::scatter_values(d_vals, d_ids, kk, d_y, s);
    }));
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 3 && std::strcmp(argv[1], "--bench") == 0)
        return bench(std::strtoll(argv[2], nullptr, 10), (int) std::strtol(argv[3], nullptr, 10));
    int fails = 0;
    fails += check(1, 128, 1);
    fails += check(37, 256, 16);
    fails += check(4097, 2560, 16);
    fails += check(40525, 2560, 16);
    fails += check(9000, 1024, 64);
    fails += check_planes(1, 128);
    fails += check_planes(37, 256);
    fails += check_planes(129, 384);
    fails += check_planes(4097, 2560);
    fails += check_planes(40525, 2560);
    std::printf("draft_head_parity: %d failure%s\n", fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
