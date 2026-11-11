// SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/qsa_split_attn_parity.cpp - the MTP draft layer's attention on the matrix engines
// (`qsa_split_attn_batch`, qsa_prompt_attn.hpp) against the scalar kernel it replaces (`qsa_decode_attn_batch`) and an
// FP64 host reference (synthetic, no model): INT8 K and V, 24/2 heads of 256, a shuffled page table.
//
// Cases: one query over 1, 15, 59, 1,000 and 26,800 cells of a 32,768-cell window, four queries with their own counts
// and windows, and a block that is not resident (page -1); each with one slot, seven and the device's own count.
// Checks, as qsa_prompt_attn_parity's:
//   1. against FP64, the new kernel's error is no larger than 4 times the scalar kernel's (both FP32 math);
//   2. the new and scalar outputs agree to a relative 1e-4 of the output scale.
// Where the device has no kernel for it (no matrix engines: the OpenCL CPU device) the new one is reported as skipped
// and the scalar kernel's distance from FP64 is printed without counting.
//
//   qsa_split_attn_parity [--selftest]
//   ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 qsa_split_attn_parity
//   qsa_split_attn_parity --bench CELLS   one query over CELLS cells: the scalar kernel's time a call and the new
//                                         one's by slot count (no check)
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "parity_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
namespace par = strata::parity;
template <typename T> T* up(par::Device& dev, const std::vector<T>& h) {
    T* d = dev.alloc<T>(h.size() + 64 / sizeof(T));
    par::copy_in(d, h.data(), h.size());
    return d;
}

void set_slots(const char* v) {
    if (v && *v) setenv("XEMAYA_MTP_ATTN_SLOTS", v, 1);
    else unsetenv("XEMAYA_MTP_ATTN_SLOTS");
}

struct Pools {
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs;
    std::vector<int32_t> table;
    k::QsaAttnPools dev;
};

Pools make_pools(par::Device& dev, const k::QsaShapes& s, int64_t cells, std::mt19937& rng, int64_t masked_page) {
    const int64_t HD = s.head_dim, PS = s.page_size, pages = (cells + PS - 1) / PS, rows = pages * s.n_head_kv * PS;
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    Pools p;
    p.kq.resize((size_t) (rows * HD));
    p.vq.resize((size_t) (rows * HD));
    p.ks.resize((size_t) (rows * 4));
    p.vs.resize((size_t) (rows * 4));
    for (auto& x : p.kq) x = (int8_t) code(rng);
    for (auto& x : p.vq) x = (int8_t) code(rng);
    for (auto& x : p.ks) x = k::f16_from_f32(sc(rng));
    for (auto& x : p.vs) x = k::f16_from_f32(sc(rng));
    p.table.resize((size_t) pages);
    for (int64_t i = 0; i < pages; ++i) p.table[(size_t) i] = (int32_t) i;
    std::shuffle(p.table.begin(), p.table.end(), rng);
    if (masked_page >= 0 && masked_page < pages) p.table[(size_t) masked_page] = -1;
    p.dev.k_q = up(dev, p.kq);
    p.dev.v_q = up(dev, p.vq);
    p.dev.k_scale = up(dev, p.ks);
    p.dev.v_scale = up(dev, p.vs);
    p.dev.page_table = up(dev, p.table);
    return p;
}

// `first[i]`, `count[i]`: query i attends to cells first[i] .. first[i] + count[i] - 1 (window_ids' selection)
int run_case(const char* name, int64_t cap, const std::vector<int64_t>& first, const std::vector<int64_t>& count,
             int64_t masked_page) {
    par::Device dev;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size, nq = (int64_t) first.size();
    std::mt19937 rng(4321 + (unsigned) (count[0] + nq));
    std::normal_distribution<float> nd(0.f, 1.f);
    int64_t cells = 0;
    for (int64_t i = 0; i < nq; ++i) cells = std::max(cells, first[(size_t) i] + count[(size_t) i]);
    Pools p = make_pools(dev, s, cells, rng, masked_page);
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t n = count[(size_t) i];
        steps[(size_t) (i * k::kStepCount + k::kStepPos)] = (int32_t) (first[(size_t) i] + n - 1);
        steps[(size_t) (i * k::kStepCount + k::kStepNKv)] = (int32_t) (first[(size_t) i] + n);
        steps[(size_t) (i * k::kStepCount + k::kStepWidth)] = (int32_t) n;
        for (int64_t c = 0; c < n; ++c) ids[(size_t) (i * cap + c)] = (int32_t) (first[(size_t) i] + c);
    }
    // FP64 reference
    std::vector<double> ref((size_t) (nq * NH * HD), 0.0);
    double scale = 0.0;
    for (int64_t i = 0; i < nq; ++i)
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV), n = count[(size_t) i];
            std::vector<double> sco((size_t) n);
            std::vector<int64_t> row((size_t) n);
            double mx = -1e300;
            for (int64_t c = 0; c < n; ++c) {
                const int64_t cell = first[(size_t) i] + c, page = p.table[(size_t) (cell / PS)];
                row[(size_t) c] = page < 0 ? -1 : (page * NKV + kvh) * PS + cell % PS;
                if (page < 0) continue;
                const int64_t r = row[(size_t) c];
                double a = 0;
                for (int64_t d = 0; d < HD; ++d)
                    a += (double) q[(size_t) ((i * NH + h) * HD + d)] * (double) p.kq[(size_t) (r * HD + d)] *
                         k::f32_from_f16(p.ks[(size_t) (r * 4 + d / 64)]);
                sco[(size_t) c] = a / std::sqrt((double) HD);
                mx = std::max(mx, sco[(size_t) c]);
            }
            double l = 0;
            for (int64_t c = 0; c < n; ++c) {
                sco[(size_t) c] = row[(size_t) c] < 0 ? 0.0 : std::exp(sco[(size_t) c] - mx);
                l += sco[(size_t) c];
            }
            for (int64_t c = 0; c < n; ++c) {
                const int64_t r = row[(size_t) c];
                if (r < 0) continue;
                const double w = l > 0 ? sco[(size_t) c] / l : 0.0;
                for (int64_t d = 0; d < HD; ++d)
                    ref[(size_t) ((i * NH + h) * HD + d)] +=
                        w * (double) p.vq[(size_t) (r * HD + d)] * k::f32_from_f16(p.vs[(size_t) (r * 4 + d / 64)]);
            }
        }
    for (double v : ref) scale = std::max(scale, std::fabs(v));

    const int32_t* d_ids = up(dev, ids);
    const int32_t* d_steps = up(dev, steps);
    const float* d_q = up(dev, q);
    const size_t n_out = (size_t) (nq * NH * HD);
    float* d_old = dev.alloc<float>(n_out);
    float* d_new = dev.alloc<float>(n_out);
    float* scratch = dev.alloc<float>((size_t) (nq * (int64_t) k::qsa_decode_attn_scratch_floats(cap, s)));
    std::vector<float> o(n_out), nw(n_out);
    k::qsa_decode_attn_batch(d_q, p.dev, d_ids, d_steps, cap, s, scratch, d_old, nq, par::stream());
    par::sync();
    par::copy_out(o.data(), d_old, n_out);
    double err_old = 0;
    for (size_t i = 0; i < n_out; ++i) err_old = std::max(err_old, std::fabs((double) o[i] - ref[i]));
    const bool old_ok = err_old <= 1e-4 * scale;
    int bad = old_ok ? 0 : 1;
    std::printf("  %-44s scalar vs FP64 %.3g of the scale %s\n", name, err_old / scale, old_ok ? "ok" : "FAIL");
    for (const char* slots : {"1", "7", ""}) {
        set_slots(slots);
        if (!k::qsa_split_attn_batch(d_q, p.dev, d_ids, d_steps, cap, s, scratch, d_new, nq, par::stream())) {
            // the scalar kernel is not this test's subject: without the new one its result is reported, not counted
            // (on the OpenCL CPU device its sub-group reductions do not hold, as qsa_dense_attn_parity shows there)
            std::printf("  %-44s SKIP: no kernel on the matrix engines for this device\n", name);
            bad = 0;
            break;
        }
        par::sync();
        par::copy_out(nw.data(), d_new, n_out);
        double err_new = 0, diff = 0;
        for (size_t i = 0; i < n_out; ++i) {
            err_new = std::max(err_new, std::fabs((double) nw[i] - ref[i]));
            diff = std::max(diff, (double) std::fabs(nw[i] - o[i]));
        }
        const bool ok1 = err_new <= std::max(4.0 * err_old, 1e-6 * scale), ok2 = diff <= 1e-4 * scale;
        std::printf("  %-44s slots %-6s vs FP64 %.3g of the scale %s; vs scalar %.3g %s\n", name,
                    *slots ? slots : "device", err_new / scale, ok1 ? "ok" : "FAIL", diff / scale, ok2 ? "ok" : "FAIL");
        bad += (ok1 ? 0 : 1) + (ok2 ? 0 : 1);
    }
    set_slots("");
    return bad;
}

int bench(int64_t cells) {
    par::Device dev;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t cap = 32768, NH = s.n_head, HD = s.head_dim;
    cells = std::min(cells, cap);
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    Pools p = make_pools(dev, s, cells, rng, -1);
    std::vector<int32_t> ids((size_t) cap, 0), step((size_t) k::kStepCount, 0);
    for (int64_t c = 0; c < cells; ++c) ids[(size_t) c] = (int32_t) c;
    step[k::kStepPos] = (int32_t) (cells - 1);
    step[k::kStepNKv] = (int32_t) cells;
    step[k::kStepWidth] = (int32_t) cells;
    std::vector<float> q((size_t) (NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;
    const int32_t* d_ids = up(dev, ids);
    const int32_t* d_step = up(dev, step);
    const float* d_q = up(dev, q);
    float* out = dev.alloc<float>((size_t) (NH * HD));
    float* scratch = dev.alloc<float>((size_t) k::qsa_decode_attn_scratch_floats(cap, s));
    const auto timed = [&](auto f) {
        for (int i = 0; i < 20; ++i) f();
        par::sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; ++i) f();
        par::sync();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200;
    };
    std::printf("qsa_split_attn --bench: one query, %lld cells of a %lld-cell window, INT8 K and V\n", (long long) cells,
                (long long) cap);
    std::printf("  the scalar kernel: %.1f us a call\n", timed([&] {
        k::qsa_decode_attn_batch(d_q, p.dev, d_ids, d_step, cap, s, scratch, out, 1, par::stream());
    }));
    for (const char* slots : {"", "4", "8", "16", "32", "64", "128", "256", "512"}) {
        set_slots(slots);
        if (!k::qsa_split_attn_batch(d_q, p.dev, d_ids, d_step, cap, s, scratch, out, 1, par::stream())) {
            std::printf("  no kernel on the matrix engines for this device\n");
            return 0;
        }
        std::printf("  the matrix engines, slots %-6s: %.1f us a call\n", *slots ? slots : "device", timed([&] {
            k::qsa_split_attn_batch(d_q, p.dev, d_ids, d_step, cap, s, scratch, out, 1, par::stream());
        }));
    }
    set_slots("");
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 && std::strcmp(argv[1], "--bench") == 0) return bench(std::strtoll(argv[2], nullptr, 10));
    const int64_t cap = 32768;
    int fails = 0;
    for (const int64_t n : {(int64_t) 1, (int64_t) 15, (int64_t) 59, (int64_t) 1000, (int64_t) 26800})
        fails += run_case(("one query, " + std::to_string(n) + " cells").c_str(), cap, {0}, {n}, -1);
    fails += run_case("four queries, their own windows", 4096, {0, 0, 5, 904}, {17, 2000, 4096, 4096}, -1);
    fails += run_case("a block that is not resident", cap, {0}, {3000}, 100);
    std::printf("FAILURES: %d\n", fails);
    return fails;
}
