// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/draft_head.cpp - see include/strata/kernels/draft_head.hpp.
#include "strata/kernels/draft_head.hpp"

#include "strata/core/runtime.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace strata::kernels {
namespace {

sycl::queue& Q(void* stream) { return core::Runtime::get().stream(stream); }

void done(void* stream, const sycl::event& e, const char* what) {
    if (!stream) core::Runtime::get().wait(e, what);
}

constexpr int kTopWG = 256, kTopPer = 16, kTopChunk = kTopWG * kTopPer;   // topk_stage's work-group and chunk

// One stage of topk: each work-group the k largest of its kTopChunk of the `n` values (their indices `ids_in[i]`, or
// i without), into out_v / out_i at k a group.  Each work-item holds kTopPer of them; the group's reductions pick one
// a turn, so no local memory and no barrier of its own.
sycl::event topk_stage(sycl::queue& q, const float* v, const int32_t* ids_in, int64_t n, int k, float* out_v,
                       int32_t* out_i) {
    const int64_t groups = (n + kTopChunk - 1) / kTopChunk;
    return q.parallel_for(sycl::nd_range<1>((size_t) groups * kTopWG, kTopWG), [=](sycl::nd_item<1> it) {
        constexpr float kNone = -std::numeric_limits<float>::infinity();
        const int64_t g = (int64_t) it.get_group(0), base = g * kTopChunk;
        const int t = (int) it.get_local_id(0);
        float val[kTopPer];
        int32_t id[kTopPer];
#pragma unroll
        for (int i = 0; i < kTopPer; ++i) {
            const int64_t at = base + (int64_t) i * kTopWG + t;
            val[i] = at < n ? v[at] : kNone;
            id[i] = at < n ? (ids_in != nullptr ? ids_in[at] : (int32_t) at) : -1;
        }
        for (int j = 0; j < k; ++j) {
            float best = kNone;
            int slot = -1;
#pragma unroll
            for (int i = 0; i < kTopPer; ++i)
                if (id[i] >= 0 && (slot < 0 || val[i] > best)) { best = val[i]; slot = i; }
            const float top = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
            // the lowest work-item holding it gives it up, so one value leaves the chunk each turn
            const int who = sycl::reduce_over_group(it.get_group(), slot >= 0 && best == top ? t : kTopWG,
                                                    sycl::minimum<int>());
            if (t == who) {
                out_v[g * k + j] = top;
                out_i[g * k + j] = id[slot];
                id[slot] = -1;
            } else if (who == kTopWG && t == 0) {   // fewer than k values in the chunk
                out_v[g * k + j] = kNone;
                out_i[g * k + j] = -1;
            }
        }
    });
}

}  // namespace

std::size_t int2_row_bytes(int n_in) {
    return (std::size_t) n_in / 4 + (std::size_t) (n_in / kInt2Group) * 4;
}

void int2_pack_rows(const float* rows, int64_t n_rows, int n_in, uint8_t* out) {
    if (n_in <= 0 || n_in % kInt2Group != 0) throw std::invalid_argument("int2_pack_rows: n_in is not a multiple of 128");
    const std::size_t rb = int2_row_bytes(n_in);
    for (int64_t r = 0; r < n_rows; ++r) {
        const float* x = rows + r * (int64_t) n_in;
        uint8_t* codes = out + (std::size_t) r * rb;
        auto* scales = reinterpret_cast<sycl::half*>(codes + n_in / 4);
        std::fill(codes, codes + n_in / 4, (uint8_t) 0);
        for (int g = 0; g < n_in / kInt2Group; ++g) {
            const float* xg = x + (std::size_t) g * kInt2Group;
            const auto [lo, hi] = std::minmax_element(xg, xg + kInt2Group);
            const sycl::half m = (sycl::half) *lo, d = (sycl::half) ((*hi - *lo) / 3.0f);
            sycl::half* md = scales + (std::size_t) g * 2;
            md[0] = m;
            md[1] = d;
            const float fm = (float) m, fd = (float) d;
            for (int i = 0; i < kInt2Group; ++i) {
                const int q = fd > 0.0f ? std::clamp((int) std::lround((xg[i] - fm) / fd), 0, 3) : 0;
                const int at = g * kInt2Group + i;
                codes[at / 4] |= (uint8_t) (q << (2 * (at % 4)));
            }
        }
    }
}

void int2_gemv(const uint8_t* w, const float* x, float* y, int n_in, int64_t n_out, void* stream) {
    if (n_out <= 0) return;
    if (n_in <= 0 || n_in % kInt2Group != 0) throw std::invalid_argument("int2_gemv: n_in is not a multiple of 128");
    // a sub-group a row at a time, its lanes on consecutive 32-bit words of codes (16 values each), a work-group of
    // kRows rows: one work-group a row left most of its time to the launch (B70 271 GB/s, RTX 4070 68 GB/s on the
    // 2560 x 106,299 draft head, 2026-10-26).  The sub-group's width is the device's.
    constexpr int WG = 256, kRows = 32;
    const std::size_t rb = int2_row_bytes(n_in);
    const int words = n_in / 16;
    const int64_t groups = (n_out + kRows - 1) / kRows;
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>((size_t) groups * WG, WG), [=](sycl::nd_item<1> it) {
        const sycl::sub_group sg = it.get_sub_group();
        const int lanes = (int) sg.get_local_range()[0], lane = (int) sg.get_local_id()[0];
        const int n_sg = (int) sg.get_group_range()[0], sgi = (int) sg.get_group_id()[0];
        const int64_t r0 = (int64_t) it.get_group(0) * kRows;
        for (int rr = sgi; rr < kRows; rr += n_sg) {
            const int64_t r = r0 + rr;
            if (r >= n_out) break;   // the same for the whole sub-group
            const uint8_t* row = w + (std::size_t) r * rb;
            const auto* code = reinterpret_cast<const uint32_t*>(row);
            const auto* sc = reinterpret_cast<const sycl::half*>(row + n_in / 4);
            float acc = 0.0f;
            for (int wd = lane; wd < words; wd += lanes) {
                const uint32_t c = code[wd];
                const sycl::float4* xw = reinterpret_cast<const sycl::float4*>(x + (std::size_t) wd * 16);
                float dot = 0.0f, sum = 0.0f;
#pragma unroll
                for (int j4 = 0; j4 < 4; ++j4) {
                    const sycl::float4 xv = xw[j4];
                    const uint32_t cb = c >> (8 * j4);
                    dot = sycl::fma((float) (cb & 3u), xv.x(), dot);
                    dot = sycl::fma((float) ((cb >> 2) & 3u), xv.y(), dot);
                    dot = sycl::fma((float) ((cb >> 4) & 3u), xv.z(), dot);
                    dot = sycl::fma((float) ((cb >> 6) & 3u), xv.w(), dot);
                    sum += (xv.x() + xv.y()) + (xv.z() + xv.w());
                }
                const sycl::half* md = sc + (std::size_t) (wd / (kInt2Group / 16)) * 2;   // the group's minimum and step
                acc = sycl::fma((float) md[1], dot, sycl::fma((float) md[0], sum, acc));
            }
            const float total = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) y[r] = total;
        }
    });
    done(stream, e, "int2_gemv");
}

namespace {
constexpr int kXStride = kInt2XBits + 3;   // a span of int2_x_planes: its planes, then its least value, step and sum
}

void int2_rows_to_planes(uint8_t* rows, int64_t n_rows, int n_in) {
    if (n_in <= 0 || n_in % kInt2Group != 0) throw std::invalid_argument("int2_rows_to_planes: n_in is not a multiple of 128");
    const std::size_t rb = int2_row_bytes(n_in);
    const int spans = n_in / 32;
    std::vector<uint32_t> planes((std::size_t) spans * 2);
    for (int64_t r = 0; r < n_rows; ++r) {
        uint8_t* codes = rows + (std::size_t) r * rb;
        for (int sp = 0; sp < spans; ++sp) {
            uint32_t lo = 0, hi = 0;
            for (int i = 0; i < 32; ++i) {
                const int at = sp * 32 + i;
                const uint32_t c = (codes[at / 4] >> (2 * (at % 4))) & 3u;
                lo |= (c & 1u) << i;
                hi |= (c >> 1) << i;
            }
            planes[(std::size_t) sp] = lo;
            planes[(std::size_t) spans + (std::size_t) sp] = hi;
        }
        std::memcpy(codes, planes.data(), planes.size() * sizeof(uint32_t));
    }
}

std::size_t int2_x_planes_bytes(int n_in) { return (std::size_t) (n_in / 32) * kXStride * sizeof(uint32_t); }

void int2_x_planes(const float* x, void* xp, int n_in, void* stream) {
    if (n_in <= 0 || n_in % kInt2Group != 0) throw std::invalid_argument("int2_x_planes: n_in is not a multiple of 128");
    auto* out = static_cast<uint32_t*>(xp);
    // a work-group a span: its least and largest value and its sum, then each plane as the group's OR
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>((size_t) n_in, 32), [=](sycl::nd_item<1> it) {
        constexpr float kTop = (float) ((1 << kInt2XBits) - 1);
        const auto grp = it.get_group();
        const int j = (int) it.get_local_id(0);
        const float xv = x[it.get_global_id(0)];
        const float lo = sycl::reduce_over_group(grp, xv, sycl::minimum<float>());
        const float hi = sycl::reduce_over_group(grp, xv, sycl::maximum<float>());
        const float sum = sycl::reduce_over_group(grp, xv, sycl::plus<float>());
        const float step = (hi - lo) / kTop;
        const uint32_t u = step > 0.0f ? (uint32_t) sycl::clamp(sycl::rint((xv - lo) / step), 0.0f, kTop) : 0u;
        uint32_t* o = out + (std::size_t) it.get_group(0) * kXStride;
#pragma unroll
        for (int b = 0; b < kInt2XBits; ++b) {
            const uint32_t word = sycl::reduce_over_group(grp, ((u >> b) & 1u) << j, sycl::bit_or<uint32_t>());
            if (j == 0) o[b] = word;
        }
        if (j == 0) {
            o[kInt2XBits] = sycl::bit_cast<uint32_t>(lo);
            o[kInt2XBits + 1] = sycl::bit_cast<uint32_t>(step);
            o[kInt2XBits + 2] = sycl::bit_cast<uint32_t>(sum);
        }
    });
    done(stream, e, "int2_x_planes");
}

void int2_planes_gemv(const uint8_t* w, const void* xp, float* y, int n_in, int64_t n_out, void* stream) {
    if (n_out <= 0) return;
    if (n_in <= 0 || n_in % kInt2Group != 0) throw std::invalid_argument("int2_planes_gemv: n_in is not a multiple of 128");
    // A sub-group takes a span a lane and kBatch rows at a time: the span's planes are read once for those rows, and a
    // row then costs two words of codes and its group's two scales a lane.  A work-group of kRows rows, whatever the
    // device's sub-group width (B70 and B65, 2560 x 40,525, 2026-11-11: see xemaya/benchmarks/2026-11-11-qwen-ab).
    constexpr int WG = 256, kRows = 128, kBatch = 8;
    const std::size_t rb = int2_row_bytes(n_in);
    const int spans = n_in / 32;
    const int64_t groups = (n_out + kRows - 1) / kRows;
    const auto* xs = static_cast<const uint32_t*>(xp);
    const auto e = Q(stream).parallel_for(sycl::nd_range<1>((size_t) groups * WG, WG), [=](sycl::nd_item<1> it) {
        const sycl::sub_group sg = it.get_sub_group();
        const int lanes = (int) sg.get_local_range()[0], lane = (int) sg.get_local_id()[0];
        const int n_sg = (int) sg.get_group_range()[0], sgi = (int) sg.get_group_id()[0];
        const int64_t r0 = (int64_t) it.get_group(0) * kRows;
        for (int j0 = 0; sgi + j0 * n_sg < kRows; j0 += kBatch) {   // the same for the whole sub-group
            float acc[kBatch];
#pragma unroll
            for (int jj = 0; jj < kBatch; ++jj) acc[jj] = 0.0f;
            for (int sp = lane; sp < spans; sp += lanes) {
                const uint32_t* xw = xs + (std::size_t) sp * kXStride;
                uint32_t u[kInt2XBits];
#pragma unroll
                for (int b = 0; b < kInt2XBits; ++b) u[b] = xw[b];
                const float xlo = sycl::bit_cast<float>(xw[kInt2XBits]);
                const float xstep = sycl::bit_cast<float>(xw[kInt2XBits + 1]);
                const float xsum = sycl::bit_cast<float>(xw[kInt2XBits + 2]);
#pragma unroll
                for (int jj = 0; jj < kBatch; ++jj) {
                    const int rr = sgi + (j0 + jj) * n_sg;
                    const int64_t r = r0 + rr;
                    if (rr < kRows && r < n_out) {
                        const uint8_t* row = w + (std::size_t) r * rb;
                        const auto* code = reinterpret_cast<const uint32_t*>(row);
                        const auto* md = reinterpret_cast<const sycl::half*>(row + n_in / 4) +
                                         (std::size_t) (sp / (kInt2Group / 32)) * 2;   // the group's minimum and step
                        const uint32_t c0 = code[sp], c1 = code[spans + sp];
                        uint32_t dot = 0;
#pragma unroll
                        for (int b = 0; b < kInt2XBits; ++b)
                            dot += (sycl::popcount(c0 & u[b]) + 2u * sycl::popcount(c1 & u[b])) << b;
                        const uint32_t csum = sycl::popcount(c0) + 2u * sycl::popcount(c1);
                        // sum of (m + d c)(xlo + xstep u) = m * the span's sum + d * (xlo * sum c + xstep * sum c u)
                        const float cx = sycl::fma(xstep, (float) dot, xlo * (float) csum);
                        acc[jj] = sycl::fma((float) md[1], cx, sycl::fma((float) md[0], xsum, acc[jj]));
                    }
                }
            }
#pragma unroll
            for (int jj = 0; jj < kBatch; ++jj) {
                const int rr = sgi + (j0 + jj) * n_sg;
                const int64_t r = r0 + rr;
                const float total = sycl::reduce_over_group(sg, acc[jj], sycl::plus<float>());
                if (lane == 0 && rr < kRows && r < n_out) y[r] = total;
            }
        }
    });
    done(stream, e, "int2_planes_gemv");
}

std::size_t topk_scratch_bytes(int64_t n, int k) {
    // the first stage's candidates at most, twice (the stages go back and forth), values and indices
    const int64_t c = ((n + kTopChunk - 1) / kTopChunk) * k;
    return (std::size_t) (2 * c) * (sizeof(float) + sizeof(int32_t));
}

void topk(const float* v, int64_t n, int k, int32_t* ids, void* scratch, void* stream) {
    if (k < 1 || k > kTopKMax || n < 1) throw std::invalid_argument("topk: k is 1 to 64 and n positive");
    auto& q = Q(stream);
    const int64_t cap = ((n + kTopChunk - 1) / kTopChunk) * k;   // topk_scratch_bytes' room for one side
    auto* fv = static_cast<float*>(scratch);
    auto* fi = reinterpret_cast<int32_t*>(fv + 2 * cap);
    float* bv[2] = {fv, fv + cap};
    int32_t* bi[2] = {fi, fi + cap};
    const float* in_v = v;
    const int32_t* in_i = nullptr;
    int side = 0;
    sycl::event e;
    while (true) {
        const int64_t groups = (n + kTopChunk - 1) / kTopChunk;
        e = topk_stage(q, in_v, in_i, n, k, bv[side], bi[side]);
        n = groups * k;
        in_v = bv[side];
        in_i = bi[side];
        side ^= 1;
        if (groups == 1) break;
    }
    e = q.memcpy(ids, in_i, (size_t) k * sizeof(int32_t));
    done(stream, e, "topk");
}

void int2_pad_from(const int32_t* pos, int32_t from, float* y, int64_t n, void* stream) {
    if (n <= 0 || pos == nullptr) return;
    const auto e = Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
        if (*pos >= from) y[i] = -1.7e38f;
    });
    done(stream, e, "int2_pad_from");
}

void scatter_values(const float* vals, const int32_t* ids, int k, float* y, void* stream) {
    if (k <= 0) return;
    const auto e = Q(stream).parallel_for(sycl::range<1>((size_t) k), [=](sycl::id<1> j) {
        const int32_t at = ids[j];
        if (at >= 0) y[at] = vals[j];
    });
    done(stream, e, "scatter_values");
}

}  // namespace strata::kernels
