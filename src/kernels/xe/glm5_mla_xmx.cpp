// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/glm5_mla_xmx.cpp - GLM-5.3-Flash's DSA attention over the latent cells (glm5_mla_attn's semantics) on
// Xe's matrix engines, for prompt chunks.  See include/strata/kernels/glm5.hpp.
//
// The scheme of qsa_prompt_attn.cpp's Intel kernel (Strata's "v2"), for MLA: K = V = the 512-dim FP16 latent row, 64
// query heads sharing it.
//   * a work-group of 16 sub-groups takes one (token, 16 heads) and walks the token's cells 16 at a time with an
//     online softmax (lazy rescaling, as FlashAttention-3); sub-groups 2g and 2g + 1 own dims [64g, 64g + 64) for both
//     q.k and p.v, heads 0-7 and 8-15 of the 16;
//   * a chunk's latent rows are gathered once into local memory as FP16 and serve as K (q.k, col-major B) and as V
//     (p.v, row-major B): the 16 heads read each cell once (the FP32 kernel read it twice for every head);
//   * q is scaled by a power of two that puts its largest value near 2^14 and split into FP16 hi + lo, and so is p:
//     two products each, so they keep about 22 bits; the 8 groups' q.k partials are added in a fixed order.
// Decode windows keep glm5_mla_attn_tiled (a token's 4 work-groups would leave the GPU idle).
#include "strata/kernels/glm5.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_glm5_mla_xmx
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace strata::kernels {
namespace {

namespace mx = sycl::ext::oneapi::experimental::matrix;
namespace ix = sycl::ext::intel::experimental::matrix;
namespace syclex = sycl::ext::oneapi::experimental;
using sycl::half;

constexpr int L = 512;            // the latent's dims
constexpr int R = 16;             // heads a work-group: two 8-row tiles
constexpr int NG = L / 64;        // 64-dim groups
constexpr int SGS = 16;           // sub-group size
constexpr int NSG = 2 * NG;       // sub-groups: a group's dims x a tile of 8 rows
constexpr int WG = NSG * SGS;     // work-items
constexpr int CH = 16;            // cells a chunk: a lane each in the softmax
constexpr int QS = L + 8;         // row strides in halves (16-byte aligned rows)
constexpr int KR = L + 8;
constexpr int KV_HALVES = 2 * R * QS > CH * KR ? 2 * R * QS : CH * KR;   // the chunk's rows; q (hi + lo) at the start
[[maybe_unused]] constexpr int TM = 8, TN = 16, TK = 16;
[[maybe_unused]] constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
[[maybe_unused]] constexpr float TAU = 8.0f;                   // a row's reference moves past 2^TAU only
[[maybe_unused]] constexpr float PSCALE = 16384.0f / 256.0f;   // p' = p 2^(14 - TAU) <= 2^14 within FP16

template<typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

struct MlaXmx {
    const float* q;
    const uint16_t* latent;
    const int32_t* ids;
    const int32_t* steps;
    int64_t cap, compact;
    int H, KP;
    float scale_log2;
    float* out;
    sycl::local_accessor<half, 1> kvq;      // the chunk's rows [CH][KR]; at the start qh [R][QS], ql [R][QS]
    sycl::local_accessor<long long, 1> rows;
    sycl::local_accessor<float, 3> part;    // [NG][R][CH]: q.k per 64-dim group
    sycl::local_accessor<half, 3> pp;       // [2][R][CH]: p' hi, lo
    sycl::local_accessor<float, 1> mrow, lsum, alpha;

    // x as FP16 hi + lo (qsa_prompt_attn.cpp's split: hi back to FP32 through its bits, so lo is not optimized away)
    static void split(float x, half& hi, half& lo) {
        hi = half(x);
        lo = half(x - f32_from_f16(sycl::bit_cast<uint16_t>(hi)));
    }

    void operator()(sycl::nd_item<2> it) const {
#if STRATA_DEVICE_NOT_INTEL
        (void) it;
#else
        const auto grp = it.get_group();
        const auto sg = it.get_sub_group();
        const int64_t t_ = (int64_t) it.get_group(0), hg = (int64_t) it.get_group(1);
        const int t = (int) it.get_local_id(1), s = (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        const int g = s / 2, mt = s % 2;   // this sub-group's dim group and row tile
        const float* qp = q + (t_ * H + hg * R) * L;
        float* op = out + (t_ * H + hg * R) * L;
        // the token's cells: its selection, then its unfinished pool's cells up to it (glm5_mla_attn's)
        const int64_t chosen = sycl::min<int64_t>(steps[t_ * kStepCount + kStepWidth], cap);
        const int32_t pos = steps[t_ * kStepCount + kStepPos];
        const int64_t tail0 = KP > 0 && pos >= 0 ? (int64_t) ((pos + 1) / KP) * KP : 0;
        const int64_t width = chosen + (KP > 0 && pos >= 0 ? (int64_t) pos + 1 - tail0 : 0);
        const int32_t* sel = ids + t_ * cap;
        const size_t QH = 0, QL = (size_t) R * QS;

        // q: 16 heads, times 2^(14 - e) for its largest |value| < 2^e, as FP16 hi + lo
        float qm = 0.0f;
        for (int i = t; i < R * L; i += WG) qm = sycl::fmax(qm, sycl::fabs(qp[i]));
        qm = sycl::reduce_over_group(grp, qm, sycl::maximum<float>());
        const int qe = qm > 0.0f ? (int) ((sycl::bit_cast<uint32_t>(qm) >> 23) & 0xFF) - 126 : 0;   // qm < 2^qe
        const float qup = sycl::ldexp(1.0f, 14 - qe), qdown = sycl::ldexp(scale_log2, qe - 14);
        for (int i = t; i < R * L; i += WG) {
            const int r = i / L, d = i % L;
            half hi, lo;
            split(qp[i] * qup, hi, lo);
            kvq[QH + (size_t) r * QS + d] = hi;
            kvq[QL + (size_t) r * QS + d] = lo;
        }
        if (t < R) { mrow[t] = NEG_INF; lsum[t] = 0.0f; }
        sycl::group_barrier(grp);
        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> qa[4], qb[4];
        for (int kk = 0; kk < 4; ++kk) {
            const size_t off = (size_t) mt * TM * QS + (size_t) g * 64 + (size_t) kk * TK;
            mx::joint_matrix_load(sg, qa[kk], at(kvq, QH + off), QS);
            mx::joint_matrix_load(sg, qb[kk], at(kvq, QL + off), QS);
        }
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc[4];
        for (int nt = 0; nt < 4; ++nt) mx::joint_matrix_fill(sg, acc[nt], 0.0f);

        for (int64_t c0 = 0; c0 < width; c0 += CH) {   // uniform over the work-group
            const int nh = (int) sycl::min<int64_t>(CH, width - c0);
            if (t < CH) {
                long long r = -1;
                if (t < nh) {
                    const int64_t j = c0 + t;
                    r = compact > 0 ? t_ * compact + j : j < chosen ? (int64_t) sel[j] : tail0 + (j - chosen);
                }
                rows[t] = r;
            }
            sycl::group_barrier(grp);   // rows ready; the last chunk (and q's staging) is done with kvq, part, pp
            for (int i = t; i < CH * (L / 8); i += WG) {
                const int c = i / (L / 8), pc = i % (L / 8);
                const long long r = rows[c];
                sycl::vec<half, 8> v(half(0.0f));
                if (r >= 0) v = *reinterpret_cast<const sycl::vec<half, 8>*>(latent + r * L + (int64_t) pc * 8);
                *reinterpret_cast<sycl::vec<half, 8>*>(&kvq[(size_t) c * KR + (size_t) pc * 8]) = v;
            }
            sycl::group_barrier(grp);
            {   // q.k of this sub-group's 8 rows and 64 dims over the 16 cells: hi and lo
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c;
                mx::joint_matrix_fill(sg, c, 0.0f);
                for (int kk = 0; kk < 4; ++kk) {
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> b;
                    mx::joint_matrix_load(sg, b, at(kvq, (size_t) g * 64 + (size_t) kk * TK), KR);
                    mx::joint_matrix_mad(sg, c, qa[kk], b, c);
                    mx::joint_matrix_mad(sg, c, qb[kk], b, c);
                }
                mx::joint_matrix_store(sg, c, at(part, (size_t) (g * R + mt * TM) * CH), CH, mx::layout::row_major);
            }
            sycl::group_barrier(grp);
            {   // online softmax: sub-group s takes row s, lane = cell; the groups' partials in order
                const int r = s, cc = lane;
                float sc = part[0][r][cc];
                for (int gg = 1; gg < NG; ++gg) sc += part[gg][r][cc];
                const float x = cc < nh && rows[cc] >= 0 ? sc * qdown : NEG_INF;
                const float mxv = sycl::reduce_over_group(sg, x, sycl::maximum<float>());
                const float m_old = mrow[r];
                const float m_new = m_old == NEG_INF || mxv > m_old + TAU ? mxv : m_old;
                const float e = x == NEG_INF ? 0.0f : sycl::exp2(x - m_new);
                const float sum = sycl::reduce_over_group(sg, e, sycl::plus<float>());
                half hi, lo;
                split(e * PSCALE, hi, lo);
                pp[0][r][cc] = hi;
                pp[1][r][cc] = lo;
                if (lane == 0) {
                    const float a = m_old == m_new ? 1.0f : m_old == NEG_INF ? 0.0f : sycl::exp2(m_old - m_new);
                    alpha[r] = a;
                    lsum[r] = sycl::fma(lsum[r], a, sum);
                    mrow[r] = m_new;
                }
            }
            sycl::group_barrier(grp);
            {   // p.v: this sub-group's 8 rows x its 64 dims; an 8 x 16 accumulator's lane holds a column, element i in
                // row i (Xe's DPAS layout, as qsa_prompt_attn.cpp relies on; glm5_parity's test_mla_xmx checks it)
                float f[TM];
                bool same = true;
                for (int i = 0; i < TM; ++i) {
                    f[i] = alpha[mt * TM + i];
                    same = same && f[i] == 1.0f;
                }
                if (!same)
                    for (int nt = 0; nt < 4; ++nt) {
                        int i = 0;
                        mx::joint_matrix_apply(sg, acc[nt], [&](float& x) { x *= f[i++]; });
                    }
                mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> ph, pl;
                mx::joint_matrix_load(sg, ph, at(pp, (size_t) (mt * TM) * CH), CH);
                mx::joint_matrix_load(sg, pl, at(pp, (size_t) (R + mt * TM) * CH), CH);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::row_major> b;
                    mx::joint_matrix_load(sg, b, at(kvq, (size_t) g * 64 + (size_t) nt * TN), KR);
                    mx::joint_matrix_mad(sg, acc[nt], ph, b, acc[nt]);
                    mx::joint_matrix_mad(sg, acc[nt], pl, b, acc[nt]);
                }
            }
        }
        sycl::group_barrier(grp);
        for (int nt = 0; nt < 4; ++nt)
            ix::joint_matrix_apply(sg, acc[nt], [&](float& x, size_t row, size_t col) {
                const int r = mt * TM + (int) row;
                const float l = lsum[r];
                op[(int64_t) r * L + g * 64 + nt * TN + (int) col] = l > 0.0f ? x * (1.0f / PSCALE) / l : 0.0f;
            });
#endif
    }
    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SGS>};
    }
};

constexpr size_t local_bytes() {
    return (size_t) KV_HALVES * 2 + (size_t) CH * 8 + (size_t) NG * R * CH * 4 + (size_t) 2 * R * CH * 2 + 3 * R * 4;
}

}  // namespace

bool glm5_mla_attn_xmx_on() {
    static const bool on = [] {
        const char* v = std::getenv("XEMAYA_MLA_ATTN");   // the default since window 3; tiled or ref choose the others
        return v == nullptr || std::strcmp(v, "xmx") == 0;
    }();
    return on;
}

bool glm5_mla_attn_xmx(const float* q, const uint16_t* latent, const int32_t* ids, int64_t cap, const int32_t* steps,
                       int n_heads, float scale, int64_t n_tokens, float* out, void* stream, int tail_kpool,
                       int64_t compact) {
    if (n_tokens <= 0) return true;
    if (n_heads <= 0 || n_heads % R != 0 || cap <= 0 || tail_kpool < 0 || tail_kpool == 1 || n_tokens > 0x7fffffff)
        return false;
    if (!xmx_available(XmxType::f16)) return false;
    auto& queue = core::Runtime::get().stream(stream);
    if (queue.get_device().get_info<sycl::info::device::local_mem_size>() < local_bytes()) return false;
    const float scale_log2 = scale * 1.4426950408889634f;
    const auto e = queue.submit([&](sycl::handler& h) {
        MlaXmx k{q, latent, ids, steps, cap, compact, n_heads, tail_kpool, scale_log2, out,
                 sycl::local_accessor<half, 1>(sycl::range<1>(KV_HALVES), h),
                 sycl::local_accessor<long long, 1>(sycl::range<1>(CH), h),
                 sycl::local_accessor<float, 3>(sycl::range<3>(NG, R, CH), h),
                 sycl::local_accessor<half, 3>(sycl::range<3>(2, R, CH), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                 sycl::local_accessor<float, 1>(sycl::range<1>(R), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) n_tokens, (size_t) (n_heads / R) * WG}, {1, (size_t) WG}), k);
    });
    if (!stream) core::Runtime::get().wait(e, "glm5_mla_attn_xmx");
    return true;
}

}  // namespace strata::kernels
