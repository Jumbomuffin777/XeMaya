// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/kernels/xe/minimax_attn_xmx.cpp - MiniMax-M3's sparse attention (mm_msa_attn's semantics) on Xe's matrix
// engines.  See include/strata/kernels/minimax.hpp.
//
// ds41_attn_xmx.cpp's scheme (glm5_mla_xmx.cpp's, after qsa_prompt_attn.cpp's Intel kernel, Strata's "v2") with
// MiniMax-M3's keys: K and V are 128-dim FP16 rows of the paged pools, and the 16 query heads of a K/V head share
// them and their selection.
//   * a work-group of 16 sub-groups takes one (token, K/V head) and walks that head's cells 16 at a time with an
//     online softmax; sub-groups 2g and 2g + 1 own dims [16g, 16g + 16) for q.k and p.v, query heads 0-7 and 8-15;
//   * a chunk's K and V rows are copied once into local memory (FP16 as they are stored): K as a col-major B, V as a
//     row-major B;
//   * q and p are FP16 hi + lo (about 22 bits), as there.
#include "strata/kernels/minimax.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/xmx_gemm.hpp"
#include "strata/core/runtime.hpp"
#include "device_target.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_minimax_attn_xmx
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

constexpr int L = 128;            // a head's dims
constexpr int R = 16;             // query heads a K/V head: two 8-row tiles
constexpr int GD = 16;            // dims a sub-group pair owns
constexpr int NG = L / GD;        // dim groups
constexpr int SGS = 16;           // sub-group size
constexpr int NSG = 2 * NG;       // sub-groups: a group's dims x a tile of 8 rows (= R: a softmax row each)
constexpr int WG = NSG * SGS;     // work-items
constexpr int CH = 16;            // cells a chunk: a lane each in the softmax
constexpr int QS = L + 8;         // row strides in halves (16-byte aligned rows)
constexpr int KR = L + 8;
[[maybe_unused]] constexpr int TM = 8, TN = 16, TK = 16;
[[maybe_unused]] constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
[[maybe_unused]] constexpr float TAU = 8.0f;                   // a row's reference moves past 2^TAU only
[[maybe_unused]] constexpr float PSCALE = 16384.0f / 256.0f;   // p' = p 2^(14 - TAU) <= 2^14 within FP16
[[maybe_unused]] constexpr float LOG2E = 1.4426950408889634f;
static_assert(NSG == R && CH * (L / 8) == WG, "a softmax row a sub-group; a work-item copies 8 values of a chunk");

template<typename A>
inline auto at(const A& a, size_t off) {
    return a.template get_multi_ptr<sycl::access::decorated::no>() + off;
}

struct MmXmx {
    const float* q;
    const uint16_t* k_pool;
    const uint16_t* v_pool;
    const int32_t* page_table;
    const int32_t* step;
    const int32_t* sel;
    const int32_t* count;
    int step_stride, top_k, blk;
    int64_t NH, NKV, PS;
    float scale_log2;
    float* out;
    sycl::local_accessor<half, 1> qhl;      // qh [R][QS], ql [R][QS]
    sycl::local_accessor<half, 1> kb, vb;   // the chunk's rows [CH][KR]
    sycl::local_accessor<long long, 1> rows;
    sycl::local_accessor<float, 3> part;    // [NG][R][CH]: q.k per dim group
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
        const int64_t t_ = (int64_t) it.get_group(0), hk = (int64_t) it.get_group(1);
        const int t = (int) it.get_local_id(1), s = (int) sg.get_group_linear_id();
        const int lane = (int) sg.get_local_linear_id();
        const int g = s / 2, mt = s % 2;   // this sub-group's dim group and row tile
        const float* qp = q + (t_ * NH + hk * R) * L;
        float* op = out + (t_ * NH + hk * R) * L;
        // the head's cells: every cell up to the token's own, or the cells of its blocks (mm_msa_attn's)
        const int64_t p = step[t_ * step_stride + kStepPos];
        const int32_t* bl = sel != nullptr ? sel + (t_ * NKV + hk) * top_k : nullptr;
        const int64_t width = sel != nullptr ? (int64_t) count[t_ * NKV + hk] * blk : p + 1;
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
            qhl[QH + (size_t) r * QS + d] = hi;
            qhl[QL + (size_t) r * QS + d] = lo;
        }
        if (t < R) {
            mrow[t] = NEG_INF;
            lsum[t] = 0.0f;
        }
        sycl::group_barrier(grp);
        mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> qa, qb;
        {
            const size_t off = (size_t) mt * TM * QS + (size_t) g * GD;
            mx::joint_matrix_load(sg, qa, at(qhl, QH + off), QS);
            mx::joint_matrix_load(sg, qb, at(qhl, QL + off), QS);
        }
        mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> acc;
        mx::joint_matrix_fill(sg, acc, 0.0f);

        for (int64_t c0 = 0; c0 < width; c0 += CH) {   // uniform over the work-group
            const int nh = (int) sycl::min<int64_t>(CH, width - c0);
            if (t < CH) {   // a cell's place in the pools, in halves; -1 none
                long long r = -1;
                if (t < nh) {
                    const int64_t j = c0 + t;
                    const int64_t c = bl == nullptr ? j : (int64_t) bl[j / blk] * blk + j % blk;
                    if (c <= p) r = (((long long) page_table[c / PS] * NKV + hk) * PS + c % PS) * L;
                }
                rows[t] = r;
            }
            sycl::group_barrier(grp);   // rows ready; the last chunk is done with kb, vb, part, pp
            {   // (a cell that is not there reads the pools' first row and stores zeros: no address outside them)
                const int c = t / (L / 8), pc = t % (L / 8);
                const long long r = rows[c];
                const uint16_t *ks = k_pool + (r < 0 ? 0 : r), *vs = v_pool + (r < 0 ? 0 : r);
                for (int u = 0; u < 8; ++u) {
                    kb[(size_t) c * KR + (size_t) pc * 8 + u] = r < 0 ? half(0.0f) : sycl::bit_cast<half>(ks[pc * 8 + u]);
                    vb[(size_t) c * KR + (size_t) pc * 8 + u] = r < 0 ? half(0.0f) : sycl::bit_cast<half>(vs[pc * 8 + u]);
                }
            }
            sycl::group_barrier(grp);
            {   // q.k of this sub-group's 8 rows and 16 dims over the 16 cells: hi and lo
                mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN> c;
                mx::joint_matrix_fill(sg, c, 0.0f);
                mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::col_major> b;
                mx::joint_matrix_load(sg, b, at(kb, (size_t) g * GD), KR);
                mx::joint_matrix_mad(sg, c, qa, b, c);
                mx::joint_matrix_mad(sg, c, qb, b, c);
                mx::joint_matrix_store(sg, c, at(part, (size_t) (g * R + mt * TM) * CH), CH, mx::layout::row_major);
            }
            sycl::group_barrier(grp);
            {   // online softmax: sub-group s takes row s, lane = cell; the groups' partials in order
                const int r = s, cc = lane;
                float sc = part[0][r][cc];
                for (int gg = 1; gg < NG; ++gg) sc += part[gg][r][cc];
                const float x = cc < nh && rows[cc] != -1 ? sc * qdown : NEG_INF;
                const float mxv = sycl::reduce_over_group(sg, x, sycl::maximum<float>());
                const float m_old = mrow[r];
                const float m_new = m_old == NEG_INF || mxv > m_old + TAU ? mxv : m_old;
                const float e = x == NEG_INF || m_new == NEG_INF ? 0.0f : sycl::exp2(x - m_new);
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
            {   // p.v: this sub-group's 8 rows x its 16 dims; an 8 x 16 accumulator's lane holds a column, element i in
                // row i (Xe's DPAS layout, as qsa_prompt_attn.cpp relies on; glm5_parity's test_mla_xmx checks it)
                float f[TM];
                bool same = true;
                for (int i = 0; i < TM; ++i) {
                    f[i] = alpha[mt * TM + i];
                    same = same && f[i] == 1.0f;
                }
                if (!same) {
                    int i = 0;
                    mx::joint_matrix_apply(sg, acc, [&](float& x) { x *= f[i++]; });
                }
                mx::joint_matrix<sycl::sub_group, half, mx::use::a, TM, TK, mx::layout::row_major> ph, pl;
                mx::joint_matrix_load(sg, ph, at(pp, (size_t) (mt * TM) * CH), CH);
                mx::joint_matrix_load(sg, pl, at(pp, (size_t) (R + mt * TM) * CH), CH);
                mx::joint_matrix<sycl::sub_group, half, mx::use::b, TK, TN, mx::layout::row_major> b;
                mx::joint_matrix_load(sg, b, at(vb, (size_t) g * GD), KR);
                mx::joint_matrix_mad(sg, acc, ph, b, acc);
                mx::joint_matrix_mad(sg, acc, pl, b, acc);
            }
        }
        sycl::group_barrier(grp);
        ix::joint_matrix_apply(sg, acc, [&](float& x, size_t row, size_t col) {
            const int r = mt * TM + (int) row;
            const float l = lsum[r];
            op[(int64_t) r * L + g * GD + (int) col] = l > 0.0f ? x * (1.0f / PSCALE) / l : 0.0f;
        });
#endif
    }
    auto get(syclex::properties_tag) const {
        return syclex::properties{sycl::ext::intel::experimental::grf_size<256>, syclex::sub_group_size<SGS>};
    }
};

constexpr size_t local_bytes() {
    return (size_t) 2 * R * QS * 2 + (size_t) 2 * CH * KR * 2 + (size_t) CH * 8 + (size_t) NG * R * CH * 4 +
           (size_t) 2 * R * CH * 2 + 3 * R * 4;
}

}  // namespace

bool mm_msa_dense() {
    static const bool on = std::getenv("XEMAYA_MM_DENSE") != nullptr;
    return on;
}

bool mm_msa_attn_xmx_on() {
    static const bool on = [] {
        const char* v = std::getenv("XEMAYA_MM_ATTN");
        return v == nullptr || std::strcmp(v, "ref") != 0;
    }();
    return on;
}

bool mm_msa_attn_xmx(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                     const int32_t* step, int step_stride, const int32_t* sel, const int32_t* count, int top_k,
                     int blk, const QsaShapes& s, float* attn, int64_t n_tok, void* stream) {
    if (n_tok <= 0) return true;
    if (s.head_dim != L || s.n_head_kv <= 0 || s.n_head != s.n_head_kv * R || s.page_size <= 0 ||
        n_tok > 0x7fffffff || (sel != nullptr && (count == nullptr || top_k <= 0 || blk <= 0)))
        return false;
    if (!xmx_available(XmxType::f16)) return false;
    auto& queue = core::Runtime::get().stream(stream);
    if (queue.get_device().get_info<sycl::info::device::local_mem_size>() < local_bytes()) return false;
    const float scale_log2 = (1.0f / std::sqrt((float) L)) * LOG2E;
    const auto e = queue.submit([&](sycl::handler& h) {
        MmXmx k{q, k_pool, v_pool, page_table, step, sel, count, step_stride, top_k, blk, s.n_head, s.n_head_kv,
                s.page_size, scale_log2, attn,
                sycl::local_accessor<half, 1>(sycl::range<1>(2 * R * QS), h),
                sycl::local_accessor<half, 1>(sycl::range<1>(CH * KR), h),
                sycl::local_accessor<half, 1>(sycl::range<1>(CH * KR), h),
                sycl::local_accessor<long long, 1>(sycl::range<1>(CH), h),
                sycl::local_accessor<float, 3>(sycl::range<3>(NG, R, CH), h),
                sycl::local_accessor<half, 3>(sycl::range<3>(2, R, CH), h),
                sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                sycl::local_accessor<float, 1>(sycl::range<1>(R), h),
                sycl::local_accessor<float, 1>(sycl::range<1>(R), h)};
        h.parallel_for(sycl::nd_range<2>({(size_t) n_tok, (size_t) s.n_head_kv * WG}, {1, (size_t) WG}), k);
    });
    if (!stream) core::Runtime::get().wait(e, "mm_msa_attn_xmx");
    return true;
}

}  // namespace strata::kernels
