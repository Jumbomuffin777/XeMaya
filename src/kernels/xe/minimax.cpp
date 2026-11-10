// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/kernels/xe/minimax.cpp - MiniMax-M3's sparse attention (MSA): the indexer's key cache, the block selection and
// the attention over the selected blocks in FP32.  See include/strata/kernels/minimax.hpp.
#include "strata/kernels/minimax.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/core/runtime.hpp"
#define STRATA_DEVICE_ARCH_TAG_TU xe_minimax
#include "strata/kernels/device_arch_tag.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::kernels {
namespace {

sycl::queue& queue_for(void* stream) { return core::Runtime::get().stream(stream); }
void sync_if_needed(void* stream, const sycl::event& event, const char* what) {
    if (!stream) core::Runtime::get().wait(event, what);
}

constexpr int kMaxDim = 256;   // a head's dims at most (the private accumulator)

// The cell a query's slot i stands for, -1 none: every cell up to its own (sel null), or the cells of its blocks
struct Cells {
    const int32_t* sel;
    int64_t p;
    int blk;
    int64_t n;   // slots
    int64_t at(int64_t i) const {
        if (sel == nullptr) return i;
        const int64_t c = (int64_t) sel[i / blk] * blk + i % blk;
        return c <= p ? c : -1;
    }
};

}  // namespace

void mm_idx_append(uint16_t* cache, const int32_t* step, int step_stride, const float* ik, int dim, int n_tok,
                   void* stream) {
    if (n_tok <= 0) return;
    if (dim <= 0) throw core::DeviceError("mm_idx_append: dim " + std::to_string(dim));
    const auto event = queue_for(stream).parallel_for(
        sycl::range<2>((size_t) n_tok, (size_t) dim), [=](sycl::id<2> id) {
            const int64_t t = (int64_t) id[0], d = (int64_t) id[1];
            const int64_t p = step[t * step_stride + kStepPos];
            if (p >= 0) cache[p * dim + d] = f16_from_f32(ik[t * dim + d]);
        });
    sync_if_needed(stream, event, "mm_idx_append");
}

void mm_msa_select(const float* iq, const uint16_t* cache, const int32_t* step, int step_stride, int n_tok, int n_kv,
                   int dim, int blk, int top_k, int local, int64_t max_blocks, float* scores, int32_t* sel,
                   int32_t* count, void* stream) {
    if (n_tok <= 0) return;
    if (n_kv <= 0 || dim <= 0 || blk <= 0 || top_k <= 0 || local < 0 || max_blocks <= 0)
        throw core::DeviceError("mm_msa_select: the shape");
    auto& q = queue_for(stream);
    // a block's best visible cell, for the tokens that see more blocks than they take
    q.parallel_for(sycl::range<3>((size_t) n_tok, (size_t) n_kv, (size_t) max_blocks), [=](sycl::id<3> id) {
        const int64_t t = (int64_t) id[0], h = (int64_t) id[1], b = (int64_t) id[2];
        const int64_t p = step[t * step_stride + kStepPos];
        const int64_t vis = p / blk + 1;
        if (vis <= top_k || b >= vis) return;
        const float* qr = iq + (t * n_kv + h) * dim;
        const int64_t c1 = sycl::min<int64_t>((b + 1) * blk - 1, p);
        float best = -INFINITY;
        for (int64_t c = b * blk; c <= c1; ++c) {
            const uint16_t* kr = cache + c * dim;
            float d = 0.0f;
            for (int i = 0; i < dim; ++i) d += qr[i] * f32_from_f16(kr[i]);
            best = sycl::fmax(best, d);
        }
        scores[(t * n_kv + h) * max_blocks + b] = best;
    });
    const auto event = q.parallel_for(sycl::range<1>((size_t) n_tok * (size_t) n_kv), [=](sycl::id<1> id) {
        const int64_t r = (int64_t) id[0], t = r / n_kv;
        const int64_t p = step[t * step_stride + kStepPos];
        const int64_t vis = sycl::min<int64_t>(p / blk + 1, max_blocks);
        int32_t* out = sel + r * top_k;
        if (vis <= top_k) {
            for (int64_t i = 0; i < vis; ++i) out[i] = (int32_t) i;
            count[r] = (int32_t) vis;
            return;
        }
        float* sc = scores + r * max_blocks;
        int n = 0;
        for (int l = 0; l < local && l < top_k && vis - 1 - l >= 0; ++l) {   // the local blocks: always
            out[n++] = (int32_t) (vis - 1 - l);
            sc[vis - 1 - l] = -INFINITY;
        }
        for (; n < top_k; ++n) {   // the best of the rest (a taken block's score is -inf from then on)
            float best = -INFINITY;
            int64_t bi = -1;
            for (int64_t b = 0; b < vis; ++b)
                if (sc[b] > best) {
                    best = sc[b];
                    bi = b;
                }
            if (bi < 0) break;
            out[n] = (int32_t) bi;
            sc[bi] = -INFINITY;
        }
        for (int i = 1; i < n; ++i) {   // ascending
            const int32_t v = out[i];
            int j = i - 1;
            for (; j >= 0 && out[j] > v; --j) out[j + 1] = out[j];
            out[j + 1] = v;
        }
        count[r] = n;
    });
    sync_if_needed(stream, event, "mm_msa_select");
}

uint64_t mm_msa_attn_scratch_floats(int splits, const QsaShapes& s) {
    return splits > 1 ? (uint64_t) s.n_head * (uint64_t) splits * (uint64_t) (s.head_dim + 2) : 0;
}

void mm_msa_attn(const float* q, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                 const int32_t* step, int step_stride, const int32_t* sel, const int32_t* count, int top_k, int blk,
                 const QsaShapes& s, int splits, float* scratch, float* attn, int64_t n_tok, void* stream) {
    if (n_tok <= 0) return;
    const int64_t NH = s.n_head, NKV = s.n_head_kv, HD = s.head_dim, PS = s.page_size;
    if (NH <= 0 || NKV <= 0 || NH % NKV != 0 || HD <= 0 || HD > kMaxDim || PS <= 0 || splits < 1 ||
        (splits > 1 && scratch == nullptr) || (sel != nullptr && (count == nullptr || top_k <= 0 || blk <= 0)))
        throw core::DeviceError("mm_msa_attn: the shape");
    const int64_t G = NH / NKV, S = splits, W = HD + 2;
    const float scale = 1.0f / std::sqrt((float) HD);
    auto& qu = queue_for(stream);
    // a part of a query's cells with an online softmax: its largest logit, the sum of exp(logit - largest) and the
    // values' sum with those weights
    const auto ev = qu.parallel_for(sycl::range<3>((size_t) n_tok, (size_t) NH, (size_t) S), [=](sycl::id<3> id) {
        const int64_t t = (int64_t) id[0], h = (int64_t) id[1], sp = (int64_t) id[2], hk = h / G;
        const int64_t p = step[t * step_stride + kStepPos];
        Cells cells{sel != nullptr ? sel + (t * NKV + hk) * top_k : nullptr, p, blk,
                    sel != nullptr ? (int64_t) count[t * NKV + hk] * blk : p + 1};
        const int64_t i0 = cells.n * sp / S, i1 = cells.n * (sp + 1) / S;
        const float* qr = q + (t * NH + h) * HD;
        float acc[kMaxDim];
        for (int d = 0; d < HD; ++d) acc[d] = 0.0f;
        float m = -INFINITY, l = 0.0f;
        for (int64_t i = i0; i < i1; ++i) {
            const int64_t c = cells.at(i);
            if (c < 0) continue;
            const int64_t row = (((int64_t) page_table[c / PS] * NKV + hk) * PS + c % PS) * HD;
            const uint16_t *kr = k_pool + row, *vr = v_pool + row;
            float x = 0.0f;
            for (int d = 0; d < HD; ++d) x += qr[d] * f32_from_f16(kr[d]);
            x *= scale;
            float w = 1.0f;
            if (x > m) {   // the reference moves: what is summed so far scales down
                const float a = m == -INFINITY ? 0.0f : sycl::exp(m - x);
                l *= a;
                for (int d = 0; d < HD; ++d) acc[d] *= a;
                m = x;
            } else {
                w = sycl::exp(x - m);
            }
            l += w;
            for (int d = 0; d < HD; ++d) acc[d] += w * f32_from_f16(vr[d]);
        }
        if (S == 1) {
            float* o = attn + (t * NH + h) * HD;
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            for (int d = 0; d < HD; ++d) o[d] = acc[d] * inv;
            return;
        }
        float* o = scratch + ((t * NH + h) * S + sp) * W;
        o[0] = m;
        o[1] = l;
        for (int d = 0; d < HD; ++d) o[2 + d] = acc[d];
    });
    if (S == 1) {
        sync_if_needed(stream, ev, "mm_msa_attn");
        return;
    }
    const auto event = qu.parallel_for(sycl::range<2>((size_t) (n_tok * NH), (size_t) HD), [=](sycl::id<2> id) {
        const int64_t r = (int64_t) id[0], d = (int64_t) id[1];
        const float* part = scratch + r * S * W;
        float m = -INFINITY;
        for (int64_t sp = 0; sp < S; ++sp) m = sycl::fmax(m, part[sp * W]);
        float l = 0.0f, v = 0.0f;
        for (int64_t sp = 0; sp < S; ++sp) {
            const float pm = part[sp * W];
            if (pm == -INFINITY) continue;
            const float a = sycl::exp(pm - m);
            l += a * part[sp * W + 1];
            v += a * part[sp * W + 2 + d];
        }
        attn[r * HD + d] = l > 0.0f ? v / l : 0.0f;
    });
    sync_if_needed(stream, event, "mm_msa_attn");
}

}  // namespace strata::kernels
