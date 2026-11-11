// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/draft_head.hpp - a draft head's coarse pass: the vocabulary's rows at 2 bits, the few best
// of their products chosen, and those rows taken again from the head itself (the idea of vllm-radiance's int2 draft
// head with an exact rerank, written anew).  A draft only proposes; the verify window decides with the model's own
// head, so a draft head may be approximate: what it costs is acceptance.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The int2 form's group: 128 values with their minimum and their step in FP16.
constexpr int kInt2Group = 128;

/// The bytes of one row of `n_in` values (a positive multiple of kInt2Group): the codes, 4 a byte, then each group's
/// minimum and step.
std::size_t int2_row_bytes(int n_in);

/// Host: `n_rows` rows of `n_in` floats to the int2 form, each group's minimum and step from its least and largest
/// value (rounded to the nearest of the four levels).
void int2_pack_rows(const float* rows, int64_t n_rows, int n_in, uint8_t* out);

/// y[r] = the int2 row r of `w` . x for the `n_out` rows (`x`: `n_in` floats).  Device pointers.
void int2_gemv(const uint8_t* w, const float* x, float* y, int n_in, int64_t n_out, void* stream);

// XeMaya: the same rows in bit planes.  A row's codes are stored a plane at a time (the low bits of its n_in / 32
// spans of 32 values as one 32-bit word each, then the high bits), and the activations are cut to kInt2XBits bits a
// span (their least value and step, the span's sum beside them): a row's product is then popcounts of a code plane
// and an activation plane, 2 * kInt2XBits a span, instead of a multiply a value.  The activations' rounding adds
// about 1e-3 of the 2-bit rows' own error to the products (a 32-value span at 6 bits against a 128-value group at 2).

/// The bits an activation keeps in the bit-plane form.
constexpr int kInt2XBits = 6;

/// Host: `n_rows` rows of int2_pack_rows' form to the bit-plane form, in place (the same bytes a row).
void int2_rows_to_planes(uint8_t* rows, int64_t n_rows, int n_in);

/// The bytes int2_x_planes writes for `n_in` activations.
std::size_t int2_x_planes_bytes(int n_in);

/// The activations `x` (`n_in` floats) cut to their bit planes, into `xp`.  Device pointers.
void int2_x_planes(const float* x, void* xp, int n_in, void* stream);

/// y[r] = the bit-plane row r of `w` . the activations of `xp` (int2_x_planes') for the `n_out` rows.
void int2_planes_gemv(const uint8_t* w, const void* xp, float* y, int n_in, int64_t n_out, void* stream);

/// The largest topk_max values a call can choose.
constexpr int kTopKMax = 64;

/// The scratch topk needs for `n` values.
std::size_t topk_scratch_bytes(int64_t n, int k);

/// The indices of the `k` (1..kTopKMax) largest of the `n` values in `v`, in no order, into `ids`.  Device pointers;
/// `scratch` holds topk_scratch_bytes(n, k).
void topk(const float* v, int64_t n, int k, int32_t* ids, void* scratch, void* stream);

/// y[0..n) = -1.7e38 (a probability of 0) when the device value *pos is `from` or more, else nothing: a decision the
/// device makes, so a captured graph takes it at every replay.
void int2_pad_from(const int32_t* pos, int32_t from, float* y, int64_t n, void* stream);

/// y[ids[j]] = vals[j] for the `k` values (the chosen rows' products written back over their coarse ones).
void scatter_values(const float* vals, const int32_t* ids, int k, float* y, void* stream);

}  // namespace strata::kernels
