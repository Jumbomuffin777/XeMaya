// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// src/core/plain_ds41_rope.hpp - a DeepSeek-V4.1-Flash layer's rotation, for the block's windows (plain_ds41.cpp) and
// its prompt chunks (src/prefill/plain_prompt_ds41.cpp).  After llama.cpp's src/models/deepseek41.cpp at master
// 8642541b2; MIT License, Copyright (c) 2023-2026 The ggml authors, see third_party/main/ggml/LICENSE.
#pragma once

#include "strata/artifact/model_spec.hpp"
#include "strata/kernels/ds41.hpp"

#include <cmath>
#include <cstdint>

namespace strata::core {

/// The layer's rotation (deepseek41.cpp: a compressed layer takes the compressed base and the YaRN scaling), over the
/// last rope_dims of a row of `dim`
inline strata::kernels::Ds41Rope ds41_layer_rope(const ModelSpec& s, int64_t l, int64_t dim) {
    strata::kernels::Ds41Rope r;
    r.n_rot = (int) (s.rope_dims > 0 ? s.rope_dims : 64);
    r.offset = (int) (dim - r.n_rot);
    const bool c = s.compress_ratios[(size_t) l] != 0;
    const bool yarn = s.rope_scaling_type == "yarn";
    r.freq_base = (float) (c ? s.compress_rope_base : s.rope_base);
    r.freq_scale = c && yarn && s.rope_scaling_factor > 0 ? (float) (1.0 / s.rope_scaling_factor) : 1.0f;
    r.ext_factor = c && yarn ? 1.0f : 0.0f;
    r.attn_factor = r.ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * std::log(1.0f / r.freq_scale));
    r.beta_fast = c ? 32.0f : 0.0f;
    r.beta_slow = c ? 1.0f : 0.0f;
    r.n_ctx_orig = c ? (int) s.rope_scaling_orig_ctx : 0;
    return r;
}

}  // namespace strata::core
