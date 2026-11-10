// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/kernels/swiglu_clamp.hpp - the routed experts' SwiGLU limit.
//
// GLM-5.3-Flash clamps its experts' SwiGLU (glm5-next.swiglu_clamp_exp, ggml_swiglu_clamp): with a limit L > 0 the gate
// is cut above at L and the up value to [-L, L] before silu(gate) * up.  Every other model has none (0).  A process
// setting, as the GDN output gate (gdn_set_silu_gate): set at load, before any graph is recorded; the GPU kernels read
// it when they are launched, so a recorded graph keeps the value it was recorded with.
//
// XeMaya: MiniMax-M3's experts take llama.cpp's swiglu-oai instead (ggml_swiglu_oai, LLM_FFN_SWIGLU_OAI_MOE: alpha
// 1.702, limit 7): x = min(gate, L), y = clamp(up, -L, L), out = x sigmoid(alpha x) (y + 1).  The setting holds it as
// a limit below 0 (expert_swiglu_set_oai), and swiglu_clamp then hands the kernels' silu(gate) * up the pair
// (alpha x, (y + 1) / alpha), whose product is that value: no kernel changes its expression.
#pragma once

#include <atomic>
#include <cmath>

namespace strata::kernels {

inline std::atomic<float> g_expert_swiglu_limit{0.0f};

inline void expert_swiglu_set_limit(float limit) {
    g_expert_swiglu_limit.store(limit > 0.0f ? limit : 0.0f, std::memory_order_relaxed);
}
/// swiglu-oai with this limit (above 0) for the routed experts, in place of the clamped SwiGLU.
inline void expert_swiglu_set_oai(float limit) {
    g_expert_swiglu_limit.store(limit > 0.0f ? -limit : 0.0f, std::memory_order_relaxed);
}
/// The setting: above 0 the clamp's limit, below 0 swiglu-oai's (negated), 0 neither.
inline float expert_swiglu_limit() { return g_expert_swiglu_limit.load(std::memory_order_relaxed); }

inline constexpr float kSwigluOaiAlpha = 1.702f;   ///< llama.cpp's (gpt-oss's) constants: build_ffn, build_moe_ffn
inline constexpr float kSwigluOaiLimit = 7.0f;

/// gate <= lim, -lim <= up <= lim when lim > 0; when lim < 0 the pair whose silu(gate) * up is swiglu-oai's value at
/// the limit -lim (host and device).
inline void swiglu_clamp(float& gate, float& up, float lim) {
    if (lim > 0.0f) {
        gate = gate < lim ? gate : lim;
        up = up < -lim ? -lim : (up > lim ? lim : up);
    } else if (lim < 0.0f) {
        const float l = -lim;
        const float x = gate < l ? gate : l;
        const float y = up < -l ? -l : (up > l ? l : up);
        gate = kSwigluOaiAlpha * x;
        up = (y + 1.0f) / kSwigluOaiAlpha;
    }
}

/// The CPU kernels' silu(gate) * up after the clamp: with lim 0, the expression they always used.
inline float swiglu_host(float gate, float up, float lim) {
    swiglu_clamp(gate, up, lim);
    return (gate / (1.f + std::exp(-gate))) * up;
}

}  // namespace strata::kernels
