// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/artifact/model_spec.hpp - a model's shape, read from its GGUF metadata.
//
// The engine was built for one model, and `ModelGeometry` (core/layout.hpp) holds that model's numbers as
// defaults.  This reads the shape of every architecture XeStrata is being taught, by the key names llama.cpp
// writes (`src/llama-arch.cpp` at de7fa0a3), so that a second model is a second set of read values and not a
// second set of constants:
//
//   qwen4exp   Qwen3.8-Flash-Next and its fine-tunes (the engine's model)
//   qwen35moe  Qwen3.5 MoE (Ornith-1.5-35B-A3B): gated delta-net + full attention every 4th layer, MTP
//   qwen35     Qwen3.5 (Qwen3.5-9B): qwen35moe's block with a dense SwiGLU FFN in place of the MoE, MTP
//   qwen3moe   Qwen3 MoE (llm-jp-4.1-32b-a3b): full attention only, no shared expert
//   glm5-next  GLM-5.3-Flash: KDA + NoPE MLA with a k-pool indexer, mHC, leading dense layers, MTP.  Also spelled
//              `glm5next` (Project Maya's quants, after unsloth's first GGUFs), with the same keys under that name.
//   deepseek41 DeepSeek-V4.1-Flash (XeMaya): MQA over compressed KV with a lightning indexer (CSA/HCA), a sliding
//              window, grouped low-rank output, mHC, sqrt-softplus routing, Engram n-gram tables.  The keys and their
//              checks follow llama.cpp's src/models/deepseek4.cpp and deepseek41.cpp (master 8642541b2; MIT License,
//              Copyright (c) 2023-2026 The ggml authors).
//
// `block_count` counts the MTP (NextN) layers too; `n_layer` here is the trunk, block_count minus
// `nextn_predict_layers`, as llama.cpp's `n_layer()`.  A key a model does not carry reads as 0 (or false), and
// the keys an architecture cannot run without are required: a missing one is an error naming the key.
//
// Header-only, like gguf_reader.hpp, and with no ggml or SYCL dependency.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata {

enum class ModelArch { Qwen4Exp, Qwen35Moe, Qwen3Moe, Glm5Next, Qwen35, DeepSeek41, MiniMaxM3 };

/// llama.cpp's `llama_expert_gating_func_type`
enum class ExpertGating : uint32_t { None = 0, Softmax = 1, Sigmoid = 2, SoftmaxWeight = 3, SqrtSoftplus = 4 };

struct ModelSpec {
    ModelArch arch = ModelArch::Qwen4Exp;
    std::string arch_name;                ///< general.architecture

    int64_t n_layer = 0;                  ///< trunk layers (block_count - nextn_predict_layers)
    int64_t n_nextn = 0;                  ///< MTP layers after the trunk
    int64_t n_embd = 0;
    int64_t context_length = 0;
    double rms_eps = 0;
    double norm_eps = 0;                  ///< attention.layer_norm_epsilon: the LayerNorms' (GLM-5.3-Flash's indexer keys)

    // attention.  head_count_kv may be per layer (GLM-5.3-Flash writes 0 on its recurrent layers).
    int64_t n_head = 0;
    std::vector<int64_t> n_head_kv;       ///< per layer, block_count entries
    int64_t key_length = 0, value_length = 0;
    int64_t rope_dims = 0;                ///< rotated dimensions of a head (llama.cpp's default: the whole key)
    std::vector<int64_t> rope_sections;   ///< MRoPE sections, empty when not written
    double rope_base = 0;
    std::string rope_scaling_type;
    double rope_scaling_factor = 0, rope_scaling_orig_ctx = 0;

    // linear attention.  `recurrent` is per layer: true for a gated delta-net / KDA layer.
    std::vector<bool> recurrent;          ///< block_count entries; the MTP layers are false
    int64_t full_attention_interval = 0;
    int64_t ssm_d_conv = 0, ssm_d_inner = 0, ssm_d_state = 0;
    int64_t ssm_dt_rank = 0;              ///< the value heads of a gated delta-net
    int64_t ssm_n_group = 0;              ///< its key heads
    int64_t kda_head_dim = 0;
    double kda_gate_lower_bound = 0;

    // MoE
    int64_t n_expert = 0, n_expert_used = 0;
    int64_t n_ff_exp = 0;                 ///< a routed expert's width
    int64_t n_ff_shexp = 0;               ///< the shared expert's width (0: none)
    int64_t n_expert_shared = 0;
    int64_t n_dense_lead = 0;             ///< leading layers with a dense FFN instead of the MoE
    int64_t n_ff = 0;                     ///< the dense FFN's width
    ExpertGating gating = ExpertGating::None;
    bool expert_weights_norm = false;
    double expert_weights_scale = 0;
    std::vector<double> swiglu_clamp_exp, swiglu_clamp_shexp;   ///< per layer, empty when not written

    // hyper-connections (qwen4exp's gated residual, GLM-5.3-Flash's mHC)
    int64_t hc = 0, hc_low_rank = 0, hc_sinkhorn_iters = 0;
    double hc_eps = 0;

    // MLA and the sparse indexer
    int64_t q_lora_rank = 0, kv_lora_rank = 0, key_length_mla = 0, value_length_mla = 0;
    int64_t idx_n_head = 0, idx_key_length = 0, idx_top_k = 0, idx_kpool = 0;
    bool idx_kpool_select_tail = false;
    // MiniMax-M3 (minimax-m3): MiniMax Sparse Attention - the indexer scores the positions, a block of
    // idx_block_size positions takes its best position's score, and each K/V head attends the idx_top_k best blocks
    // (the idx_local_blocks blocks ending at the query's own always among them)
    int64_t idx_block_size = 0, idx_local_blocks = 0;

    // DeepSeek-V4.1-Flash (deepseek41)
    int64_t sliding_window = 0;
    int64_t o_group_count = 0, o_lora_rank = 0;   ///< the grouped low-rank attention output
    double compress_rope_base = 0;                ///< RoPE base of the compressed KV
    int64_t hash_layer_count = 0;                 ///< leading hash-routed layers (V4.1: none)
    std::vector<int64_t> compress_ratios;         ///< per trunk layer: 0 none, 1 or 2 (llama.cpp's dsv4_compress_ratios)
    std::vector<int64_t> kv_src, topk_src;        ///< per trunk layer: the layer whose compressed KV / top-k it reads, -1
    int64_t cand_src = -1, cand_topk_blocks = 0, cand_block_size = 0;   ///< the indexer's candidate blocks
    // Engram (deepseek41.ple.*): one n-gram hash module per Engram layer, all tables in one per_layer_token_embd
    std::vector<int64_t> engram_layers;           ///< trunk layers with a module, ascending
    int64_t engram_ngram = 0, engram_heads_per_ngram = 0, engram_head_dim = 0;
    int64_t engram_n_heads = 0;                   ///< columns (hash heads) of one module: (ngram - 1) * heads_per_ngram
    std::vector<uint64_t> engram_multipliers;     ///< modules x ngram
    std::vector<uint64_t> engram_offsets, engram_vocab;   ///< modules x n_heads: each head's first row and row count
    int64_t engram_pad_id = 0;
    std::vector<int32_t> engram_token_map;        ///< token id -> Engram token id

    /// The Engram module of trunk layer `l`, -1 when it has none.
    int64_t engram_index(int64_t l) const {
        for (size_t i = 0; i < engram_layers.size(); ++i)
            if (engram_layers[i] == l) return (int64_t) i;
        return -1;
    }

    /// True when trunk layer `l` routes experts (the MTP layers are not counted here).
    bool layer_routes(int64_t l) const { return n_expert > 0 && l >= n_dense_lead && l < n_layer; }
    /// Trunk layers that route experts.
    int64_t n_moe_layers() const { return n_expert > 0 && n_layer > n_dense_lead ? n_layer - n_dense_lead : 0; }
    /// The 0-based index of trunk layer `l` among the layers that route, -1 when it does not route.  An expert's
    /// number in the cache is `moe_index(l) * n_expert + e`, which is `l * n_expert + e` when every layer routes.
    int64_t moe_index(int64_t l) const { return layer_routes(l) ? l - n_dense_lead : -1; }
};

namespace detail {

inline const MetaValue* spec_get(const GgufFile& g, const std::string& arch, const char* key) {
    return g.get(arch + "." + key);
}
inline int64_t spec_int(const GgufFile& g, const std::string& arch, const char* key, int64_t dflt = 0) {
    const MetaValue* v = spec_get(g, arch, key);
    if (!v || !v->is_num()) return dflt;
    return (int64_t) v->u;
}
inline double spec_num(const GgufFile& g, const std::string& arch, const char* key, double dflt = 0) {
    const MetaValue* v = spec_get(g, arch, key);
    return v && v->is_num() ? v->num() : dflt;
}
/// A key written either as one number or as one number per layer (llama.cpp's get_key_or_arr).
inline bool spec_per_layer(const GgufFile& g, const std::string& arch, const char* key, int64_t n,
                           std::vector<double>& out) {
    out.clear();
    const MetaValue* v = spec_get(g, arch, key);
    if (!v) return false;
    if (v->type == MetaType::ARRAY) {
        if ((int64_t) v->items.size() != n) return false;
        for (const MetaValue& e : v->items) out.push_back(e.num());
    } else if (v->is_num()) {
        out.assign((size_t) n, v->num());
    } else {
        return false;
    }
    return true;
}

}  // namespace detail

namespace detail {

inline bool spec_uarr(const GgufFile& g, const std::string& arch, const char* key, std::vector<uint64_t>& out) {
    out.clear();
    const MetaValue* v = spec_get(g, arch, key);
    if (!v || v->type != MetaType::ARRAY) return false;
    for (const MetaValue& e : v->items) out.push_back(e.u);
    return true;
}

}  // namespace detail

/// DeepSeek-V4.1-Flash's own keys and their consistency, as llama.cpp's deepseek4.cpp and deepseek41.cpp check them
/// (load_arch_hparams).  Called by read_model_spec after the common keys.
inline bool read_deepseek41(const GgufFile& g, ModelSpec& s, std::string& err) {
    using namespace detail;
    const std::string& A = s.arch_name;
    const char* required[] = {"attention.q_lora_rank", "attention.sliding_window", "attention.output_group_count",
                              "attention.output_lora_rank", "attention.compress_rope_freq_base",
                              "attention.compress_ratios", "attention.indexer.head_count",
                              "attention.indexer.key_length", "attention.indexer.top_k",
                              "attention.kv_source_layers", "attention.indexer.source_layers",
                              "hyper_connection.count", "hyper_connection.sinkhorn_iterations",
                              "expert_shared_count", "expert_weights_scale"};
    for (const char* k : required)
        if (!spec_get(g, A, k)) { err = "missing " + A + "." + k; return false; }
    if (s.n_nextn != 0) {
        err = A + ": DeepSeek-V4.1 has no MTP layer in the model (its draft is a separate DSpark file)";
        return false;
    }
    if (s.n_dense_lead != 0) { err = A + ": leading dense layers are not part of DeepSeek-V4.1"; return false; }
    if (s.idx_top_k <= 0) { err = A + ": attention.indexer.top_k must be above 0"; return false; }
    if (s.hc != 4) { err = A + ".hyper_connection.count " + std::to_string(s.hc) + ", only 4 is supported"; return false; }
    s.sliding_window = spec_int(g, A, "attention.sliding_window");
    s.o_group_count = spec_int(g, A, "attention.output_group_count");
    s.o_lora_rank = spec_int(g, A, "attention.output_lora_rank");
    s.compress_rope_base = spec_num(g, A, "attention.compress_rope_freq_base");
    s.hash_layer_count = spec_int(g, A, "hash_layer_count");
    if (s.n_ff_shexp == 0) s.n_ff_shexp = s.n_ff_exp * s.n_expert_shared;   // deepseek41.cpp: n_ff_exp per shared expert
    if (s.sliding_window <= 0 || s.o_group_count <= 0 || s.o_lora_rank <= 0 || s.q_lora_rank <= 0 ||
        (s.n_head * s.key_length) % s.o_group_count != 0) {
        err = A + ": sliding_window, output_group_count (dividing head_count x key_length), output_lora_rank and "
                  "q_lora_rank must be above 0";
        return false;
    }
    if (s.hash_layer_count != 0) {
        err = A + ".hash_layer_count " + std::to_string(s.hash_layer_count) + ": hash-routed layers are not supported";
        return false;
    }

    // compress ratios, and the source of each compressed layer's KV and top-k (deepseek41.cpp)
    std::vector<uint64_t> ratios, kv_src, idx_src;
    spec_uarr(g, A, "attention.compress_ratios", ratios);
    spec_uarr(g, A, "attention.kv_source_layers", kv_src);
    spec_uarr(g, A, "attention.indexer.source_layers", idx_src);
    const int64_t n = s.n_layer;
    if ((int64_t) ratios.size() < n) {
        err = A + ".attention.compress_ratios is shorter than block_count";
        return false;
    }
    auto has = [](const std::vector<uint64_t>& v, int64_t x) { return std::find(v.begin(), v.end(), (uint64_t) x) != v.end(); };
    for (uint64_t x : kv_src) if ((int64_t) x >= n) { err = A + ": kv source layer out of range"; return false; }
    for (uint64_t x : idx_src) if ((int64_t) x >= n) { err = A + ": index source layer out of range"; return false; }
    s.compress_ratios.assign(ratios.begin(), ratios.begin() + n);
    s.kv_src.assign((size_t) n, -1);
    s.topk_src.assign((size_t) n, -1);
    int64_t last_kv = -1, last_idx = -1;
    for (int64_t l = 0; l < n; ++l) {
        const int64_t r = s.compress_ratios[(size_t) l];
        const bool is_kv = has(kv_src, l), is_idx = has(idx_src, l);
        if ((is_kv || is_idx) && r == 0) {
            err = A + ": source layer " + std::to_string(l) + " has compress ratio 0";
            return false;
        }
        if (is_kv) last_kv = l;
        if (is_idx) last_idx = l;
        if (r == 0) continue;
        if (r > 2 || last_kv < 0 || last_idx < last_kv || s.compress_ratios[(size_t) last_kv] != r ||
            s.compress_ratios[(size_t) last_idx] != r) {
            err = A + ": layer " + std::to_string(l) + ": ratio " + std::to_string(r) + " without a matching kv/index source";
            return false;
        }
        s.kv_src[(size_t) l] = last_kv;
        s.topk_src[(size_t) l] = last_idx;
    }
    s.cand_src = spec_get(g, A, "attention.indexer.candidate_source_layer")
                     ? spec_int(g, A, "attention.indexer.candidate_source_layer") : -1;
    if (s.cand_src >= 0) {
        s.cand_topk_blocks = spec_int(g, A, "attention.indexer.candidate_top_k_blocks");
        s.cand_block_size = spec_int(g, A, "attention.indexer.candidate_block_size");
        const int64_t c = s.cand_src;
        if (c >= n || s.topk_src[(size_t) c] != c || s.cand_block_size <= 0 || s.cand_topk_blocks <= 0) {
            err = A + ": invalid candidate selection (source must be an index source, block size and top-k blocks > 0)";
            return false;
        }
        for (int64_t l = c + 1; l < n; ++l)
            if (s.topk_src[(size_t) l] == l && s.compress_ratios[(size_t) l] != s.compress_ratios[(size_t) c]) {
                err = A + ": candidate consumers must share the source's compress ratio";
                return false;
            }
    }

    // Engram: the PLE n-gram hash with one set of constants per Engram layer
    std::vector<uint64_t> layers;
    if (spec_uarr(g, A, "ple.layers", layers) && !layers.empty()) {
        s.engram_ngram = spec_int(g, A, "ple.ngram_size");
        s.engram_heads_per_ngram = spec_int(g, A, "ple.heads_per_ngram");
        s.engram_head_dim = spec_int(g, A, "embedding_length_per_layer_input");
        s.engram_pad_id = spec_int(g, A, "engram.pad_token_id");
        spec_uarr(g, A, "ple.layer_multipliers", s.engram_multipliers);
        spec_uarr(g, A, "ple.head_offsets", s.engram_offsets);
        spec_uarr(g, A, "ple.head_vocab_sizes", s.engram_vocab);
        std::vector<uint64_t> map;
        spec_uarr(g, A, "engram.token_map", map);
        const size_t ne = layers.size();
        s.engram_n_heads = s.engram_ngram >= 2 ? (s.engram_ngram - 1) * s.engram_heads_per_ngram : 0;
        const size_t nc = (size_t) s.engram_n_heads;
        if (nc == 0 || s.engram_head_dim <= 0 || s.engram_multipliers.size() != ne * (size_t) s.engram_ngram ||
            s.engram_vocab.size() != ne * nc || s.engram_offsets.size() != ne * nc || s.engram_pad_id < 0 ||
            (size_t) s.engram_pad_id >= map.size()) {
            err = A + ": the Engram metadata is inconsistent";
            return false;
        }
        for (size_t e = 0; e < ne; ++e) {
            if ((int64_t) layers[e] >= n || (e > 0 && layers[e] <= layers[e - 1])) {
                err = A + ": Engram layer " + std::to_string(layers[e]) + " is out of range or out of order";
                return false;
            }
            s.engram_layers.push_back((int64_t) layers[e]);
        }
        for (size_t c = 0; c < ne * nc; ++c)
            if (s.engram_vocab[c] < 2 || s.engram_offsets[c] + s.engram_vocab[c] > (uint64_t) INT32_MAX) {
                err = A + ": Engram head " + std::to_string(c) + ": bucket range out of range";
                return false;
            }
        s.engram_token_map.reserve(map.size());
        for (uint64_t t : map) s.engram_token_map.push_back((int32_t) (uint32_t) t);
    }
    return true;
}

/// Reads the shape of the model `g` holds (its first shard, which carries the metadata).  Returns false with a
/// reason naming the key or value at fault.
inline bool read_model_spec(const GgufFile& g, ModelSpec& s, std::string& err) {
    using namespace detail;
    s = ModelSpec{};
    const MetaValue* a = g.get("general.architecture");
    if (!a || a->type != MetaType::STRING) { err = "missing general.architecture"; return false; }
    s.arch_name = a->s;
    const std::string& A = s.arch_name;
    if (A == "qwen4exp") s.arch = ModelArch::Qwen4Exp;
    else if (A == "qwen35moe") s.arch = ModelArch::Qwen35Moe;
    else if (A == "qwen35") s.arch = ModelArch::Qwen35;
    else if (A == "qwen3moe") s.arch = ModelArch::Qwen3Moe;
    else if (A == "glm5-next" || A == "glm5next") s.arch = ModelArch::Glm5Next;
    else if (A == "deepseek41") s.arch = ModelArch::DeepSeek41;
    else if (A == "minimax-m3") s.arch = ModelArch::MiniMaxM3;
    else {
        err = "architecture '" + A + "' is not supported (qwen4exp, qwen35moe, qwen35, qwen3moe, glm5-next, deepseek41, "
              "minimax-m3)";
        return false;
    }

    // the keys every architecture here needs; the experts' all but the dense qwen35
    const bool dense = s.arch == ModelArch::Qwen35;
    const char* required[] = {"block_count", "embedding_length", "attention.head_count", "expert_count",
                              "expert_used_count", "attention.layer_norm_rms_epsilon"};
    for (const char* k : required)
        if (!(dense && std::strncmp(k, "expert_", 7) == 0) && !spec_get(g, A, k)) {
            err = "missing " + A + "." + k;
            return false;
        }

    const int64_t n_all = spec_int(g, A, "block_count");
    s.n_nextn = spec_int(g, A, "nextn_predict_layers");
    if (n_all <= 0 || s.n_nextn < 0 || s.n_nextn >= n_all) {
        err = A + ".block_count " + std::to_string(n_all) + " with " + std::to_string(s.n_nextn) + " MTP layers";
        return false;
    }
    s.n_layer = n_all - s.n_nextn;
    s.n_embd = spec_int(g, A, "embedding_length");
    s.context_length = spec_int(g, A, "context_length");
    s.rms_eps = spec_num(g, A, "attention.layer_norm_rms_epsilon");
    s.norm_eps = spec_num(g, A, "attention.layer_norm_epsilon");

    s.n_head = spec_int(g, A, "attention.head_count");
    std::vector<double> kv;
    if (spec_get(g, A, "attention.head_count_kv") && !spec_per_layer(g, A, "attention.head_count_kv", n_all, kv)) {
        err = A + ".attention.head_count_kv is neither a number nor one per layer";
        return false;
    }
    s.n_head_kv.assign((size_t) n_all, 0);
    for (size_t i = 0; i < kv.size(); ++i) s.n_head_kv[i] = (int64_t) kv[i];
    // llama.cpp's defaults (llama_model::load_hparams): a head is n_embd / n_head wide unless the file says, and
    // rotated over its whole key unless the file says (qwen3moe writes no rope.dimension_count)
    const int64_t head_w = s.n_head > 0 ? s.n_embd / s.n_head : 0;
    s.key_length = spec_int(g, A, "attention.key_length", head_w);
    s.value_length = spec_int(g, A, "attention.value_length", head_w);
    s.rope_dims = spec_int(g, A, "rope.dimension_count", s.key_length);
    if (const MetaValue* v = spec_get(g, A, "rope.dimension_sections"); v && v->type == MetaType::ARRAY)
        for (const MetaValue& e : v->items) s.rope_sections.push_back((int64_t) e.u);
    s.rope_base = spec_num(g, A, "rope.freq_base");
    if (const MetaValue* v = spec_get(g, A, "rope.scaling.type"); v && v->type == MetaType::STRING)
        s.rope_scaling_type = v->s;
    s.rope_scaling_factor = spec_num(g, A, "rope.scaling.factor");
    s.rope_scaling_orig_ctx = spec_num(g, A, "rope.scaling.original_context_length");

    s.full_attention_interval = spec_int(g, A, "full_attention_interval");
    s.ssm_d_conv = spec_int(g, A, "ssm.conv_kernel");
    s.ssm_d_inner = spec_int(g, A, "ssm.inner_size");
    s.ssm_d_state = spec_int(g, A, "ssm.state_size");
    s.ssm_dt_rank = spec_int(g, A, "ssm.time_step_rank");
    s.ssm_n_group = spec_int(g, A, "ssm.group_count");
    s.kda_head_dim = spec_int(g, A, "kda.head_dim");
    s.kda_gate_lower_bound = spec_num(g, A, "kda.gate_lower_bound");

    // which layers are recurrent: llama.cpp's rule per architecture
    s.recurrent.assign((size_t) n_all, false);
    switch (s.arch) {
    case ModelArch::Qwen4Exp:
    case ModelArch::Qwen35Moe:
    case ModelArch::Qwen35: {
        // qwen35moe.cpp: attention.recurrent_layers when written, else every layer but each interval-th
        std::vector<double> r;
        if (spec_per_layer(g, A, "attention.recurrent_layers", n_all, r)) {
            for (int64_t i = 0; i < n_all; ++i) s.recurrent[(size_t) i] = i < s.n_layer && r[(size_t) i] != 0;
        } else {
            const int64_t iv = s.full_attention_interval > 0 ? s.full_attention_interval : 4;
            for (int64_t i = 0; i < s.n_layer; ++i) s.recurrent[(size_t) i] = (i + 1) % iv != 0;
        }
        if (s.ssm_d_inner <= 0 || s.ssm_d_state <= 0 || s.ssm_dt_rank <= 0 || s.ssm_n_group <= 0 || s.ssm_d_conv <= 0) {
            err = A + ": the gated delta-net needs ssm.conv_kernel, inner_size, state_size, time_step_rank and group_count";
            return false;
        }
        break;
    }
    case ModelArch::Glm5Next:
        // glm5-next.cpp: a layer without K/V heads is a KDA layer
        for (int64_t i = 0; i < s.n_layer; ++i) s.recurrent[(size_t) i] = s.n_head_kv[(size_t) i] == 0;
        break;
    case ModelArch::Qwen3Moe:
    case ModelArch::DeepSeek41:
    case ModelArch::MiniMaxM3:
        break;
    }

    s.n_expert = spec_int(g, A, "expert_count");
    s.n_expert_used = spec_int(g, A, "expert_used_count");
    s.n_ff_exp = spec_int(g, A, "expert_feed_forward_length");
    s.n_ff_shexp = spec_int(g, A, "expert_shared_feed_forward_length");
    s.n_expert_shared = spec_int(g, A, "expert_shared_count", s.n_ff_shexp > 0 ? 1 : 0);
    s.n_dense_lead = spec_int(g, A, "leading_dense_block_count");
    s.n_ff = spec_int(g, A, "feed_forward_length");
    s.gating = (ExpertGating) spec_int(g, A, "expert_gating_func");
    if (s.gating == ExpertGating::None)   // llama.cpp's defaults: GLM-5.3-Flash sigmoid, the Qwens softmax
        s.gating = s.arch == ModelArch::Glm5Next ? ExpertGating::Sigmoid : ExpertGating::Softmax;
    // deepseek4.cpp: DeepSeek-V4 scores its experts by sqrt(softplus(logit)) and nothing else
    if (s.arch == ModelArch::DeepSeek41 && s.gating != ExpertGating::SqrtSoftplus) {
        err = A + ".expert_gating_func " + std::to_string((uint32_t) s.gating) + ": DeepSeek-V4.1 needs 4 (sqrtsoftplus)";
        return false;
    }
    if (s.gating != ExpertGating::Softmax && s.gating != ExpertGating::Sigmoid &&
        !(s.arch == ModelArch::DeepSeek41 && s.gating == ExpertGating::SqrtSoftplus)) {
        err = A + ".expert_gating_func " + std::to_string((uint32_t) s.gating) + " is not supported";
        return false;
    }
    s.expert_weights_norm = spec_int(g, A, "expert_weights_norm") != 0;
    s.expert_weights_scale = spec_num(g, A, "expert_weights_scale");
    spec_per_layer(g, A, "swiglu_clamp_exp", n_all, s.swiglu_clamp_exp);
    spec_per_layer(g, A, "swiglu_clamp_shexp", n_all, s.swiglu_clamp_shexp);
    if (dense) {   // llama.cpp's qwen35: ffn_gate, ffn_up and ffn_down in every block, n_ff wide
        if (s.n_expert != 0 || s.n_ff <= 0) {
            err = A + ": expert_count " + std::to_string(s.n_expert) + ", feed_forward_length " + std::to_string(s.n_ff);
            return false;
        }
    } else if (s.n_expert <= 0 || s.n_expert_used <= 0 || s.n_expert_used > s.n_expert || s.n_ff_exp <= 0) {
        err = A + ": expert_count " + std::to_string(s.n_expert) + ", expert_used_count " +
              std::to_string(s.n_expert_used) + ", expert_feed_forward_length " + std::to_string(s.n_ff_exp);
        return false;
    }
    if (s.n_dense_lead < 0 || s.n_dense_lead >= s.n_layer || (s.n_dense_lead > 0 && s.n_ff <= 0)) {
        err = A + ".leading_dense_block_count " + std::to_string(s.n_dense_lead) + " with feed_forward_length " +
              std::to_string(s.n_ff);
        return false;
    }

    s.hc = spec_int(g, A, "hyper_connection.count");
    s.hc_low_rank = spec_int(g, A, "hyper_connection.low_rank");
    s.hc_sinkhorn_iters = spec_int(g, A, "hyper_connection.sinkhorn_iterations");
    s.hc_eps = spec_num(g, A, "hyper_connection.epsilon");

    s.q_lora_rank = spec_int(g, A, "attention.q_lora_rank");
    s.kv_lora_rank = spec_int(g, A, "attention.kv_lora_rank");
    s.key_length_mla = spec_int(g, A, "attention.key_length_mla");
    s.value_length_mla = spec_int(g, A, "attention.value_length_mla");
    s.idx_n_head = spec_int(g, A, "attention.indexer.head_count");
    s.idx_key_length = spec_int(g, A, "attention.indexer.key_length");
    s.idx_top_k = spec_int(g, A, "attention.indexer.top_k");
    s.idx_kpool = spec_int(g, A, "attention.indexer.kpool");
    // llama.cpp's default where the file has no such key (llama-hparams.h): GLM-5.3-Flash's GGUF writes none
    s.idx_kpool_select_tail = spec_int(g, A, "attention.indexer.kpool_select_tail", 1) != 0;

    if (s.arch == ModelArch::DeepSeek41 && !read_deepseek41(g, s, err)) return false;

    if (s.arch == ModelArch::MiniMaxM3) {
        // minimax-m3.cpp (llama.cpp master 8642541b2, load_arch_hparams and its graph's asserts): sigmoid routing with
        // a shared expert, the leading dense blocks, one indexer head a K/V head, blocks of block_size positions
        const char* need[] = {"expert_feed_forward_length", "expert_shared_count", "expert_gating_func",
                              "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k",
                              "attention.indexer.block_size", "attention.indexer.local_blocks"};
        for (const char* k : need)
            if (!spec_get(g, A, k)) { err = "missing " + A + "." + k; return false; }
        s.idx_block_size = spec_int(g, A, "attention.indexer.block_size");
        s.idx_local_blocks = spec_int(g, A, "attention.indexer.local_blocks");
        if (s.n_ff_shexp == 0) s.n_ff_shexp = s.n_ff_exp * s.n_expert_shared;   // n_ff_exp a shared expert
        const int64_t kvh = s.n_head_kv.empty() ? 0 : s.n_head_kv[0];
        for (int64_t i = 0; i < s.n_layer; ++i)
            if (s.n_head_kv[(size_t) i] != kvh) { err = A + ": attention.head_count_kv differs between layers"; return false; }
        if (s.n_nextn != 0) { err = A + ": MiniMax-M3 has no MTP layer in the model"; return false; }
        if (s.gating != ExpertGating::Sigmoid) {
            err = A + ".expert_gating_func " + std::to_string((uint32_t) s.gating) + ": MiniMax-M3 needs 2 (sigmoid)";
            return false;
        }
        if (s.idx_block_size <= 0 || s.idx_top_k <= 0 || s.idx_local_blocks < 0 || s.idx_key_length <= 0 || kvh <= 0 ||
            s.n_head % kvh != 0 || s.idx_n_head != kvh || s.key_length != s.value_length || s.n_expert_shared <= 0 ||
            s.rope_dims <= 0 || s.rope_dims > s.key_length || s.rope_dims > s.idx_key_length) {
            err = A + ": the sparse attention's shape (one indexer head a K/V head, block_size and top_k above 0, "
                      "rope.dimension_count within the key and the indexer key)";
            return false;
        }
    }

    if (s.arch == ModelArch::Glm5Next) {
        // glm5-next.cpp asserts these: an mHC of 4 streams and a pool size that divides the indexer's budget
        if (s.hc != 4) { err = A + ".hyper_connection.count " + std::to_string(s.hc) + ", only 4 is supported"; return false; }
        if (s.idx_kpool <= 1 || s.idx_top_k <= 0 || s.idx_top_k % s.idx_kpool != 0) {
            err = A + ": indexer.kpool " + std::to_string(s.idx_kpool) + " must be above 1 and divide indexer.top_k " +
                  std::to_string(s.idx_top_k);
            return false;
        }
        if (s.kda_head_dim <= 0 || s.kv_lora_rank <= 0 || s.q_lora_rank <= 0) {
            err = A + ": needs kda.head_dim, attention.q_lora_rank and attention.kv_lora_rank";
            return false;
        }
    }
    return true;
}

}  // namespace strata
