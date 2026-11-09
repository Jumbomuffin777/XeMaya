// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// include/strata/artifact/ds41_manifest.hpp - the tensors a DeepSeek-V4.1-Flash (deepseek41) GGUF must hold, with
// their shapes, as llama.cpp's llama_model_deepseek41::load_arch_tensors creates them (master 8642541b2; MIT License,
// Copyright (c) 2023-2026 The ggml authors), and the check of a model's shards against that list.
//
// Header-only, with no ggml or SYCL dependency: the check runs before anything is uploaded, and on its own
// (ds41_manifest_test FILE).
#pragma once

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/model_spec.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace strata {

struct Ds41Tensor {
    std::string name;
    std::vector<uint64_t> shape;   ///< GGUF order (dim 0 fastest); empty: any shape (checked elsewhere)
    bool required = true;
    bool reshape = false;          ///< llama.cpp's TENSOR_ALLOW_RESHAPE: the file may store it flatter, same dim 0
};

/// Every tensor of the model `s` describes.  n_vocab: token_embd's rows (from the file).
inline std::vector<Ds41Tensor> ds41_manifest(const ModelSpec& s, uint64_t n_vocab) {
    std::vector<Ds41Tensor> out;
    const uint64_t N = (uint64_t) s.n_embd, D = (uint64_t) s.key_length, H = (uint64_t) s.n_head;
    const uint64_t QL = (uint64_t) s.q_lora_rank, G = (uint64_t) s.o_group_count, OL = (uint64_t) s.o_lora_rank;
    const uint64_t HC = (uint64_t) s.hc, MIXD = (2 + HC) * HC, E = (uint64_t) s.n_expert, F = (uint64_t) s.n_ff_exp;
    const uint64_t FS = (uint64_t) (s.n_ff_exp * s.n_expert_shared);
    const uint64_t IDE = (uint64_t) s.idx_key_length, IDH = (uint64_t) s.idx_n_head;
    out.push_back({"token_embd.weight", {N, n_vocab}});
    out.push_back({"output_norm.weight", {N}});
    out.push_back({"output.weight", {N, n_vocab}});
    if (!s.engram_layers.empty())   // all Engram tables in one lazily read tensor; its rows are checked below
        out.push_back({"per_layer_token_embd.weight", {}});
    for (int64_t i = 0; i < s.n_layer; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        const auto add = [&](const std::string& n, std::vector<uint64_t> shape, bool req = true, bool rs = false) {
            out.push_back({b + n, std::move(shape), req, rs});
        };
        add("attn_norm.weight", {N});
        add("attn_sinks.weight", {H});
        add("attn_q_a.weight", {N, QL});
        add("attn_q_a_norm.weight", {QL});
        add("attn_q_b.weight", {QL, H * D});
        add("attn_kv.weight", {N, D});
        add("attn_kv_a_norm.weight", {D});
        add("attn_output_a.weight", {H * D / G, OL, G}, true, true);
        add("attn_output_b.weight", {G * OL, N});
        for (const char* p : {"hc_attn_", "hc_ffn_"}) {
            add(std::string(p) + "fn.weight", {HC * N, MIXD});
            add(std::string(p) + "base.weight", {MIXD});
            add(std::string(p) + "scale.weight", {3});
        }
        if (s.kv_src[(size_t) i] == i) {
            add("attn_compressor_kv.weight", {N, D});
            add("attn_compressor_norm.weight", {D});
            if (s.compress_ratios[(size_t) i] > 1) add("attn_compressor_gate.weight", {N, D});
            add("indexer.attn_k.weight", {D, IDE});
            add("indexer.k_norm.weight", {IDE});
        }
        if (s.topk_src[(size_t) i] == i) {
            add("indexer.attn_q_b.weight", {QL, IDH * IDE});
            add("indexer.proj.weight", {N, IDH});
        }
        if (s.engram_index(i) >= 0) {
            add("engram_kv.weight", {(uint64_t) (s.engram_n_heads * s.engram_head_dim), N * (HC + 1)});
            add("engram_gate.weight", {N, HC});
        }
        add("ffn_gate_inp.weight", {N, E});
        add("exp_probs_b.bias", {E});
        add("exp_probs_b_vl.bias", {E}, false);   // the vision router bias (pictures)
        add("ffn_norm.weight", {N});
        add("ffn_gate_exps.weight", {N, F, E});
        add("ffn_down_exps.weight", {F, N, E});
        add("ffn_up_exps.weight", {N, F, E});
        add("ffn_gate_shexp.weight", {N, FS});
        add("ffn_down_shexp.weight", {FS, N});
        add("ffn_up_shexp.weight", {N, FS});
    }
    return out;
}

/// Checks the shards of `m` against the manifest of `s`: every required tensor present with its shape (a reshape
/// tensor: the same dim 0 and element count), inside its file, and no tensor the manifest does not know (a newer
/// conversion's tensor the engine would ignore).  `report` gets one line per problem.  The Engram table must hold the
/// rows its heads address.
inline bool ds41_check_tensors(const GgufModel& m, const ModelSpec& s, std::string& err, std::string* report = nullptr) {
    const TensorInfo* emb = m.find("token_embd.weight");
    if (!emb || emb->shape.size() != 2) { err = "token_embd.weight is missing or not 2-D"; return false; }
    const auto manifest = ds41_manifest(s, emb->shape[1]);
    std::set<std::string> known;
    long long bad = 0;
    const auto note = [&](const std::string& line) {
        ++bad;
        if (report) *report += line + "\n";
    };
    const auto elements = [](const std::vector<uint64_t>& v) {
        uint64_t n = 1;
        for (uint64_t d : v) n *= d;
        return n;
    };
    for (const Ds41Tensor& want : manifest) {
        known.insert(want.name);
        size_t shard = 0;
        const TensorInfo* t = m.find(want.name, &shard);
        if (!t) {
            if (want.required) note("missing " + want.name);
            continue;
        }
        if (!m.in_bounds(*t, shard)) note(want.name + ": its bytes are not inside " + m.shard(shard).path());
        if (want.shape.empty()) continue;
        const bool same = t->shape == want.shape;
        const bool flat = want.reshape && !t->shape.empty() && t->shape[0] == want.shape[0] &&
                          elements(t->shape) == elements(want.shape);
        if (!same && !flat) {
            std::string got, exp;
            for (uint64_t d : t->shape) got += (got.empty() ? "" : ",") + std::to_string(d);
            for (uint64_t d : want.shape) exp += (exp.empty() ? "" : ",") + std::to_string(d);
            note(want.name + ": shape [" + got + "], expected [" + exp + "]");
        }
    }
    if (const TensorInfo* tab = m.find("per_layer_token_embd.weight")) {
        uint64_t rows = 0;
        for (size_t c = 0; c < s.engram_offsets.size(); ++c)
            rows = std::max<uint64_t>(rows, s.engram_offsets[c] + s.engram_vocab[c]);
        if (tab->shape.size() != 2 || tab->shape[0] != (uint64_t) s.engram_head_dim || tab->shape[1] < rows)
            note("per_layer_token_embd.weight: too few rows for the Engram heads (" + std::to_string(rows) + ")");
    }
    for (size_t i = 0; i < m.size(); ++i)
        for (const TensorInfo& t : m.shard(i).tensors())
            if (!known.count(t.name)) note("unexpected tensor " + t.name);
    if (bad) {
        err = std::to_string(bad) + " problem(s) in the DeepSeek-V4.1 tensors";
        return false;
    }
    return true;
}

}  // namespace strata
