// SPDX-FileCopyrightText: 2026 the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/artifact/ds41_manifest_test.cpp - DeepSeek-V4.1-Flash's tensor manifest (ds41_manifest.hpp).
//
//   ds41_manifest_test            the manifest of the model's shape: the tensor count of a 40-layer model
//                                 (llama.cpp's 1043 with the vision bias and the Engram table), the per-layer rules
//   ds41_manifest_test FILE...    checks real models (any shard each): every tensor, its shape, its bytes in bounds
#include "strata/artifact/ds41_manifest.hpp"

#include <cstdio>
#include <exception>
#include <string>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-66s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// DeepSeek-V4.1-Flash-1017's shape, as read_model_spec gives it for ISTA-DASLab's GGUF (model_spec_test's fixture)
strata::ModelSpec v41() {
    strata::ModelSpec s;
    s.arch = strata::ModelArch::DeepSeek41;
    s.n_layer = 40, s.n_embd = 5120, s.n_head = 64, s.key_length = 512, s.q_lora_rank = 1280;
    s.o_group_count = 8, s.o_lora_rank = 1024, s.hc = 4, s.n_expert = 384, s.n_ff_exp = 2304, s.n_expert_shared = 1;
    s.idx_key_length = 128, s.idx_n_head = 32;
    s.compress_ratios.assign(40, 0);
    s.kv_src.assign(40, -1);
    s.topk_src.assign(40, -1);
    const int kv[] = {2, 8, 14, 20}, idx[] = {2, 8, 14, 20, 24, 28, 32, 36};
    int last_kv = -1, last_idx = -1;
    for (int l = 0; l < 40; ++l) {
        s.compress_ratios[(size_t) l] = l < 2 ? 0 : l < 20 ? 2 : 1;
        for (int k : kv) if (k == l) last_kv = l;
        for (int k : idx) if (k == l) last_idx = l;
        if (s.compress_ratios[(size_t) l]) s.kv_src[(size_t) l] = last_kv, s.topk_src[(size_t) l] = last_idx;
    }
    s.engram_layers = {1, 14};
    s.engram_n_heads = 24, s.engram_head_dim = 256;
    return s;
}

int selftest() {
    std::printf("ds41_manifest: the tensors of a 40-layer DeepSeek-V4.1-Flash\n");
    const auto m = ds41_manifest(v41(), 129280);
    size_t comp = 0, gate = 0, idxq = 0, engram = 0;
    for (const auto& t : m) {
        comp += t.name.find("attn_compressor_kv") != std::string::npos;
        gate += t.name.find("attn_compressor_gate") != std::string::npos;
        idxq += t.name.find("indexer.attn_q_b") != std::string::npos;
        engram += t.name.find("engram_kv") != std::string::npos;
    }
    check(m.size() == 1043, "1043 tensors (llama.cpp's count for the GSQ-RCO GGUF), got " + std::to_string(m.size()));
    check(comp == 4 && gate == 3, "a compressor on the 4 KV sources, a gate on the 3 of ratio 2");
    check(idxq == 8 && engram == 2, "an indexer query on the 8 index sources, Engram on 2 layers");
    check(m[0].name == "token_embd.weight" && m[0].shape[1] == 129280, "the embedding first, with the vocabulary");
    std::printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1) return selftest();
        int bad = 0;
        for (int i = 1; i < argc; ++i) {
            const strata::GgufModel m = strata::GgufModel::open(argv[i]);
            strata::ModelSpec s;
            std::string err, report;
            if (!strata::read_model_spec(m.meta(), s, err)) {
                std::printf("%s: REFUSED: %s\n", argv[i], err.c_str());
                ++bad;
                continue;
            }
            size_t n = 0;
            for (size_t k = 0; k < m.size(); ++k) n += m.shard(k).tensors().size();
            const bool ok = strata::ds41_check_tensors(m, s, err, &report);
            std::printf("%s: %zu shards, %zu tensors: %s\n%s", argv[i], m.size(), n, ok ? "OK" : err.c_str(),
                        report.c_str());
            bad += !ok;
        }
        return bad ? 1 : 0;
    } catch (const std::exception& e) {
        std::printf("ds41_manifest_test: %s\n", e.what());
        return 1;
    }
}
