// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/core/plain_segment.cpp - XeMaya: a PlainModel's last blocks behind the hybrid segment link (the worker).
#include "strata/core/plain_segment.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <list>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core {

namespace {

std::atomic<bool> g_stop{false};

#if defined(_WIN32)
// no WINAPI: x86-64 Windows has one calling convention, and the device pass of this file warns about __stdcall
BOOL on_console(DWORD) {
    g_stop = true;
    return TRUE;
}
#else
void on_signal(int) { g_stop = true; }
#endif

class PlainSegment : public strata::hybrid::SegmentExecutor {
public:
    PlainSegment(PlainModel& m, int64_t state_bytes) : m_(m), budget_(state_bytes) {
        strata::hybrid::random_bytes(reinterpret_cast<uint8_t*>(&epoch_), sizeof epoch_);
    }

    strata::hybrid::Shape shape() override {
        strata::hybrid::Shape s;
        const ModelSpec& sp = m_.spec();
        s.fingerprint = m_.fingerprint();
        s.epoch = epoch_;
        s.seg_first = m_.seg_first();
        s.seg_last = m_.seg_last();
        s.n_layer = sp.n_layer;
        s.n_embd = sp.n_embd;
        s.hand_w = m_.hand_width();
        s.n_vocab = m_.vocab();
        s.max_t = m_.max_window();
        s.max_cells = m_.max_cells();
        s.prompt_chunk = m_.prompt_chunk();
        s.hand_cap = m_.prompt_chunk();
        s.arch = std::to_string((int) sp.arch);
        const auto st = m_.stages();
        for (const auto& g : st) s.device += (s.device.empty() ? "" : " + ") + g.name;
        return s;
    }
    bool reset(std::string& err) override { return m_.reset(err); }
    bool sampler(const strata::hybrid::SamplerWire& w, std::string&) override {
        strata::kernels::SamplerParams sp;
        sp.top_k = w.top_k; sp.min_keep = w.min_keep; sp.penalty_last_n = w.penalty_last_n;
        sp.top_p = w.top_p; sp.min_p = w.min_p; sp.temperature = w.temperature;
        sp.penalty_repeat = w.penalty_repeat; sp.penalty_freq = w.penalty_freq; sp.penalty_present = w.penalty_present;
        sp.seed = w.seed; sp.greedy = w.greedy != 0; sp.gumbel = w.gumbel != 0;
        m_.set_sampler(sp);
        return true;
    }
    bool forward(const int32_t* tokens, int T, int64_t pos, const float* rows, bool defer, int32_t* out, float* logits,
                 std::string& err) override {
        const auto t0 = std::chrono::steady_clock::now();
        if (!m_.segment_forward(tokens, T, pos, rows, out, logits, err, defer)) return false;
        window_ms_ += since(t0);
        window_rows_ += T;
        return true;
    }
    bool prompt_begin(const int32_t* tokens, int64_t n, int64_t pos, int64_t T, int64_t& longest, int64_t& cap,
                      bool& too_short, std::string& err) override {
        prompt_.assign(tokens, tokens + n);
        prompt_pos_ = pos;
        if (m_.prompt_open(n, pos, T, longest, cap, err)) return true;
        too_short = err.empty();
        if (too_short) err = "the worker's tier lends too little for a prompt chunk";
        return false;
    }
    bool prompt_chunk(int64_t c0, int64_t c1, const float* rows, int32_t* out, float* logits, std::string& err) override {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t n = (int64_t) prompt_.size();
        if (!m_.prompt_step(prompt_.data(), n, prompt_pos_, c0, c1, rows, out, logits, err)) return false;
        prompt_ms_ += since(t0);
        prompt_rows_ += std::min(c1, n) - c0;
        prompt_done_ = c1 >= n;
        return true;
    }
    bool prompt_end(std::string& err) override {
        const bool settle = prompt_done_;
        prompt_done_ = false;
        prompt_.clear();
        return m_.prompt_close(settle, err);
    }
    bool save(uint64_t id, int64_t tokens, bool recurrent, uint64_t& bytes, std::string& err) override {
        drop(id);
        State st;
        st.id = id;
        st.recurrent = recurrent;
        if (!(recurrent ? m_.save_recurrent(tokens, st.snap, err) : m_.save_state(tokens, st.snap, err))) return false;
        bytes = st.snap.bytes();
        held_ += bytes;
        states_.push_front(std::move(st));
        // the oldest conversations go when the budget is passed (the newest always stays); the caller's ticket for
        // one of them then finds nothing, and it reads that conversation again from its tokens
        while (held_ > (uint64_t) budget_ && states_.size() > 1) {
            held_ -= states_.back().snap.bytes();
            states_.pop_back();
            ++evicted_;
        }
        return true;
    }
    bool restore(uint64_t id, bool recurrent, std::string& err) override {
        for (auto it = states_.begin(); it != states_.end(); ++it) {
            if (it->id != id || it->recurrent != recurrent) continue;
            states_.splice(states_.begin(), states_, it);
            return recurrent ? m_.restore_recurrent(states_.front().snap, err) : m_.restore_state(states_.front().snap, err);
        }
        err = "this worker no longer holds that conversation's state";
        return false;
    }
    void drop(uint64_t id) override {
        for (auto it = states_.begin(); it != states_.end(); ++it)
            if (it->id == id) {
                held_ -= it->snap.bytes();
                states_.erase(it);
                return;
            }
    }
    std::string stats_json() override {
        uint64_t used = 0, total = 0;
        for (const auto& g : m_.stages()) {
            used += g.total_b - std::min(g.free_b, g.total_b);
            total += g.total_b;
        }
        char b[640];
        std::snprintf(b, sizeof b,
                      "\"first_block\":%lld,\"last_block\":%lld,\"window_rows\":%lld,\"window_ms\":%.1f,"
                      "\"prompt_rows\":%lld,\"prompt_ms\":%.1f,\"vram_used_mib\":%llu,\"vram_total_mib\":%llu,"
                      "\"expert_slots\":%lld,\"experts_resident\":%lld,\"expert_hits\":%lld,\"expert_misses\":%lld,"
                      "\"states\":%zu,\"state_mib\":%llu,\"states_evicted\":%lld",
                      (long long) m_.seg_first(), (long long) m_.seg_last() - 1, (long long) window_rows_, window_ms_,
                      (long long) prompt_rows_, prompt_ms_, (unsigned long long) (used >> 20),
                      (unsigned long long) (total >> 20), (long long) m_.slots(), (long long) m_.resident(),
                      (long long) m_.hits(), (long long) m_.misses(), states_.size(),
                      (unsigned long long) (held_ >> 20), (long long) evicted_);
        return b;
    }

private:
    struct State {
        uint64_t id = 0;
        bool recurrent = false;
        PlainModel::Snapshot snap;
    };
    static double since(std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    }
    PlainModel& m_;
    int64_t budget_;
    uint64_t epoch_ = 0, held_ = 0;
    std::list<State> states_;
    std::vector<int32_t> prompt_;
    int64_t prompt_pos_ = 0, window_rows_ = 0, prompt_rows_ = 0, evicted_ = 0;
    bool prompt_done_ = false;
    double window_ms_ = 0, prompt_ms_ = 0;
};

}  // namespace

int plain_segment_serve(PlainModel& model, const std::string& host, int port, const std::string& token,
                        int64_t state_mib) {
    if (model.seg_first() == 0) {
        std::fprintf(stderr, "strata segment: --segment-serve needs --segment-first K (the worker's first block)\n");
        return 2;
    }
#if defined(_WIN32)
    SetConsoleCtrlHandler(on_console, TRUE);
#else
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
#endif
    PlainSegment seg(model, std::max<int64_t>(state_mib, 0) << 20);
    strata::hybrid::ServeOptions opt;
    opt.host = host;
    opt.port = port;
    opt.token = token;
    opt.stop = &g_stop;
    opt.on_event = [](const std::string& j) {
        std::printf("SEGMENT %s\n", j.c_str());
        std::fflush(stdout);
    };
    const strata::hybrid::Shape s = seg.shape();
    std::printf("SEGMENT {\"event\":\"loaded\",\"first_block\":%lld,\"last_block\":%lld,\"blocks\":%lld,\"hand_width\":%lld,"
                "\"context\":%lld,\"fingerprint\":\"%016llx\",%s}\n",
                (long long) s.seg_first, (long long) s.seg_last - 1, (long long) s.n_layer, (long long) s.hand_w,
                (long long) s.max_cells, (unsigned long long) s.fingerprint, seg.stats_json().c_str());
    std::fflush(stdout);
    std::string err;
    if (!strata::hybrid::serve(seg, opt, err)) {
        std::fprintf(stderr, "strata segment: %s\n", err.c_str());
        return 1;
    }
    return 0;
}

}  // namespace strata::core
