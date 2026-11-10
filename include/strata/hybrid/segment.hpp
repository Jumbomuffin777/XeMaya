// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/hybrid/segment.hpp - XeMaya: the two ends of the hybrid segment link (wire.hpp).
//
// The worker end (`serve`) runs whatever implements SegmentExecutor: the engine's PlainModel loaded with the model's
// last blocks (src/core/plain_segment.cpp), or a test's stand-in, so the link is tested without a GPU.  The caller end
// (RemoteSegment) is what the engine holding the first blocks talks through.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "strata/hybrid/wire.hpp"

namespace strata::hybrid {

/// The model's last blocks, as the link drives them.  Every call returns false with the reason; the link stays up.
class SegmentExecutor {
public:
    virtual ~SegmentExecutor() = default;
    virtual Shape shape() = 0;
    virtual bool reset(std::string& err) = 0;
    virtual bool sampler(const SamplerWire& sp, std::string& err) = 0;
    /// A window: T tokens at pos, their T rows of shape().hand_w floats -> T picks, and with `logits` T rows of them
    virtual bool forward(const int32_t* tokens, int T, int64_t pos, const float* rows, bool defer, int32_t* out,
                         float* logits, std::string& err) = 0;
    /// A prompt of n tokens at pos in chunks of at most T: `longest` the chunk the worker's buffers take, `hand_cap`
    /// the most rows one PromptChunk may bring.  `too_short` (with true): its tier lends too little, nothing was taken
    virtual bool prompt_begin(const int32_t* tokens, int64_t n, int64_t pos, int64_t T, int64_t& longest,
                              int64_t& hand_cap, bool& too_short, std::string& err) = 0;
    /// The rows of prompt tokens [c0, c1); after the prompt's last token `out` is the pick (and `logits` its row)
    virtual bool prompt_chunk(int64_t c0, int64_t c1, const float* rows, int32_t* out, float* logits,
                              std::string& err) = 0;
    virtual bool prompt_end(std::string& err) = 0;
    virtual bool save(uint64_t id, int64_t tokens, bool recurrent, uint64_t& bytes, std::string& err) = 0;
    virtual bool restore(uint64_t id, bool recurrent, std::string& err) = 0;
    virtual void drop(uint64_t id) = 0;
    /// A JSON object's members (no braces), or empty: what the executor counted, for the manager's display
    virtual std::string stats_json() = 0;
};

struct ServeOptions {
    std::string host = "127.0.0.1";
    int port = 0;                  ///< 0: any free port (on_event's "listening" says which)
    std::string token;             ///< the shared secret; empty only on a loopback address
    std::atomic<bool>* stop = nullptr;   ///< set from elsewhere: the loop ends after the message in hand
    int max_sessions = 0;          ///< 0: until stop; tests: return after this many connections
    /// One JSON object a line: {"event":"listening","port":..}, "connected", "disconnected", "stats" (at most one a
    /// second while work arrives).  The worker program prints them for the program that manages it.
    std::function<void(const std::string&)> on_event;
};

/// Serves one caller at a time until `stop`.  A caller that goes away mid-prompt has its buffers given back and its
/// sequence reset; saved states stay for when it returns.  False with the reason when the port cannot be taken.
bool serve(SegmentExecutor& ex, const ServeOptions& opt, std::string& err);

struct LinkStats {
    uint64_t calls = 0, bytes_out = 0, bytes_in = 0, connects = 0;
    double round_ms = 0;    ///< the calls' time on this side, request to answer
    double remote_ms = 0;   ///< ... of it, the worker's own time in its blocks (the rest is the link)
    double last_round_ms = 0, last_remote_ms = 0;
};

class RemoteSegment {
public:
    RemoteSegment() = default;
    ~RemoteSegment();
    RemoteSegment(const RemoteSegment&) = delete;
    RemoteSegment& operator=(const RemoteSegment&) = delete;

    /// `addr` host:port.  Both ends prove the token; `f16` sends the rows as halves.  timeout_s bounds each answer.
    bool connect(const std::string& addr, const std::string& token, bool f16, std::string& err, int timeout_s = 900);
    /// The same worker again after the link broke (its sequence is gone: the caller resets its own too)
    bool reconnect(std::string& err);
    void close();
    bool connected() const { return sock_ != platform::net::kNoSocket; }
    const Shape& shape() const { return shape_; }
    const LinkStats& link() const { return stats_; }
    const std::string& address() const { return addr_; }
    bool f16() const { return f16_; }

    bool reset(std::string& err);
    bool sampler(const SamplerWire& sp, std::string& err);
    bool forward(const int32_t* tokens, int T, int64_t pos, const float* rows, bool defer, int32_t* out, float* logits,
                 std::string& err);
    bool prompt_begin(const int32_t* tokens, int64_t n, int64_t pos, int64_t T, int64_t& longest, int64_t& hand_cap,
                      bool& too_short, std::string& err);
    bool prompt_chunk(int64_t c0, int64_t c1, int64_t n, const float* rows, int32_t* out, float* logits,
                      std::string& err);
    bool prompt_end(std::string& err);
    bool save(uint64_t id, int64_t tokens, bool recurrent, std::string& err);
    bool restore(uint64_t id, bool recurrent, std::string& err);
    void drop(uint64_t id);
    bool stats(std::string& json, std::string& err);

private:
    bool call(Op op, uint16_t flags, const Writer& head, const void* rows, size_t n_rows_bytes, Header& h,
              std::string& err);
    const void* pack_rows(const float* rows, size_t n, size_t& bytes);

    platform::net::Socket sock_ = platform::net::kNoSocket;
    std::string addr_, token_;
    bool f16_ = false;
    int timeout_ms_ = 900000;
    uint32_t seq_ = 0;
    Shape shape_;
    LinkStats stats_;
    std::vector<uint8_t> in_;
    std::vector<uint16_t> half_;
};

}  // namespace strata::hybrid
