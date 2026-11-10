// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/hybrid/wire.hpp - XeMaya: the hybrid segment link's messages.
//
// Two engines run one model: the first holds blocks [0, K) on its GPUs, the second (the segment worker, on another
// machine and another maker's GPU) blocks [K, n) and the head.  Each keeps its own weights, experts and K/V cache; what
// crosses the link is the boundary's rows (the rows a layer split hands from one GPU to the next) once a window or a
// prompt chunk, and the picked tokens back.  No weight and no K/V row travels.
//
// A message is a 24-byte header and a payload, little-endian (both ends are x86-64).  The caller sends a request and
// reads one answer (Ok or Error) before the next: there is one conversation in flight, as in the engine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "strata/platform/net.hpp"

namespace strata::hybrid {

inline constexpr uint32_t kMagic = 0x31594D58;   // "XMY1"
inline constexpr uint16_t kProtocol = 1;
/// The largest payload either end accepts: a prompt chunk's rows (8,192 rows of four 8,192-wide streams) with room.
inline constexpr uint64_t kMaxPayload = 2ull << 30;

enum class Op : uint16_t {
    Hello = 1,     ///< protocol, a nonce -> the worker's shape and its nonce
    Auth,          ///< HMAC(token, worker nonce | caller nonce) -> HMAC(token, caller nonce | worker nonce)
    Reset,         ///< the next window starts a sequence at position 0
    Sampler,       ///< how the worker picks its tokens
    Forward,       ///< a window: T tokens at pos, their boundary rows -> T picked tokens (and their logits)
    PromptBegin,   ///< a prompt of n tokens at pos: the worker borrows its chunk buffers -> its chunk and capacity
    PromptChunk,   ///< the boundary rows of prompt tokens [c0, c1) -> after the last, the picked token (and its logits)
    PromptEnd,     ///< the buffers go back, the tiers settle
    Save,          ///< the worker sets aside its state of the first `tokens` tokens under an id (it stays there)
    Restore,       ///< ... and puts it back
    Drop,          ///< ... or forgets it
    Stats,         ///< -> a JSON object: what the worker counted since its start
    Bye,
    Ok = 100,
    Error = 101,   ///< the payload is the reason
};

enum : uint16_t {
    kFlagF16 = 1,      ///< the rows are IEEE halves (half the bytes; rounds each row once at the boundary)
    kFlagLogits = 2,   ///< the caller wants the logits back
    kFlagDefer = 4,    ///< Forward: a verify window, its rows not yet known to be kept
    kFlagRecurrent = 8,   ///< Save / Restore: the recurrent layers' state alone
    kFlagShort = 16,   ///< PromptBegin's answer: the worker's tier lends too little; the prompt goes through windows
};

struct Header {
    uint32_t magic = kMagic;
    uint16_t op = 0;
    uint16_t flags = 0;
    uint32_t seq = 0;        ///< the request's number, echoed by its answer
    uint32_t reserved = 0;
    uint64_t len = 0;        ///< the payload's bytes
};
static_assert(sizeof(Header) == 24, "the wire header is 24 bytes");

/// What a worker says of itself at Hello; the caller checks it against its own half of the model.
struct Shape {
    uint64_t fingerprint = 0;   ///< of the model's tensor directory and shape (model_fingerprint)
    uint64_t epoch = 0;         ///< random, a worker process's: a saved state's ticket names it
    int64_t seg_first = 0, seg_last = 0, n_layer = 0;
    int64_t n_embd = 0, hand_w = 0, n_vocab = 0;
    int64_t max_t = 0, max_cells = 0;
    int64_t prompt_chunk = 0, hand_cap = 0;   ///< 0: the worker reads prompts through its windows
    std::string arch, device;
};

/// The sampler's settings that travel (kernels/sampler.hpp's, without the device arrays a remote worker cannot read).
struct SamplerWire {
    int32_t top_k = 20, min_keep = 1, penalty_last_n = 0;
    float top_p = 0.95f, min_p = 0.0f, temperature = 1.0f;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    uint64_t seed = 0;
    uint8_t greedy = 0, gumbel = 0, pad[6] = {};
};
static_assert(sizeof(SamplerWire) == 56, "the sampler's wire form is 56 bytes");

/// A payload being written: plain values, strings (u32 length), raw bytes.
class Writer {
public:
    template <typename T> Writer& put(const T& v) {
        const size_t at = b_.size();
        b_.resize(at + sizeof(T));
        std::memcpy(b_.data() + at, &v, sizeof(T));
        return *this;
    }
    Writer& str(const std::string& s);
    Writer& bytes(const void* p, size_t n);
    const std::vector<uint8_t>& data() const { return b_; }

private:
    std::vector<uint8_t> b_;
};

/// ... and read: a short payload makes every later get fail and ok() false; nothing is read past the end.
class Reader {
public:
    Reader(const uint8_t* p, size_t n) : p_(p), n_(n) {}
    template <typename T> T get() {
        T v{};
        if (!ok_ || n_ - at_ < sizeof(T)) { ok_ = false; return v; }
        std::memcpy(&v, p_ + at_, sizeof(T));
        at_ += sizeof(T);
        return v;
    }
    std::string str();
    /// n bytes in place (null when they are not there)
    const uint8_t* bytes(size_t n);
    size_t left() const { return ok_ ? n_ - at_ : 0; }
    bool ok() const { return ok_; }

private:
    const uint8_t* p_;
    size_t n_, at_ = 0;
    bool ok_ = true;
};

void put_shape(Writer& w, const Shape& s);
bool get_shape(Reader& r, Shape& s);

/// One message: the header, then payload `a` and (the rows, sent from where they are) `b`.
bool send_msg(platform::net::Socket s, Op op, uint16_t flags, uint32_t seq, const void* a, size_t na,
              const void* b = nullptr, size_t nb = 0);
/// The next message.  False with the reason: the peer went away, sent something else, or was silent for timeout_ms.
bool recv_msg(platform::net::Socket s, Header& h, std::vector<uint8_t>& payload, int timeout_ms, std::string& err);

void sha256(const void* data, size_t n, uint8_t out[32]);
void hmac_sha256(const void* key, size_t nk, const void* msg, size_t nm, uint8_t out[32]);
/// Equal in time that does not depend on where they differ
bool same_bytes(const uint8_t* a, const uint8_t* b, size_t n);
void random_bytes(uint8_t* p, size_t n);

void f32_to_f16(const float* x, uint16_t* y, size_t n);
void f16_to_f32(const uint16_t* x, float* y, size_t n);

/// FNV-1a over a list of 64-bit values and strings: the model fingerprint's hash (the two ends feed it the same
/// things in the same order: PlainModel::fingerprint).
struct Fnv {
    uint64_t h = 1469598103934665603ull;
    void add(const void* p, size_t n) {
        const auto* c = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 1099511628211ull;
    }
    void add(uint64_t v) { add(&v, sizeof v); }
    void add(const std::string& s) { add((uint64_t) s.size()); add(s.data(), s.size()); }
};

}  // namespace strata::hybrid
