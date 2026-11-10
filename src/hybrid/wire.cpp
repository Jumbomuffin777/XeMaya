// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/hybrid/wire.cpp - XeMaya: the hybrid segment link's messages (see the header).
#include "strata/hybrid/wire.hpp"

#include <algorithm>
#include <random>

namespace strata::hybrid {

namespace net = strata::platform::net;

Writer& Writer::str(const std::string& s) {
    put<uint32_t>((uint32_t) s.size());
    return bytes(s.data(), s.size());
}

Writer& Writer::bytes(const void* p, size_t n) {
    if (n > 0) {
        const size_t at = b_.size();
        b_.resize(at + n);
        std::memcpy(b_.data() + at, p, n);
    }
    return *this;
}

std::string Reader::str() {
    const uint32_t n = get<uint32_t>();
    const uint8_t* p = bytes(n);
    return p != nullptr ? std::string(reinterpret_cast<const char*>(p), n) : std::string();
}

const uint8_t* Reader::bytes(size_t n) {
    if (!ok_ || n_ - at_ < n) { ok_ = false; return nullptr; }
    const uint8_t* p = p_ + at_;
    at_ += n;
    return p;
}

void put_shape(Writer& w, const Shape& s) {
    w.put(s.fingerprint).put(s.epoch).put(s.seg_first).put(s.seg_last).put(s.n_layer).put(s.n_embd).put(s.hand_w)
        .put(s.n_vocab).put(s.max_t).put(s.max_cells).put(s.prompt_chunk).put(s.hand_cap).str(s.arch).str(s.device);
}

bool get_shape(Reader& r, Shape& s) {
    s.fingerprint = r.get<uint64_t>();
    s.epoch = r.get<uint64_t>();
    for (int64_t* v : {&s.seg_first, &s.seg_last, &s.n_layer, &s.n_embd, &s.hand_w, &s.n_vocab, &s.max_t, &s.max_cells,
                       &s.prompt_chunk, &s.hand_cap})
        *v = r.get<int64_t>();
    s.arch = r.str();
    s.device = r.str();
    return r.ok();
}

bool send_msg(net::Socket s, Op op, uint16_t flags, uint32_t seq, const void* a, size_t na, const void* b, size_t nb) {
    Header h;
    h.op = (uint16_t) op;
    h.flags = flags;
    h.seq = seq;
    h.len = (uint64_t) na + (uint64_t) nb;
    return net::send_all(s, &h, sizeof h) && (na == 0 || net::send_all(s, a, na)) && (nb == 0 || net::send_all(s, b, nb));
}

bool recv_msg(net::Socket s, Header& h, std::vector<uint8_t>& payload, int timeout_ms, std::string& err) {
    if (!net::recv_all(s, &h, sizeof h, timeout_ms)) {
        err = "hybrid link: the peer closed the connection or did not answer";
        return false;
    }
    if (h.magic != kMagic) { err = "hybrid link: not a segment link message"; return false; }
    if (h.len > kMaxPayload) { err = "hybrid link: a message longer than the link takes"; return false; }
    payload.resize((size_t) h.len);
    // the payload follows its header at once: a stall here is a broken link whatever the caller's patience
    if (h.len > 0 && !net::recv_all(s, payload.data(), (size_t) h.len, 60000)) {
        err = "hybrid link: the connection broke inside a message";
        return false;
    }
    return true;
}

// ---- SHA-256 (FIPS 180-4) and HMAC (RFC 2104): the link's proof of the shared token -------------------------------

namespace {

constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2};

uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

struct Sha {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t block[64] = {};
    size_t fill = 0;
    uint64_t total = 0;

    void compress(const uint8_t* p) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void add(const void* data, size_t n) {
        const auto* p = static_cast<const uint8_t*>(data);
        total += n;
        while (n > 0) {
            const size_t take = std::min(n, sizeof block - fill);
            std::memcpy(block + fill, p, take);
            fill += take;
            p += take;
            n -= take;
            if (fill == sizeof block) { compress(block); fill = 0; }
        }
    }
    void finish(uint8_t out[32]) {
        const uint64_t bits = total * 8;
        const uint8_t one = 0x80, zero = 0;
        add(&one, 1);
        while (fill != 56) add(&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = (uint8_t) (bits >> (56 - 8 * i));
        add(len, 8);
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 4; ++j) out[4 * i + j] = (uint8_t) (h[i] >> (24 - 8 * j));
    }
};

}  // namespace

void sha256(const void* data, size_t n, uint8_t out[32]) {
    Sha s;
    s.add(data, n);
    s.finish(out);
}

void hmac_sha256(const void* key, size_t nk, const void* msg, size_t nm, uint8_t out[32]) {
    uint8_t k[64] = {};
    if (nk > sizeof k) sha256(key, nk, k);
    else if (nk > 0) std::memcpy(k, key, nk);
    uint8_t ipad[64], opad[64], inner[32];
    for (size_t i = 0; i < sizeof k; ++i) {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }
    Sha a;
    a.add(ipad, sizeof ipad);
    a.add(msg, nm);
    a.finish(inner);
    Sha b;
    b.add(opad, sizeof opad);
    b.add(inner, sizeof inner);
    b.finish(out);
}

bool same_bytes(const uint8_t* a, const uint8_t* b, size_t n) {
    uint8_t d = 0;
    for (size_t i = 0; i < n; ++i) d |= (uint8_t) (a[i] ^ b[i]);
    return d == 0;
}

void random_bytes(uint8_t* p, size_t n) {
    std::random_device rd;   // the system's source on Linux and on Windows
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t) rd();
}

// ---- IEEE halves: round to nearest even, as the K/V cache's FP16 rows are --------------------------------------------

void f32_to_f16(const float* x, uint16_t* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t b;
        std::memcpy(&b, &x[i], 4);
        const uint32_t sign = (b >> 16) & 0x8000u;
        const int32_t e = (int32_t) ((b >> 23) & 0xff) - 127 + 15;
        uint32_t m = b & 0x7fffffu;
        uint32_t h;
        if (((b >> 23) & 0xff) == 0xff) h = 0x7c00u | (m != 0 ? 0x200u : 0);   // inf, NaN
        else if (e >= 31) h = 0x7c00u;                                          // too large: inf
        else if (e <= 0) {                                                      // a subnormal half, or zero
            if (e < -10) h = 0;
            else {
                m |= 0x800000u;
                const int shift = 14 - e;
                h = m >> shift;
                const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
                if (rem > half || (rem == half && (h & 1u))) ++h;
            }
        } else {
            h = ((uint32_t) e << 10) | (m >> 13);
            const uint32_t rem = m & 0x1fffu;
            if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;   // a carry into the exponent is the right value
        }
        y[i] = (uint16_t) (sign | h);
    }
}

void f16_to_f32(const uint16_t* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t h = x[i], sign = (h & 0x8000u) << 16;
        uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu, b;
        if (e == 0x1f) b = sign | 0x7f800000u | (m << 13);
        else if (e != 0) b = sign | ((e + 112) << 23) | (m << 13);
        else if (m == 0) b = sign;
        else {
            e = 113;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            b = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
        std::memcpy(&y[i], &b, 4);
    }
}

}  // namespace strata::hybrid
