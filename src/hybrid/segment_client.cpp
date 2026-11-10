// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/hybrid/segment_client.cpp - XeMaya: the caller end of the hybrid segment link (segment.hpp).
#include <algorithm>
#include <chrono>

#include "strata/hybrid/segment.hpp"

namespace strata::hybrid {

namespace net = strata::platform::net;

RemoteSegment::~RemoteSegment() { close(); }

void RemoteSegment::close() {
    if (sock_ == net::kNoSocket) return;
    (void) send_msg(sock_, Op::Bye, 0, ++seq_, nullptr, 0);
    net::close(sock_);
    sock_ = net::kNoSocket;
}

bool RemoteSegment::connect(const std::string& addr, const std::string& token, bool f16, std::string& err,
                            int timeout_s) {
    addr_ = addr;
    token_ = token;
    f16_ = f16;
    timeout_ms_ = std::max(1, timeout_s) * 1000;
    return reconnect(err);
}

bool RemoteSegment::reconnect(std::string& err) {
    if (sock_ != net::kNoSocket) { net::close(sock_); sock_ = net::kNoSocket; }
    std::string host;
    int port = 0;
    if (!net::split_host_port(addr_, host, port)) { err = "hybrid segment: \"" + addr_ + "\" is not host:port"; return false; }
    const net::Socket s = net::connect_to(host, port, 5000, err);
    if (s == net::kNoSocket) { err = "hybrid segment: " + err; return false; }
    auto fail = [&](const std::string& why) {
        net::close(s);
        err = why;
        return false;
    };
    uint8_t nonce_c[16];
    random_bytes(nonce_c, 16);
    Writer hello;
    hello.put<uint16_t>(kProtocol).bytes(nonce_c, 16);
    Header h;
    std::string e;
    if (!send_msg(s, Op::Hello, 0, ++seq_, hello.data().data(), hello.data().size()) || !recv_msg(s, h, in_, 10000, e))
        return fail("hybrid segment: no hello from " + addr_ + (e.empty() ? "" : " (" + e + ")"));
    if (h.op != (uint16_t) Op::Ok)
        return fail("hybrid segment: " + std::string(in_.begin(), in_.end()));
    Reader r(in_.data(), in_.size());
    Shape shape;
    const bool got = get_shape(r, shape);
    const uint8_t* ns = got ? r.bytes(16) : nullptr;
    if (ns == nullptr) return fail("hybrid segment: the worker's hello is not this protocol's");
    uint8_t nonce_s[16], msg[33], mac[32], want[32];
    std::memcpy(nonce_s, ns, 16);
    msg[0] = 'c';
    std::memcpy(msg + 1, nonce_s, 16);
    std::memcpy(msg + 17, nonce_c, 16);
    hmac_sha256(token_.data(), token_.size(), msg, sizeof msg, mac);
    if (!send_msg(s, Op::Auth, 0, ++seq_, mac, sizeof mac) || !recv_msg(s, h, in_, 10000, e))
        return fail("hybrid segment: the worker closed the link at the token (" + e + ")");
    msg[0] = 's';
    std::memcpy(msg + 1, nonce_c, 16);
    std::memcpy(msg + 17, nonce_s, 16);
    hmac_sha256(token_.data(), token_.size(), msg, sizeof msg, want);
    if (h.op != (uint16_t) Op::Ok || in_.size() != 32 || !same_bytes(in_.data(), want, 32))
        return fail(h.op == (uint16_t) Op::Error ? "hybrid segment: " + std::string(in_.begin(), in_.end())
                                                 : "hybrid segment: the worker did not prove the token");
    sock_ = s;
    shape_ = shape;
    ++stats_.connects;
    return true;
}

bool RemoteSegment::call(Op op, uint16_t flags, const Writer& head, const void* rows, size_t n_rows_bytes, Header& h,
                         std::string& err) {
    if (sock_ == net::kNoSocket) {
        err = "hybrid segment: the link to " + addr_ + " is down";
        return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t seq = ++seq_;
    std::string e;
    if (!send_msg(sock_, op, flags, seq, head.data().data(), head.data().size(), rows, n_rows_bytes) ||
        !recv_msg(sock_, h, in_, timeout_ms_, e) || h.seq != seq) {
        // a broken or confused link is not resumed mid-sequence: the worker's state can no longer be trusted
        net::close(sock_);
        sock_ = net::kNoSocket;
        err = "hybrid segment: the link to " + addr_ + " broke" + (e.empty() ? "" : " (" + e + ")");
        return false;
    }
    ++stats_.calls;
    stats_.bytes_out += sizeof(Header) + head.data().size() + n_rows_bytes;
    stats_.bytes_in += sizeof(Header) + in_.size();
    stats_.last_round_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    stats_.round_ms += stats_.last_round_ms;
    if (h.op == (uint16_t) Op::Error) {
        err = "hybrid segment (" + addr_ + "): " + std::string(in_.begin(), in_.end());
        return false;
    }
    if (h.op != (uint16_t) Op::Ok) { err = "hybrid segment: an answer that is none"; return false; }
    return true;
}

const void* RemoteSegment::pack_rows(const float* rows, size_t n, size_t& bytes) {
    if (!f16_) { bytes = n * 4; return rows; }
    half_.resize(n);
    f32_to_f16(rows, half_.data(), n);
    bytes = n * 2;
    return half_.data();
}

bool RemoteSegment::reset(std::string& err) {
    Header h;
    return call(Op::Reset, 0, Writer(), nullptr, 0, h, err);
}

bool RemoteSegment::sampler(const SamplerWire& sp, std::string& err) {
    Header h;
    Writer w;
    w.bytes(&sp, sizeof sp);
    return call(Op::Sampler, 0, w, nullptr, 0, h, err);
}

bool RemoteSegment::forward(const int32_t* tokens, int T, int64_t pos, const float* rows, bool defer, int32_t* out,
                            float* logits, std::string& err) {
    Writer w;
    w.put<int32_t>(T).put<int32_t>(0).put<int64_t>(pos).bytes(tokens, (size_t) T * 4);
    if ((T & 1) != 0) w.put<int32_t>(0);   // the rows start at a multiple of 8 bytes
    size_t nb = 0;
    const void* x = pack_rows(rows, (size_t) T * (size_t) shape_.hand_w, nb);
    Header h;
    const uint16_t flags = (uint16_t) ((f16_ ? kFlagF16 : 0) | (logits != nullptr ? kFlagLogits : 0) | (defer ? kFlagDefer : 0));
    if (!call(Op::Forward, flags, w, x, nb, h, err)) return false;
    Reader r(in_.data(), in_.size());
    const float ms = r.get<float>();
    const int32_t n = r.get<int32_t>();
    const uint8_t* picks = n == T ? r.bytes((size_t) T * 4) : nullptr;
    const uint8_t* lg = picks != nullptr && logits != nullptr ? r.bytes((size_t) T * (size_t) shape_.n_vocab * 4) : nullptr;
    if (picks == nullptr || (logits != nullptr && lg == nullptr)) { err = "hybrid segment: a window's answer is short"; return false; }
    std::memcpy(out, picks, (size_t) T * 4);
    if (logits != nullptr) std::memcpy(logits, lg, (size_t) T * (size_t) shape_.n_vocab * 4);
    stats_.last_remote_ms = ms;
    stats_.remote_ms += ms;
    return true;
}

bool RemoteSegment::prompt_begin(const int32_t* tokens, int64_t n, int64_t pos, int64_t T, int64_t& longest,
                                 int64_t& hand_cap, bool& too_short, std::string& err) {
    Writer w;
    w.put(n).put(pos).put(T).bytes(tokens, (size_t) n * 4);
    Header h;
    too_short = false;
    if (!call(Op::PromptBegin, 0, w, nullptr, 0, h, err)) return false;
    Reader r(in_.data(), in_.size());
    longest = r.get<int64_t>();
    hand_cap = r.get<int64_t>();
    if (!r.ok()) { err = "hybrid segment: a prompt's answer is short"; return false; }
    if ((h.flags & kFlagShort) != 0) {
        too_short = true;
        err.clear();
        return false;
    }
    return true;
}

bool RemoteSegment::prompt_chunk(int64_t c0, int64_t c1, int64_t n, const float* rows, int32_t* out, float* logits,
                                 std::string& err) {
    Writer w;
    w.put(c0).put(c1);
    size_t nb = 0;
    const void* x = pack_rows(rows, (size_t) (std::min(c1, n) - c0) * (size_t) shape_.hand_w, nb);
    Header h;
    const bool final = c1 >= n;
    const uint16_t flags = (uint16_t) ((f16_ ? kFlagF16 : 0) | (final && logits != nullptr ? kFlagLogits : 0));
    if (!call(Op::PromptChunk, flags, w, x, nb, h, err)) return false;
    Reader r(in_.data(), in_.size());
    const float ms = r.get<float>();
    const int32_t was_final = r.get<int32_t>(), pick = r.get<int32_t>();
    (void) r.get<int32_t>();
    if (!r.ok() || (was_final != 0) != final) { err = "hybrid segment: a chunk's answer is short"; return false; }
    stats_.last_remote_ms = ms;
    stats_.remote_ms += ms;
    if (!final) return true;
    *out = pick;
    if (logits != nullptr) {
        const uint8_t* lg = r.bytes((size_t) shape_.n_vocab * 4);
        if (lg == nullptr) { err = "hybrid segment: the prompt's logits are short"; return false; }
        std::memcpy(logits, lg, (size_t) shape_.n_vocab * 4);
    }
    return true;
}

bool RemoteSegment::prompt_end(std::string& err) {
    Header h;
    return call(Op::PromptEnd, 0, Writer(), nullptr, 0, h, err);
}

bool RemoteSegment::save(uint64_t id, int64_t tokens, bool recurrent, std::string& err) {
    Header h;
    return call(Op::Save, recurrent ? kFlagRecurrent : 0, Writer().put(id).put(tokens), nullptr, 0, h, err);
}

bool RemoteSegment::restore(uint64_t id, bool recurrent, std::string& err) {
    Header h;
    return call(Op::Restore, recurrent ? kFlagRecurrent : 0, Writer().put(id), nullptr, 0, h, err);
}

void RemoteSegment::drop(uint64_t id) {
    Header h;
    std::string e;
    (void) call(Op::Drop, 0, Writer().put(id), nullptr, 0, h, e);
}

bool RemoteSegment::stats(std::string& json, std::string& err) {
    Header h;
    if (!call(Op::Stats, 0, Writer(), nullptr, 0, h, err)) return false;
    json.assign(in_.begin(), in_.end());
    return true;
}

}  // namespace strata::hybrid
