// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/hybrid/segment_server.cpp - XeMaya: the worker end of the hybrid segment link (segment.hpp).
#include <algorithm>
#include <chrono>
#include <cstdio>

#include "strata/hybrid/segment.hpp"

namespace strata::hybrid {

namespace net = strata::platform::net;

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (const char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char) c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    return o;
}

bool loopback(const std::string& host) { return host == "127.0.0.1" || host == "::1" || host == "localhost"; }

struct Session {
    SegmentExecutor& ex;
    const ServeOptions& opt;
    net::Socket s;
    Shape shape;
    bool in_prompt = false;
    int64_t prompt_n = 0;
    uint64_t calls = 0, bytes_in = 0, bytes_out = 0;
    double busy_ms = 0;
    std::vector<uint8_t> in;
    std::vector<float> rows, logits;
    std::vector<int32_t> out;
    std::chrono::steady_clock::time_point said = std::chrono::steady_clock::now();

    Session(SegmentExecutor& e, const ServeOptions& o, net::Socket sock) : ex(e), opt(o), s(sock), shape(e.shape()) {}

    bool answer(const Header& q, uint16_t flags, const Writer& w, const void* b = nullptr, size_t nb = 0) {
        bytes_out += sizeof(Header) + w.data().size() + nb;
        return send_msg(s, Op::Ok, flags, q.seq, w.data().data(), w.data().size(), b, nb);
    }
    bool refuse(const Header& q, const std::string& why) {
        bytes_out += sizeof(Header) + why.size();
        return send_msg(s, Op::Error, 0, q.seq, why.data(), why.size());
    }
    // the rows of a request, from where the payload has them (FP32, 8-byte aligned by the caller) or widened
    const float* take_rows(Reader& r, uint16_t flags, size_t n) {
        if ((flags & kFlagF16) != 0) {
            const uint8_t* p = r.bytes(n * 2);
            if (p == nullptr) return nullptr;
            rows.resize(n);
            f16_to_f32(reinterpret_cast<const uint16_t*>(p), rows.data(), n);
            return rows.data();
        }
        const uint8_t* p = r.bytes(n * 4);
        if (p == nullptr) return nullptr;
        if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0) return reinterpret_cast<const float*>(p);
        rows.resize(n);
        std::memcpy(rows.data(), p, n * 4);
        return rows.data();
    }
    void stats_event() {
        if (!opt.on_event || ms_since(said) < 1000.0) return;
        said = std::chrono::steady_clock::now();
        std::string j = "{\"event\":\"stats\",\"calls\":" + std::to_string(calls) + ",\"bytes_in\":" +
                        std::to_string(bytes_in) + ",\"bytes_out\":" + std::to_string(bytes_out) + ",\"busy_ms\":" +
                        std::to_string((long long) busy_ms);
        const std::string more = ex.stats_json();
        if (!more.empty()) j += "," + more;
        opt.on_event(j + "}");
    }

    bool handshake(std::string& err) {
        Header h;
        if (!recv_msg(s, h, in, 10000, err)) return false;
        Reader r(in.data(), in.size());
        const uint16_t proto = r.get<uint16_t>();
        const uint8_t* nc = r.bytes(16);
        if (h.op != (uint16_t) Op::Hello || nc == nullptr) { err = "hybrid link: the caller did not say hello"; return false; }
        if (proto != kProtocol) {
            err = "hybrid link: the caller speaks protocol " + std::to_string(proto) + ", this worker " +
                  std::to_string(kProtocol);
            refuse(h, err);
            return false;
        }
        uint8_t nonce_c[16], nonce_s[16];
        std::memcpy(nonce_c, nc, 16);
        random_bytes(nonce_s, 16);
        Writer w;
        put_shape(w, shape);
        w.bytes(nonce_s, 16);
        if (!answer(h, 0, w) || !recv_msg(s, h, in, 10000, err)) return false;
        uint8_t msg[33], want[32];
        msg[0] = 'c';
        std::memcpy(msg + 1, nonce_s, 16);
        std::memcpy(msg + 17, nonce_c, 16);
        hmac_sha256(opt.token.data(), opt.token.size(), msg, sizeof msg, want);
        if (h.op != (uint16_t) Op::Auth || in.size() != 32 || !same_bytes(in.data(), want, 32)) {
            err = "hybrid link: the caller's token does not match this worker's";
            refuse(h, err);
            return false;
        }
        msg[0] = 's';
        std::memcpy(msg + 1, nonce_c, 16);
        std::memcpy(msg + 17, nonce_s, 16);
        hmac_sha256(opt.token.data(), opt.token.size(), msg, sizeof msg, want);
        Writer proof;
        proof.bytes(want, 32);
        return answer(h, 0, proof);
    }

    // one request; false: the connection is done (Bye, or it broke)
    bool step(std::string& err) {
        Header h;
        if (!recv_msg(s, h, in, 0, err)) return false;
        bytes_in += sizeof(Header) + in.size();
        ++calls;
        const auto t0 = std::chrono::steady_clock::now();
        Reader r(in.data(), in.size());
        std::string why;
        bool sent = true;
        switch ((Op) h.op) {
        case Op::Reset:
            sent = ex.reset(why) ? answer(h, 0, Writer()) : refuse(h, why);
            break;
        case Op::Sampler: {
            const uint8_t* p = r.bytes(sizeof(SamplerWire));
            SamplerWire sp;
            if (p == nullptr) { sent = refuse(h, "a short sampler message"); break; }
            std::memcpy(&sp, p, sizeof sp);
            sent = ex.sampler(sp, why) ? answer(h, 0, Writer()) : refuse(h, why);
            break;
        }
        case Op::Forward: {
            const int32_t T = r.get<int32_t>();
            (void) r.get<int32_t>();
            const int64_t pos = r.get<int64_t>();
            if (!r.ok() || T < 1 || T > shape.max_t) { sent = refuse(h, "a window out of range"); break; }
            const uint8_t* tok = r.bytes((size_t) (T + (T & 1)) * 4);   // padded to 8 bytes: the rows stay aligned
            const float* x = tok != nullptr ? take_rows(r, h.flags, (size_t) T * (size_t) shape.hand_w) : nullptr;
            if (x == nullptr) { sent = refuse(h, "a window's rows are short"); break; }
            const bool want = (h.flags & kFlagLogits) != 0;
            out.assign((size_t) T, 0);
            if (want) logits.resize((size_t) T * (size_t) shape.n_vocab);
            std::vector<int32_t> tokens((size_t) T);
            std::memcpy(tokens.data(), tok, (size_t) T * 4);
            if (!ex.forward(tokens.data(), T, pos, x, (h.flags & kFlagDefer) != 0, out.data(), want ? logits.data() : nullptr,
                            why)) {
                sent = refuse(h, why);
                break;
            }
            Writer w;
            w.put<float>((float) ms_since(t0)).put<int32_t>(T).bytes(out.data(), (size_t) T * 4);
            sent = answer(h, want ? kFlagLogits : 0, w, want ? logits.data() : nullptr,
                          want ? (size_t) T * (size_t) shape.n_vocab * 4 : 0);
            break;
        }
        case Op::PromptBegin: {
            const int64_t n = r.get<int64_t>(), pos = r.get<int64_t>(), T = r.get<int64_t>();
            const uint8_t* tok = r.ok() && n > 0 && n <= shape.max_cells ? r.bytes((size_t) n * 4) : nullptr;
            if (tok == nullptr || T < 1) { sent = refuse(h, "a prompt out of range"); break; }
            if (in_prompt) { std::string e2; (void) ex.prompt_end(e2); in_prompt = false; }
            std::vector<int32_t> tokens((size_t) n);
            std::memcpy(tokens.data(), tok, (size_t) n * 4);
            int64_t longest = 0, cap = 0;
            bool too_short = false;
            if (!ex.prompt_begin(tokens.data(), n, pos, T, longest, cap, too_short, why)) {
                // too short is an answer, not a failure: the caller reads the prompt through the windows instead
                sent = too_short ? answer(h, kFlagShort, Writer().put<int64_t>(0).put<int64_t>(0)) : refuse(h, why);
                break;
            }
            in_prompt = true;
            prompt_n = n;
            sent = answer(h, 0, Writer().put(longest).put(cap));
            break;
        }
        case Op::PromptChunk: {
            const int64_t c0 = r.get<int64_t>(), c1 = r.get<int64_t>();
            if (!in_prompt || !r.ok() || c0 < 0 || c1 <= c0 || c0 >= prompt_n) { sent = refuse(h, "a chunk out of its prompt"); break; }
            const int64_t n_rows = std::min(c1, prompt_n) - c0;
            const float* x = take_rows(r, h.flags, (size_t) n_rows * (size_t) shape.hand_w);
            if (x == nullptr) { sent = refuse(h, "a chunk's rows are short"); break; }
            const bool final = c1 >= prompt_n, want = final && (h.flags & kFlagLogits) != 0;
            int32_t pick = 0;
            if (want) logits.resize((size_t) shape.n_vocab);
            if (!ex.prompt_chunk(c0, c1, x, &pick, want ? logits.data() : nullptr, why)) { sent = refuse(h, why); break; }
            Writer w;
            w.put<float>((float) ms_since(t0)).put<int32_t>(final ? 1 : 0).put<int32_t>(pick).put<int32_t>(0);
            sent = answer(h, want ? kFlagLogits : 0, w, want ? logits.data() : nullptr, want ? (size_t) shape.n_vocab * 4 : 0);
            break;
        }
        case Op::PromptEnd:
            in_prompt = false;
            sent = ex.prompt_end(why) ? answer(h, 0, Writer()) : refuse(h, why);
            break;
        case Op::Save: {
            const uint64_t id = r.get<uint64_t>();
            const int64_t tokens = r.get<int64_t>();
            uint64_t bytes = 0;
            if (!r.ok()) { sent = refuse(h, "a short save message"); break; }
            sent = ex.save(id, tokens, (h.flags & kFlagRecurrent) != 0, bytes, why) ? answer(h, 0, Writer().put(bytes))
                                                                                    : refuse(h, why);
            break;
        }
        case Op::Restore: {
            const uint64_t id = r.get<uint64_t>();
            if (!r.ok()) { sent = refuse(h, "a short restore message"); break; }
            sent = ex.restore(id, (h.flags & kFlagRecurrent) != 0, why) ? answer(h, 0, Writer()) : refuse(h, why);
            break;
        }
        case Op::Drop:
            ex.drop(r.get<uint64_t>());
            sent = answer(h, 0, Writer());
            break;
        case Op::Stats: {
            const std::string j = "{" + ex.stats_json() + "}";
            Writer w;
            w.bytes(j.data(), j.size());
            sent = answer(h, 0, w);
            break;
        }
        case Op::Bye:
            (void) answer(h, 0, Writer());
            return false;
        default:
            sent = refuse(h, "an operation this worker does not know");
        }
        busy_ms += ms_since(t0);
        stats_event();
        if (!sent) err = "hybrid link: the caller went away before its answer";
        return sent;
    }
};

}  // namespace

bool serve(SegmentExecutor& ex, const ServeOptions& opt, std::string& err) {
    if (opt.token.empty() && !loopback(opt.host)) {
        err = "hybrid segment: a worker that listens beyond this machine needs a token";
        return false;
    }
    const net::Socket listener = net::listen_on(opt.host, opt.port, err);
    if (listener == net::kNoSocket) return false;
    auto say = [&](const std::string& j) { if (opt.on_event) opt.on_event(j); };
    say("{\"event\":\"listening\",\"port\":" + std::to_string(net::local_port(listener)) + "}");
    int sessions = 0;
    while (opt.stop == nullptr || !opt.stop->load()) {
        std::string peer, e;
        const net::Socket s = net::accept_one(listener, 500, &peer, e);
        if (s == net::kNoSocket) {
            if (e.empty()) continue;
            err = e;
            net::close(listener);
            return false;
        }
        Session ss(ex, opt, s);
        std::string why;
        if (ss.handshake(why)) {
            say("{\"event\":\"connected\",\"peer\":\"" + json_escape(peer) + "\"}");
            why.clear();
            while (opt.stop == nullptr || !opt.stop->load()) {
                const int r = net::wait_readable(s, 500);
                if (r == 0) continue;
                if (r < 0 || !ss.step(why)) break;
            }
            // the caller is gone: its prompt's buffers go back and its sequence ends; saved states stay
            std::string e2;
            if (ss.in_prompt) (void) ex.prompt_end(e2);
            (void) ex.reset(e2);
        }
        say("{\"event\":\"disconnected\",\"peer\":\"" + json_escape(peer) + "\",\"reason\":\"" + json_escape(why) + "\"}");
        net::close(s);
        if (opt.max_sessions > 0 && ++sessions >= opt.max_sessions) break;
    }
    net::close(listener);
    return true;
}

}  // namespace strata::hybrid
