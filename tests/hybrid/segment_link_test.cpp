// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/hybrid/segment_link_test.cpp - XeMaya: the hybrid segment link over the loopback, without a GPU.
//
// A stand-in takes the place of the model's last blocks: it checks that the rows, tokens and positions arrive as they
// were sent and answers with values computed from them.  What is checked: the token proof (and its refusal), windows
// and prompt chunks in FP32 (bit for bit) and in halves, the logits back, saved states, an executor's error (the link
// stays up), a caller that vanishes inside a prompt (the worker gives its buffers back and takes the next caller),
// and noise on the port.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <thread>

#include "strata/hybrid/segment.hpp"

using namespace strata::hybrid;
namespace net = strata::platform::net;

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

constexpr int64_t kW = 48, kVocab = 97, kMaxT = 8, kCells = 4096;

float row_value(int64_t pos, int64_t i) { return (float) std::sin(0.37 * (double) pos + 0.11 * (double) i) * 3.5f; }

// the pick of a row: a function of its position, token and values, so a wrong byte anywhere changes it
int32_t pick_of(int64_t pos, int32_t token, const float* row) {
    double s = 0;
    for (int64_t i = 0; i < kW; ++i) s += (double) row[i] * (double) (i + 1);
    return (int32_t) ((pos * 31 + token * 7 + (int64_t) std::llround(s * 16.0)) % kVocab + kVocab) % (int32_t) kVocab;
}

struct Mock : SegmentExecutor {
    std::mutex m;
    int resets = 0, prompt_ends = 0, prompts_open = 0;
    bool exact = true;       // the rows must equal row_value bit for bit (FP32); else within a half's rounding
    bool rows_ok = true;
    int64_t next_pos = 0;
    std::vector<int32_t> prompt;
    int64_t prompt_pos = 0;
    std::map<uint64_t, int64_t> saved;
    SamplerWire sp;

    Shape shape() override {
        Shape s;
        s.fingerprint = 0xfeedbeefcafe1234ull;
        s.epoch = 42;
        s.seg_first = 30; s.seg_last = 40; s.n_layer = 40;
        s.n_embd = kW; s.hand_w = kW; s.n_vocab = kVocab; s.max_t = kMaxT; s.max_cells = kCells;
        s.prompt_chunk = 64; s.hand_cap = 64;
        s.arch = "mock"; s.device = "no GPU";
        return s;
    }
    void check_row(int64_t pos, const float* row) {
        for (int64_t i = 0; i < kW; ++i) {
            const float want = row_value(pos, i);
            if (exact ? row[i] != want : std::fabs(row[i] - want) > 2e-3f) rows_ok = false;
        }
    }
    bool reset(std::string&) override { std::lock_guard<std::mutex> g(m); ++resets; next_pos = 0; return true; }
    bool sampler(const SamplerWire& s, std::string&) override { sp = s; return true; }
    bool forward(const int32_t* tokens, int T, int64_t pos, const float* rows, bool, int32_t* out, float* logits,
                 std::string& err) override {
        if (tokens[0] == 666) { err = "the stand-in refuses token 666"; return false; }
        for (int t = 0; t < T; ++t) {
            check_row(pos + t, rows + (size_t) t * kW);
            out[t] = pick_of(pos + t, tokens[t], rows + (size_t) t * kW);
            if (logits != nullptr)
                for (int64_t v = 0; v < kVocab; ++v) logits[(size_t) t * kVocab + (size_t) v] = v == out[t] ? 9.0f : (float) v * 0.01f;
        }
        next_pos = pos + T;
        return true;
    }
    bool prompt_begin(const int32_t* tokens, int64_t n, int64_t pos, int64_t T, int64_t& longest, int64_t& cap,
                      bool& too_short, std::string& err) override {
        if (n == 7) { too_short = true; err = "too little to lend"; return false; }
        std::lock_guard<std::mutex> g(m);
        prompt.assign(tokens, tokens + n);
        prompt_pos = pos;
        longest = std::min<int64_t>(T, 32);
        cap = 64;
        ++prompts_open;
        return true;
    }
    bool prompt_chunk(int64_t c0, int64_t c1, const float* rows, int32_t* out, float*logits, std::string&) override {
        const int64_t n = (int64_t) prompt.size(), end = std::min(c1, n);
        for (int64_t c = c0; c < end; ++c) check_row(prompt_pos + c, rows + (size_t) (c - c0) * kW);
        if (c1 >= n) {
            *out = pick_of(prompt_pos + n - 1, prompt[(size_t) n - 1], rows + (size_t) (end - 1 - c0) * kW);
            if (logits != nullptr)
                for (int64_t v = 0; v < kVocab; ++v) logits[(size_t) v] = v == *out ? 5.0f : 0.0f;
            next_pos = prompt_pos + n;
        }
        return true;
    }
    bool prompt_end(std::string&) override { std::lock_guard<std::mutex> g(m); ++prompt_ends; --prompts_open; return true; }
    bool save(uint64_t id, int64_t tokens, bool, uint64_t& bytes, std::string&) override { saved[id] = tokens; bytes = 1000; return true; }
    bool restore(uint64_t id, bool, std::string& err) override {
        if (saved.count(id) == 0) { err = "this worker no longer holds that state"; return false; }
        next_pos = saved[id];
        return true;
    }
    void drop(uint64_t id) override { saved.erase(id); }
    std::string stats_json() override { return "\"windows\":" + std::to_string(next_pos); }
};

std::vector<float> rows_at(int64_t pos, int64_t n) {
    std::vector<float> x((size_t) (n * kW));
    for (int64_t t = 0; t < n; ++t)
        for (int64_t i = 0; i < kW; ++i) x[(size_t) (t * kW + i)] = row_value(pos + t, i);
    return x;
}

void test_primitives() {
    uint8_t d[32];
    sha256("abc", 3, d);
    const uint8_t abc[4] = {0xba, 0x78, 0x16, 0xbf};
    CHECK(std::memcmp(d, abc, 4) == 0 && d[31] == 0xad);
    // RFC 4231, test case 2
    hmac_sha256("Jefe", 4, "what do ya want for nothing?", 28, d);
    const uint8_t rfc[4] = {0x5b, 0xdc, 0xc1, 0x46};
    CHECK(std::memcmp(d, rfc, 4) == 0 && d[31] == 0x43);
    // halves: exact values survive, the rest round to the nearest, the ends saturate
    const float in[10] = {0.0f, -0.0f, 1.0f, -2.5f, 65504.0f, 1e9f, 6.1e-5f, 1.0f + 1.0f / 2048.0f, 3.14159265f, -1e-9f};
    uint16_t h[10];
    float back[10];
    f32_to_f16(in, h, 10);
    f16_to_f32(h, back, 10);
    CHECK(back[0] == 0.0f && back[2] == 1.0f && back[3] == -2.5f && back[4] == 65504.0f);
    CHECK(std::isinf(back[5]) && h[1] == 0x8000 && back[9] == 0.0f);
    CHECK(std::fabs(back[6] - 6.1e-5f) < 1e-7f);               // a subnormal half
    CHECK(back[7] == 1.0f);                                     // a tie rounds to the even mantissa
    CHECK(std::fabs(back[8] - 3.14159265f) < 1e-3f);
    std::string host;
    int port = 0;
    CHECK(net::split_host_port("10.1.2.3:7411", host, port) && host == "10.1.2.3" && port == 7411);
    CHECK(net::split_host_port("[::1]:9", host, port) && host == "::1" && port == 9);
    CHECK(!net::split_host_port("nohost", host, port));
}

void run_caller(int port, Mock& mock, bool f16) {
    const std::string addr = "127.0.0.1:" + std::to_string(port);
    std::string err;
    {   // the wrong token is refused, and says so
        RemoteSegment bad;
        CHECK(!bad.connect(addr, "not the token", false, err));
        CHECK(err.find("token") != std::string::npos);
    }
    mock.exact = !f16;
    RemoteSegment seg;
    CHECK(seg.connect(addr, "s3cret", f16, err));
    if (!seg.connected()) { std::fprintf(stderr, "connect: %s\n", err.c_str()); return; }
    CHECK(seg.shape().hand_w == kW && seg.shape().n_vocab == kVocab && seg.shape().seg_first == 30);
    CHECK(seg.shape().arch == "mock" && seg.shape().epoch == 42);
    CHECK(seg.reset(err));
    SamplerWire sp;
    sp.temperature = 0.7f;
    sp.seed = 99;
    CHECK(seg.sampler(sp, err) && mock.sp.seed == 99 && mock.sp.temperature == 0.7f);

    // windows of 1 .. max_t rows, odd and even (the padding before the rows)
    int64_t pos = 0;
    for (int T = 1; T <= (int) kMaxT; ++T) {
        std::vector<int32_t> tok((size_t) T), out((size_t) T, -1);
        for (int t = 0; t < T; ++t) tok[(size_t) t] = (int32_t) (pos + t) % 50;
        const std::vector<float> x = rows_at(pos, T);
        std::vector<float> logits((size_t) T * kVocab);
        CHECK(seg.forward(tok.data(), T, pos, x.data(), false, out.data(), T % 2 ? logits.data() : nullptr, err));
        for (int t = 0; t < T; ++t) {
            // in halves the worker sees rounded rows; the stand-in's own check bounds them, the pick is its own
            if (!f16) CHECK(out[(size_t) t] == pick_of(pos + t, tok[(size_t) t], x.data() + (size_t) t * kW));
            if (T % 2) CHECK(logits[(size_t) t * kVocab + (size_t) out[(size_t) t]] == 9.0f);
        }
        pos += T;
    }
    CHECK(mock.rows_ok);

    // an executor's refusal is the call's error, and the link goes on
    {
        const int32_t bad = 666;
        int32_t out = 0;
        const std::vector<float> x = rows_at(pos, 1);
        CHECK(!seg.forward(&bad, 1, pos, x.data(), false, &out, nullptr, err));
        CHECK(err.find("666") != std::string::npos && seg.connected());
    }

    // a prompt of 100 tokens in chunks of the worker's 32
    {
        const int64_t n = 100, p0 = 0;
        std::vector<int32_t> tok((size_t) n);
        for (int64_t i = 0; i < n; ++i) tok[(size_t) i] = (int32_t) (i * 3 % 50);
        int64_t longest = 0, cap = 0;
        bool too_short = false;
        CHECK(seg.prompt_begin(tok.data(), n, p0, 4096, longest, cap, too_short, err));
        CHECK(longest == 32 && cap == 64 && !too_short);
        int32_t pick = -1;
        std::vector<float> logits((size_t) kVocab);
        for (int64_t c0 = 0; c0 < n; c0 += longest) {
            const std::vector<float> x = rows_at(p0 + c0, std::min(longest, n - c0));
            CHECK(seg.prompt_chunk(c0, c0 + longest, n, x.data(), &pick, logits.data(), err));
        }
        const std::vector<float> last = rows_at(p0 + n - 1, 1);
        if (!f16) CHECK(pick == pick_of(p0 + n - 1, tok[(size_t) n - 1], last.data()));
        CHECK(pick >= 0 && logits[(size_t) pick] == 5.0f);
        CHECK(seg.prompt_end(err));
        CHECK(mock.rows_ok && mock.prompts_open == 0);
        // a worker whose tier lends too little: not an error, the caller takes the windows
        CHECK(!seg.prompt_begin(tok.data(), 7, 0, 4096, longest, cap, too_short, err));
        CHECK(too_short && err.empty() && seg.connected());
    }

    // states stay on the worker under their ids
    CHECK(seg.save(5, 77, false, err));
    CHECK(seg.restore(5, false, err) && mock.next_pos == 77);
    seg.drop(5);
    CHECK(!seg.restore(5, false, err) && err.find("no longer holds") != std::string::npos);
    std::string json;
    CHECK(seg.stats(json, err) && json.find("\"windows\":77") != std::string::npos);
    CHECK(seg.link().calls > 10 && seg.link().bytes_out > seg.link().bytes_in);

    // the caller vanishes inside a prompt: no Bye, the socket just closes
    {
        RemoteSegment gone;
        seg.close();
        CHECK(gone.connect(addr, "s3cret", f16, err));
        std::vector<int32_t> tok(40, 1);
        int64_t longest = 0, cap = 0;
        bool too_short = false;
        CHECK(gone.prompt_begin(tok.data(), 40, 0, 4096, longest, cap, too_short, err));
        CHECK(mock.prompts_open == 1);
    }   // ~RemoteSegment sends Bye and closes: the worker must give the prompt's buffers back
    // noise on the port: the worker drops it and keeps serving
    {
        std::string e;
        const net::Socket raw = net::connect_to("127.0.0.1", port, 2000, e);
        CHECK(raw != net::kNoSocket);
        const char junk[64] = "GET / HTTP/1.1\r\n\r\n";
        (void) net::send_all(raw, junk, sizeof junk);
        net::close(raw);
    }
    RemoteSegment again;
    CHECK(again.connect(addr, "s3cret", f16, err));
    CHECK(mock.prompts_open == 0);
    CHECK(again.reset(err));
}

}  // namespace

int main() {
    test_primitives();
    for (const bool f16 : {false, true}) {
        Mock mock;
        std::atomic<bool> stop{false};
        std::atomic<int> port{0};
        ServeOptions opt;
        opt.token = "s3cret";
        opt.stop = &stop;
        opt.on_event = [&](const std::string& j) {
            const size_t at = j.find("\"port\":");
            if (at != std::string::npos) port = std::atoi(j.c_str() + at + 7);
        };
        std::string serr;
        bool served = false;
        std::thread worker([&] { served = serve(mock, opt, serr); });
        for (int i = 0; i < 500 && port == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(port != 0);
        if (port != 0) run_caller(port, mock, f16);
        stop = true;
        worker.join();
        CHECK(served);
        CHECK(mock.resets >= 3);   // one for each caller that left
    }
    {   // a worker open to the network without a token does not start
        Mock mock;
        ServeOptions opt;
        opt.host = "0.0.0.0";
        std::string err;
        CHECK(!serve(mock, opt, err) && err.find("token") != std::string::npos);
    }
    if (failures != 0) {
        std::fprintf(stderr, "segment_link_test: %d checks failed\n", failures);
        return 1;
    }
    std::printf("segment_link_test: PASS\n");
    return 0;
}
