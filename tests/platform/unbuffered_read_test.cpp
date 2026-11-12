// SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// tests/platform/unbuffered_read_test.cpp - unbuffered_read against the same ranges read through the file cache:
// every alignment of the offset, the length and the destination, the file's last bytes, a range past its end.
//   unbuffered_read_test [DIR]            the file is written in DIR (default: the working directory)
//   unbuffered_read_test --bench FILE [THREADS [MIB]]   FILE's first 4 GiB in ranges of MIB (default 8) on THREADS
//                                                       threads (default 8), unbuffered
// Exit 0 with "SKIP" where the platform or DIR's file system has no unbuffered reads.
#include "strata/platform/unbuffered_read.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace sp = strata::platform;

static int bench(const std::string& path, int threads, int mib) {
    const int fd = sp::unbuffered_open(path);
    if (fd < 0) { std::printf("unbuffered_read_test: SKIP (no unbuffered reads of %s)\n", path.c_str()); return 0; }
    const uint64_t range = (uint64_t) mib << 20, total = 4ull << 30;
    std::atomic<uint64_t> next{0}, got{0};
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&] {
            std::vector<uint8_t> buf((size_t) range + 1);
            for (uint64_t at; (at = next.fetch_add(range)) < total;)
                if (sp::unbuffered_read(fd, buf.data() + 1, range, at + 123)) got += range;
        });
    for (auto& x : th) x.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    sp::unbuffered_close(fd);
    std::printf("unbuffered_read_test: %.2f GiB in %.2f s, %.2f GiB/s on %d threads (%d MiB a read)\n",
                (double) got / 1073741824.0, s, (double) got / 1073741824.0 / s, threads, mib);
    return got > 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 2 && std::strcmp(argv[1], "--bench") == 0)
        return bench(argv[2], argc > 3 ? std::max(1, std::atoi(argv[3])) : 8, argc > 4 ? std::max(1, std::atoi(argv[4])) : 8);
    const std::string path = std::string(argc > 1 ? argv[1] : ".") + "/unbuffered-read-test.bin";
    const size_t size = 3 * 1048576 + 1234;   // not a multiple of a block
    std::vector<uint8_t> want(size);
    std::mt19937 rng(20261111);
    for (auto& b : want) b = (uint8_t) rng();
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write((const char*) want.data(), (std::streamsize) size);
        if (!f) { std::fprintf(stderr, "unbuffered_read_test: cannot write %s\n", path.c_str()); return 1; }
    }
    const int fd = sp::unbuffered_open(path);
    if (fd < 0) {
        std::remove(path.c_str());
        std::printf("unbuffered_read_test: SKIP (no unbuffered reads of %s)\n", path.c_str());
        return 0;
    }
    int bad = 0, n_ok = 0;
    std::vector<uint8_t> raw(size + 8192 + 64);
    uint8_t* aligned = raw.data() + (4096 - (uintptr_t) raw.data() % 4096) % 4096;
    auto one = [&](uint64_t off, uint64_t n, int shift) {
        uint8_t* dst = aligned + shift;
        std::memset(dst, 0xA5, (size_t) n + 1);
        const bool ok = sp::unbuffered_read(fd, dst, n, off);
        if (!ok || std::memcmp(dst, want.data() + off, (size_t) n) != 0 || dst[n] != 0xA5) {
            std::fprintf(stderr, "FAIL offset %llu length %llu shift %d (%s)\n", (unsigned long long) off,
                         (unsigned long long) n, shift, ok ? "bytes differ" : "read failed");
            ++bad;
        } else {
            ++n_ok;
        }
    };
    const uint64_t offs[] = {0, 1, 511, 512, 4095, 4096, 4097, 8192, 1048576 + 7, 2 * 1048576};
    const uint64_t lens[] = {0, 1, 511, 4096, 4097, 8192, 65536 + 3, 1048576, 1048576 + 4095};
    for (uint64_t off : offs)
        for (uint64_t n : lens)
            for (int shift : {0, 1, 512})
                if (off + n <= size) one(off, n, shift);
    for (int shift : {0, 3}) {   // the file's last bytes, and the whole file
        one(size - 1, 1, shift);
        one(size - 5000, 5000, shift);
        one(4096, size - 4096, shift);
        one(0, size, shift);
    }
    std::uniform_int_distribution<uint64_t> any(0, size - 1);
    for (int i = 0; i < 200; ++i) {
        const uint64_t off = any(rng), n = std::min<uint64_t>(any(rng) % 300000, size - off);
        one(off, n, (int) (rng() % 4096));
    }
    // a range that runs past the end of the file is refused, in place and through the buffer
    if (sp::unbuffered_read(fd, aligned, 8192, size - 100) || sp::unbuffered_read(fd, aligned + 1, 4096, size - 4095) ||
        sp::unbuffered_read(fd, aligned, 8192, (size / 4096) * 4096)) {
        std::fprintf(stderr, "FAIL a range past the end of the file was read\n");
        ++bad;
    }
    sp::unbuffered_close(fd);
    std::remove(path.c_str());
    std::printf("unbuffered_read_test: %d ranges equal, %d failed: %s\n", n_ok, bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
