// SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/platform/unbuffered_read.cpp - see include/strata/platform/unbuffered_read.hpp.
#include "strata/platform/unbuffered_read.hpp"

#if defined(__linux__)
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace strata::platform {

#if defined(__linux__)
namespace {
// the memory, offset and length alignment asked of every request: a multiple of any drive's logical block
constexpr uint64_t kAlign = 4096;
/// The calling thread's aligned buffer, grown to the largest window it has read.
struct Bounce {
    uint8_t* p = nullptr;
    uint64_t cap = 0;
    ~Bounce() { std::free(p); }
    uint8_t* take(uint64_t bytes) {
        if (bytes <= cap) return p;
        std::free(p);
        void* q = nullptr;
        p = posix_memalign(&q, kAlign, (size_t) bytes) == 0 ? (uint8_t*) q : nullptr;
        cap = p != nullptr ? bytes : 0;
        return p;
    }
};
thread_local Bounce tl_bounce;
/// `bytes` at `off` into `buf` (all aligned); how many arrived (fewer only at the end of the file or on an error).
uint64_t read_window(int fd, uint8_t* buf, uint64_t bytes, uint64_t off) {
    uint64_t done = 0;
    while (done < bytes) {
        const ssize_t r = ::pread(fd, buf + done, (size_t) (bytes - done), (off_t) (off + done));
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        done += (uint64_t) r;
        if ((uint64_t) r % kAlign != 0) break;   // the end of the file: a next request would not be aligned
    }
    return done;
}
}  // namespace

int unbuffered_open(const std::string& path) { return ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT); }

void unbuffered_close(int fd) {
    if (fd >= 0) ::close(fd);
}

bool unbuffered_read(int fd, uint8_t* dst, uint64_t n, uint64_t off) {
    if (fd < 0 || dst == nullptr) return false;
    if (n == 0) return true;
    const uint64_t a0 = off / kAlign * kAlign, a1 = (off + n + kAlign - 1) / kAlign * kAlign;
    if (off == a0 && (uintptr_t) dst % kAlign == 0) {   // in place: the whole blocks, then the tail through the buffer
        const uint64_t whole = n / kAlign * kAlign;
        if (whole > 0 && read_window(fd, dst, whole, off) != whole) return false;
        if (whole == n) return true;
        uint8_t* b = tl_bounce.take(kAlign);
        if (b == nullptr || read_window(fd, b, kAlign, off + whole) < n - whole) return false;
        std::memcpy(dst + whole, b, (size_t) (n - whole));
        return true;
    }
    uint8_t* b = tl_bounce.take(a1 - a0);
    if (b == nullptr || read_window(fd, b, a1 - a0, a0) < off - a0 + n) return false;
    std::memcpy(dst, b + (off - a0), (size_t) n);
    return true;
}
#else
int unbuffered_open(const std::string& path) { (void) path; return -1; }
void unbuffered_close(int fd) { (void) fd; }
bool unbuffered_read(int fd, uint8_t* dst, uint64_t n, uint64_t off) {
    (void) fd; (void) dst; (void) n; (void) off;
    return false;
}
#endif

}  // namespace strata::platform
