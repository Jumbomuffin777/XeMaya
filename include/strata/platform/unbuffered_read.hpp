// SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/platform/unbuffered_read.hpp - a byte range of a file read past the file cache (Linux: O_DIRECT).
//
// A buffered read of megabytes that the cache does not hold goes to the drive a read-ahead window at a time
// (/sys/block/<dev>/queue/read_ahead_kb, 128 KiB by default) and passes every byte through the cache; an unbuffered
// one is a single request the block layer splits and queues at once.  For data the cache cannot keep anyway (a model's
// experts larger than the RAM), read once and copied on to a GPU.  Where it is not available (not Linux, a file
// system that refuses O_DIRECT) `unbuffered_open` answers -1 and the caller keeps its buffered reads.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

/// The file opened for unbuffered reads; -1 when it cannot be.
int unbuffered_open(const std::string& path);
void unbuffered_close(int fd);
/// The `n` bytes at `off` into `dst` (any address, offset and length: the aligned window around the range is read,
/// straight into `dst` where its address allows, else through a buffer of the calling thread's).  False when a read
/// failed or the file ended before the range did: the caller reads it another way.
bool unbuffered_read(int fd, uint8_t* dst, uint64_t n, uint64_t off);

}  // namespace strata::platform
