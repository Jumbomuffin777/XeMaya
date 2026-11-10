// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/core/plain_segment.hpp - XeMaya: a PlainModel's last blocks behind the hybrid segment link.
#pragma once

#include <cstdint>
#include <string>

#include "strata/core/plain_model.hpp"
#include "strata/hybrid/segment.hpp"

namespace strata::core {

/// Serves `model` (loaded with PlainOptions::seg_first > 0) to the engine that holds the blocks before it, on
/// host:port, until the process is told to stop (SIGINT, SIGTERM; Ctrl-C and the console's close on Windows).
/// Prints one line a status change to stdout, `SEGMENT {json}`, for the program that manages the worker.
/// `state_mib`: the most the conversations set aside may take (the oldest go first).  The exit code.
int plain_segment_serve(PlainModel& model, const std::string& host, int port, const std::string& token,
                        int64_t state_mib);

}  // namespace strata::core
