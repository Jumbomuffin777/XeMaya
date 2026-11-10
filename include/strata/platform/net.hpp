// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/platform/net.hpp - XeMaya: blocking TCP sockets, the same calls on Linux and on Windows (Winsock).
//
// For the hybrid segment link (strata/hybrid/): one connection between two engines, a few large messages a token or a
// prompt chunk.  No name service beyond getaddrinfo, no TLS: the link is a private network's, and the peers prove a
// shared token to each other before anything else (hybrid/wire.hpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::platform::net {

using Socket = intptr_t;
inline constexpr Socket kNoSocket = -1;

/// Once before any other call (Windows: WSAStartup; elsewhere: SIGPIPE ignored, so a peer that went away is an error
/// of the send, not the end of the process).  Safe to call again.
bool init(std::string& err);
/// A listening socket on host:port (port 0: any free one; local_port says which).  `host` empty: every interface.
Socket listen_on(const std::string& host, int port, std::string& err);
int local_port(Socket s);
/// The next connection, or kNoSocket: with `err` empty when none came within timeout_ms (< 0: wait for ever).
Socket accept_one(Socket listener, int timeout_ms, std::string* peer, std::string& err);
Socket connect_to(const std::string& host, int port, int timeout_ms, std::string& err);
/// Small messages go out at once (TCP_NODELAY), the buffers hold a prompt chunk's rows, and a peer that vanished
/// without closing is noticed (keepalive).
void tune(Socket s);
/// All n bytes, or false (the peer closed, an error, or - recv - nothing for timeout_ms; <= 0: no limit).
bool send_all(Socket s, const void* p, size_t n);
bool recv_all(Socket s, void* p, size_t n, int timeout_ms);
/// Whether something can be read within timeout_ms: 1 yes (data, or the peer's close), 0 not yet, -1 an error.
int wait_readable(Socket s, int timeout_ms);
void close(Socket s);
/// "host:port" -> its parts ("[::1]:7" too); false when there is no port.
bool split_host_port(const std::string& s, std::string& host, int& port);

}  // namespace strata::platform::net
