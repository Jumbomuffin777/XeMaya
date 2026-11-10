// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/platform/net.cpp - XeMaya: blocking TCP sockets on Linux and on Windows (see the header).
#include "strata/platform/net.hpp"

#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_arg = int;
using native_socket = SOCKET;
#else
#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
using socklen_arg = socklen_t;
using native_socket = int;
#endif

namespace strata::platform::net {

namespace {

native_socket nat(Socket s) { return (native_socket) s; }

std::string last_error() {
#if defined(_WIN32)
    return "winsock error " + std::to_string(WSAGetLastError());
#else
    return std::strerror(errno);
#endif
}

void close_native(native_socket s) {
#if defined(_WIN32)
    ::closesocket(s);
#else
    ::close(s);
#endif
}

// readable (or, `write`, writable) within timeout_ms: 1 yes, 0 timed out, -1 error
int wait_for(native_socket s, bool write, int timeout_ms) {
#if defined(_WIN32)
    WSAPOLLFD p{};
    p.fd = s;
    p.events = (SHORT) (write ? POLLWRNORM : POLLRDNORM);
    const int r = WSAPoll(&p, 1, timeout_ms);
#else
    pollfd p{};
    p.fd = s;
    p.events = (short) (write ? POLLOUT : POLLIN);
    int r = 0;
    do r = ::poll(&p, 1, timeout_ms);
    while (r < 0 && errno == EINTR);
#endif
    return r > 0 ? 1 : r;
}

bool set_blocking(native_socket s, bool blocking) {
#if defined(_WIN32)
    u_long mode = blocking ? 0 : 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
#else
    const int f = fcntl(s, F_GETFL, 0);
    return f >= 0 && fcntl(s, F_SETFL, blocking ? f & ~O_NONBLOCK : f | O_NONBLOCK) == 0;
#endif
}

struct Addr {
    addrinfo* list = nullptr;
    ~Addr() { if (list != nullptr) freeaddrinfo(list); }
};

bool resolve(const std::string& host, int port, bool passive, Addr& a, std::string& err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = passive ? AI_PASSIVE : 0;
    const std::string p = std::to_string(port);
    const int r = getaddrinfo(host.empty() ? nullptr : host.c_str(), p.c_str(), &hints, &a.list);
    if (r != 0 || a.list == nullptr) {
        err = "net: no address for \"" + host + "\"";
        return false;
    }
    return true;
}

}  // namespace

bool init(std::string& err) {
#if defined(_WIN32)
    static const int started = [] { WSADATA d; return WSAStartup(MAKEWORD(2, 2), &d); }();
    if (started != 0) { err = "net: WSAStartup failed"; return false; }
#else
    static const bool ignored = [] { std::signal(SIGPIPE, SIG_IGN); return true; }();
    (void) ignored;
    (void) err;
#endif
    return true;
}

Socket listen_on(const std::string& host, int port, std::string& err) {
    if (!init(err)) return kNoSocket;
    Addr a;
    if (!resolve(host, port, true, a, err)) return kNoSocket;
    for (const addrinfo* i = a.list; i != nullptr; i = i->ai_next) {
        const native_socket s = ::socket(i->ai_family, i->ai_socktype, i->ai_protocol);
        if ((Socket) s == kNoSocket) continue;
        const int one = 1;
#if !defined(_WIN32)   // Windows' SO_REUSEADDR lets a second program take a port in use
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
#else
        setsockopt(s, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*) &one, sizeof one);
#endif
        if (::bind(s, i->ai_addr, (socklen_arg) i->ai_addrlen) == 0 && ::listen(s, 4) == 0) return (Socket) s;
        err = "net: cannot listen on " + host + ":" + std::to_string(port) + ": " + last_error();
        close_native(s);
    }
    if (err.empty()) err = "net: cannot listen on " + host + ":" + std::to_string(port);
    return kNoSocket;
}

int local_port(Socket s) {
    sockaddr_storage ss{};
    socklen_arg n = sizeof ss;
    if (getsockname(nat(s), (sockaddr*) &ss, &n) != 0) return 0;
    if (ss.ss_family == AF_INET) return ntohs(((const sockaddr_in*) &ss)->sin_port);
    if (ss.ss_family == AF_INET6) return ntohs(((const sockaddr_in6*) &ss)->sin6_port);
    return 0;
}

Socket accept_one(Socket listener, int timeout_ms, std::string* peer, std::string& err) {
    err.clear();
    const int w = wait_for(nat(listener), false, timeout_ms);
    if (w == 0) return kNoSocket;
    if (w < 0) { err = "net: waiting for a connection: " + last_error(); return kNoSocket; }
    sockaddr_storage ss{};
    socklen_arg n = sizeof ss;
    const native_socket s = ::accept(nat(listener), (sockaddr*) &ss, &n);
    if ((Socket) s == kNoSocket) { err = "net: accept: " + last_error(); return kNoSocket; }
    if (peer != nullptr) {
        char host[NI_MAXHOST] = {}, serv[NI_MAXSERV] = {};
        if (getnameinfo((sockaddr*) &ss, n, host, sizeof host, serv, sizeof serv, NI_NUMERICHOST | NI_NUMERICSERV) == 0)
            *peer = std::string(host) + ":" + serv;
    }
    tune((Socket) s);
    return (Socket) s;
}

Socket connect_to(const std::string& host, int port, int timeout_ms, std::string& err) {
    if (!init(err)) return kNoSocket;
    Addr a;
    if (!resolve(host, port, false, a, err)) return kNoSocket;
    for (const addrinfo* i = a.list; i != nullptr; i = i->ai_next) {
        const native_socket s = ::socket(i->ai_family, i->ai_socktype, i->ai_protocol);
        if ((Socket) s == kNoSocket) continue;
        // a bounded connect: non-blocking, wait until writable, read the result
        bool ok = set_blocking(s, false);
        if (ok) {
            const int r = ::connect(s, i->ai_addr, (socklen_arg) i->ai_addrlen);
#if defined(_WIN32)
            const bool pending = r != 0 && WSAGetLastError() == WSAEWOULDBLOCK;
#else
            const bool pending = r != 0 && errno == EINPROGRESS;
#endif
            ok = r == 0;
            if (pending && wait_for(s, true, timeout_ms > 0 ? timeout_ms : -1) == 1) {
                int e = 0;
                socklen_arg n = sizeof e;
                ok = getsockopt(s, SOL_SOCKET, SO_ERROR, (char*) &e, &n) == 0 && e == 0;
            }
        }
        if (ok && set_blocking(s, true)) {
            tune((Socket) s);
            return (Socket) s;
        }
        close_native(s);
    }
    err = "net: cannot connect to " + host + ":" + std::to_string(port);
    return kNoSocket;
}

void tune(Socket s) {
    const int one = 1;
    setsockopt(nat(s), IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    setsockopt(nat(s), SOL_SOCKET, SO_KEEPALIVE, (const char*) &one, sizeof one);
    // room for a prompt chunk's rows (the system caps what it grants; less only means more round trips of the stack)
    const int buf = 8 << 20;
    setsockopt(nat(s), SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof buf);
    setsockopt(nat(s), SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof buf);
}

bool send_all(Socket s, const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    while (n > 0) {
        const size_t part = n < ((size_t) 1 << 30) ? n : ((size_t) 1 << 30);
#if defined(_WIN32)
        const int r = ::send(nat(s), c, (int) part, 0);
        if (r <= 0) return false;
#else
        const ssize_t r = ::send(nat(s), c, part, MSG_NOSIGNAL);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
#endif
        c += r;
        n -= (size_t) r;
    }
    return true;
}

bool recv_all(Socket s, void* p, size_t n, int timeout_ms) {
    char* c = static_cast<char*>(p);
    while (n > 0) {
        if (timeout_ms > 0 && wait_for(nat(s), false, timeout_ms) != 1) return false;
        const size_t part = n < ((size_t) 1 << 30) ? n : ((size_t) 1 << 30);
#if defined(_WIN32)
        const int r = ::recv(nat(s), c, (int) part, 0);
        if (r <= 0) return false;
#else
        const ssize_t r = ::recv(nat(s), c, part, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
#endif
        c += r;
        n -= (size_t) r;
    }
    return true;
}

int wait_readable(Socket s, int timeout_ms) { return wait_for(nat(s), false, timeout_ms); }

void close(Socket s) {
    if (s == kNoSocket) return;
#if defined(_WIN32)
    ::shutdown(nat(s), SD_BOTH);
#else
    ::shutdown(nat(s), SHUT_RDWR);
#endif
    close_native(nat(s));
}

bool split_host_port(const std::string& s, std::string& host, int& port) {
    const size_t c = s.rfind(':');
    if (c == std::string::npos || c + 1 >= s.size()) return false;
    char* end = nullptr;
    const long p = std::strtol(s.c_str() + c + 1, &end, 10);
    if (*end != 0 || p < 0 || p > 65535) return false;
    host = s.substr(0, c);
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
    port = (int) p;
    return true;
}

}  // namespace strata::platform::net
