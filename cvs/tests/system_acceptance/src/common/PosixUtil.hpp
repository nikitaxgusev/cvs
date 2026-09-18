/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#pragma once
//
// Small POSIX leaf helpers shared by the harness's process-control paths: the
// Runner watchdog, the SshLauncher fan-out, the local-ranks launcher, and the
// SocketBootstrap deadlines. Header-only (no extra TU); each function is a thin,
// dependency-free wrapper that was previously copy-pasted into several files.
//
#include <cerrno>
#include <ctime>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace acceptance {
namespace posix_util {

// EINTR-safe waitpid. `flags` may include WNOHANG. Returns the waitpid() value
// (pid, 0 for WNOHANG-no-change, or -1 with errno set).
inline pid_t waitpid_eintr(pid_t pid, int* status, int flags) {
  for (;;) {
    const pid_t r = ::waitpid(pid, status, flags);
    if (r < 0 && errno == EINTR) continue;
    return r;
  }
}

// Sleep `ms` milliseconds, resuming across EINTR so a watchdog/backoff cadence
// is not cut short by signal delivery.
inline void sleep_ms(long ms) {
  struct timespec req {};
  req.tv_sec = ms / 1000;
  req.tv_nsec = (ms % 1000) * 1000000L;
  struct timespec rem {};
  while (::nanosleep(&req, &rem) != 0 && errno == EINTR) {
    req = rem;
  }
}

// Monotonic clock in milliseconds, for deadlines/elapsed (never wall-clock, so
// it is immune to clock steps). Seconds are just `monotonic_ms() / 1000`.
inline long monotonic_ms() {
  struct timespec ts {};
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<long>(ts.tv_sec) * 1000L + ts.tv_nsec / 1000000L;
}

}  // namespace posix_util
}  // namespace acceptance
