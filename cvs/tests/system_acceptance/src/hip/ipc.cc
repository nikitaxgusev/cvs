// HIP acceptance tests — IPC (legacy hipIpc* handle path) group.
//
// Mirrors RCCL's p2pSendSetup/p2pRecvSetup: export a 64-byte handle, ship it to
// a peer, import with lazy-enable-peer-access. The test fork()s a child before
// touching HIP (so each process inits the runtime independently; the category is
// force-isolated for this reason). Handle + sync flags travel through an
// anonymous MAP_SHARED region (flags via std::atomic_ref); the importer verifies
// with a real kernel read. All waits are bounded so a wedged peer fails, not
// hangs. Built only when ACCEPT_WITH_HIP=ON.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <string>

#include <csignal>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "common/Perf.hpp"
#include "common/TestRegistry.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

namespace {

// FillArgs (parent write) and MismArgs (child verify) are shared in HipTest.hpp.

// Cross-process state in a MAP_SHARED region;
//  plain int flags accessed via std::atomic_ref for ordering.
struct IpcShared {
  int handle_ready;          // exporter -> importer: handle is populated
  int child_done;            // importer -> exporter: import+verify finished
  int child_ok;              // importer result: 1 = pattern verified
  hipIpcMemHandle_t handle;  // byte-copyable 64-byte IPC handle
};

// Importer (child) exit codes, surfaced in the parent's failure message.
enum ImporterExit {
  kSetDeviceFailed = 10,
  kHandleWaitTimeout = 11,
  kOpenHandleFailed = 12,
  kVerifyFailed = 13,
};

// Bounded acquire-wait on a shared flag: true if it became non-zero
//  within `timeout_ms`, false on timeout (1 ms polls).
bool wait_flag(int* flag, int timeout_ms) {
  std::atomic_ref<int> a(*flag);
  for (int waited = 0; waited < timeout_ms; ++waited) {
    if (a.load(std::memory_order_acquire) != 0) return true;
    struct timespec req {};
    req.tv_sec = 0;
    req.tv_nsec = 1'000'000L;  // 1 ms
    (void)nanosleep(&req, nullptr);
  }
  return a.load(std::memory_order_acquire) != 0;
}

// Bounded reap: WNOHANG-poll for the child, then SIGKILL+reap
// if still wedged, so the parent never blocks indefinitely.
void reap_child(pid_t pid, int* wstatus_out) {
  for (int waited = 0; waited < 2000; ++waited) {  // up to ~2 s
    int wstatus = 0;
    pid_t r = ::waitpid(pid, &wstatus, WNOHANG);
    if (r == pid) {
      if (wstatus_out) *wstatus_out = wstatus;
      return;
    }
    if (r == -1) {  // already reaped / no such child
      if (wstatus_out) *wstatus_out = 0;
      return;
    }
    struct timespec req {};
    req.tv_sec = 0;
    req.tv_nsec = 1'000'000L;  // 1 ms
    (void)nanosleep(&req, nullptr);
  }
  // Still alive after the grace period: force it down and reap (blocking is now
  // safe — the child is being killed).
  (void)::kill(pid, SIGKILL);
  int wstatus = 0;
  (void)::waitpid(pid, &wstatus, 0);
  if (wstatus_out) *wstatus_out = wstatus;
}

void publish_flag(int* flag) {
  std::atomic_ref<int> a(*flag);
  a.store(1, std::memory_order_release);
}

// Kernel-verify the imported mapping on-device; true only
// if the pattern reads back intact (RAII frees the counter).
bool ipc_kernel_verify_pattern(const void* d_in, int n) {
  DeviceBuffer mismatches;  // single-int on-device mismatch counter
  if (mismatches.allocate(sizeof(int)) != hipSuccess) {
    return false;
  }
  if (hipMemset(mismatches.get(), 0, sizeof(int)) != hipSuccess) {
    return false;
  }

  MismArgs mism_args{static_cast<const int*>(d_in), n, mismatches.as<int>()};
  std::string launch_err;
  if (hip_kernel_launch("k_count_mismatch_indices", grid_for(n),
                        dim3(kBlockSize), 0, 0, &mism_args, sizeof(mism_args),
                        &launch_err) != hipSuccess) {
    return false;
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    return false;
  }

  int host_count = -1;
  if (hipMemcpy(&host_count, mismatches.get(), sizeof(int),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return false;
  }
  return host_count == 0;
}

// Child (importer): open the parent's IPC handle and kernel-verify the pattern
// through the mapping, then _exit with a precise code (never returns).
void run_ipc_importer_child(IpcShared* sh, int n, int wait_ms) {
  auto child_exit = [&](int code) {
    sh->child_ok = (code == 0) ? 1 : 0;
    publish_flag(&sh->child_done);
    _exit(code);
  };

  if (hipSetDevice(0) != hipSuccess) {
    child_exit(kSetDeviceFailed);
  }
  if (!wait_flag(&sh->handle_ready, wait_ms)) {
    child_exit(kHandleWaitTimeout);
  }

  void* d_in = nullptr;
  if (hipIpcOpenMemHandle(&d_in, sh->handle, hipIpcMemLazyEnablePeerAccess) !=
      hipSuccess) {
    child_exit(kOpenHandleFailed);
  }

  const bool verified = ipc_kernel_verify_pattern(d_in, n);
  (void)hipIpcCloseMemHandle(d_in);
  child_exit(verified ? 0 : kVerifyFailed);
}

// Parent (exporter): stage the pattern, export the IPC handle (timed), hand it
// to the importer and bounded-wait for its verified result. Returns "" on pass.
std::string run_ipc_exporter_parent(IpcShared* sh, DeviceBuffer& buf,
                                    Stream& stream, int n, size_t bytes,
                                    int wait_ms, PerfSink& perf_sink) {
  // Phase 1: bring up device, buffer and stream.
  if (hipSetDevice(0) != hipSuccess) {
    return "parent hipSetDevice(0) failed";
  }
  if (buf.allocate(bytes) != hipSuccess) {
    return "parent hipMalloc failed";
  }
  if (stream.create() != hipSuccess) {
    return "parent hipStreamCreate failed";
  }

  // Phase 2: stage the index pattern into the buffer with a kernel write.
  FillArgs fill_args{buf.as<int>(), n};
  std::string launch_err;
  if (hip_kernel_launch("k_fill_indices", grid_for(n), dim3(kBlockSize), 0,
                        stream.get(), &fill_args, sizeof(fill_args),
                        &launch_err) != hipSuccess) {
    return "parent kernel write failed: " + launch_err;
  }
  if (hipStreamSynchronize(stream.get()) != hipSuccess) {
    return "parent hipStreamSynchronize failed";
  }

  // Phase 3: export the IPC handle into shared memory (timed under perf mode).
  PerfLoop loop;
  hipError_t handle_err = hipSuccess;
  for (auto pass : loop) {
    handle_err = hipIpcGetMemHandle(&sh->handle, buf.get());
    if (handle_err != hipSuccess) {
      break;
    }
  }
  if (handle_err != hipSuccess) {
    return std::string("hipIpcGetMemHandle: ") + hipGetErrorString(handle_err);
  }
  loop.emit_latency(perf_sink, "ipc.get_handle");

  // Phase 4: signal the handle is ready (release), then bounded-wait for the
  // importer and read back its verified result.
  publish_flag(&sh->handle_ready);
  if (!wait_flag(&sh->child_done, wait_ms)) {
    return "importer did not finish within " + std::to_string(wait_ms) + " ms";
  }
  if (sh->child_ok != 1) {
    return "importer failed to verify the IPC pattern";
  }
  return "";
}

}  // namespace

// Forked parent exports an IPC mem handle; child imports it and kernel-verifies the pattern.
ACCEPT_TEST(Ipc_Legacy_GetOpen_RoundTrip, "hip/ipc", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");
  const int n = 1024;
  const size_t bytes = sizeof(int) * static_cast<size_t>(n);
  const int kWaitMs = 15000;  // bounded; runner watchdog is the hard backstop

  // Shared region for sync flags + the IPC handle.
  void* shm = ::mmap(nullptr, sizeof(IpcShared), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (shm == MAP_FAILED)
    return fail(std::string("mmap(MAP_SHARED|MAP_ANONYMOUS): ") +
                std::strerror(errno));
  auto* sh = static_cast<IpcShared*>(shm);
  sh->handle_ready = 0;
  sh->child_done = 0;
  sh->child_ok = 0;

  // fork BEFORE any HIP call so both processes init HIP independently.
  const pid_t pid = ::fork();
  if (pid < 0) {
    (void)::munmap(shm, sizeof(IpcShared));
    return fail(std::string("fork: ") + std::strerror(errno));
  }

  if (pid == 0) {
    run_ipc_importer_child(sh, n, kWaitMs);
  }

  // Parent: buf/stream stay owned here so they free on scope exit, after the
  // bounded reap+unmap tail below.
  DeviceBuffer buf;
  Stream stream;
  PerfSink perf_sink;
  const std::string err =
      run_ipc_exporter_parent(sh, buf, stream, n, bytes, kWaitMs, perf_sink);

  // Bounded reap: if the importer is wedged, reap_child SIGKILLs rather than
  // blocking forever on waitpid — preserves the no-hang guarantee.
  int wstatus = 0;
  reap_child(pid, &wstatus);
  (void)::munmap(shm, sizeof(IpcShared));

  if (!err.empty())
    return fail(err + " (child exit status " + std::to_string(wstatus) + ")");

  return pass_with_perf(std::move(perf_sink));
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
