// HIP acceptance tests — Kernel Launch group (RCCL launch+error path).
//
// Built only when ACCEPT_WITH_HIP=ON. The TU compiles as plain C++ (no HIP
// toolchain): HIP APIs resolve through the dlopen wrapper (HipApi.hpp). It covers
// the launch floor RCCL relies on before every collective:
//   * host-symbol launches -- hipLaunchKernel / hipGetFuncBySymbol /
//     hipLaunchKernelExC / hipFuncSetAttribute on a STATICALLY linked, runtime
//     -registered kernel (StaticKernels.hpp -> prebuilt kernels_static.o). These
//     APIs map a host function pointer to a device kernel, which a runtime-loaded
//     module cannot provide, so the kernels are statically linked.
//   * the module launch path -- hipModuleLaunchKernel on a function from a module
//     compiled at runtime by hiprtc (HipModule.hpp), needing no device compiler
//     at build time and no GPU arch baked in.
//
// Conventions: all tests are deterministic functional checks -> MUST_PASS
// (REGRESSION is reserved for crash/hang/timeout). The module-path legs skip()
// when hiprtc is unavailable on the runtime.

#ifndef ACCEPT_HAVE_HIP
// No HIP — nothing to register.
#else

#include "hip/utils/HipApi.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

#include "common/TestRegistry.hpp"
#include "hip/kernel_launch/StaticKernels.hpp"
#include "hip/utils/HipHelpers.hpp"
#include "hip/utils/HipModule.hpp"
#include "hip/utils/HipTest.hpp"

namespace acceptance {

// Launch a kernel with dynamic shared memory (hipLaunchKernel on a static kernel)
// and confirm hipGetLastError is clean.
ACCEPT_TEST(Kernel_Launch_DynShared_LastError, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int n = 1024;
  const unsigned block = 256;
  const unsigned grid = (n + block - 1) / block;
  const size_t dyn = block * sizeof(int);

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));

  int n_arg = n;
  void* args[] = {&d_buf, &n_arg};
  HIP_CHECK(hipLaunchKernel(sk_dyn_shmem_fill_ptr(), dim3(grid), dim3(block),
                            args, dyn, s));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("dyn-shmem launch result wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));
  return pass();
}

// Raise the kernel dynamic shared-memory limit via hipFuncSetAttribute on a
// static kernel symbol, launch, and verify output.
ACCEPT_TEST(Launch_DynShared_FuncAttr, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));

  int optin = 0;
  HIP_CHECK(hipDeviceGetAttribute(&optin, hipDeviceAttributeSharedMemPerBlockOptin,
                                  dev));

  hipFuncAttributes fa{};
  HIP_CHECK(hipFuncGetAttributes(&fa, sk_dyn_shmem_fill_ptr()));

  const int n = 1024;
  const unsigned block = 256;
  const unsigned grid = (n + block - 1) / block;

  // Choose a dyn-shmem size to opt into: at least what the kernel needs, bounded
  // to a sane ceiling, and never above the device opt-in maximum.
  const size_t need = block * sizeof(int);
  size_t dyn = (optin > 0) ? static_cast<size_t>(optin) : need;
  dyn = std::min<size_t>(dyn, 32768);
  dyn = std::max<size_t>(dyn, need);

  HIP_CHECK(hipFuncSetAttribute(sk_dyn_shmem_fill_ptr(),
                                hipFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(dyn)));

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));

  int n_arg = n;
  void* args[] = {&d_buf, &n_arg};
  HIP_CHECK(hipLaunchKernel(sk_dyn_shmem_fill_ptr(), dim3(grid), dim3(block),
                            args, dyn, s));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("opt-in dyn-shmem launch wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v) +
                " (dyn=" + std::to_string(dyn) + ")");
  return pass();
}

// Smoke-test hipLaunchKernel on a static kernel (Part A) and hipModuleLaunchKernel
// on a hiprtc-compiled module function (Part B).
ACCEPT_TEST(Kernel_LaunchVariants_Smoke, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  if (!hip_kernels_available()) return skip("requires hiprtc");

  const int n = 256;
  const dim3 grid(1), block(256);

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));

  // Poison the buffer so a no-op launch leaves detectable stale data (parity
  // with Part B, which memsets before its module launch).
  HIP_CHECK(hipMemset(d_buf, 0xff, sizeof(int) * n));

  // Part A: hipLaunchKernel with a void* args[] array on the static kernel.
  int n_arg = n;
  void* argsA[] = {&d_buf, &n_arg};
  HIP_CHECK(hipLaunchKernel(sk_fill_indices_ptr(), grid, block, argsA, 0, s));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  if (!hip_verify_indices(d_buf, n, &bad_i, &bad_v)) {
    (void)hipStreamDestroy(s);
    (void)hipFree(d_buf);
    return fail("hipLaunchKernel result wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));
  }

  // Part B: hipModuleLaunchKernel with packed args on a hiprtc module function.
  HIP_CHECK(hipMemset(d_buf, 0xff, sizeof(int) * n));
  FillArgs kargs{d_buf, n};
  std::string kerr;
  if (hip_kernel_launch("k_fill_indices", grid, block, 0, s, &kargs,
                        sizeof(kargs), &kerr) != hipSuccess) {
    (void)hipStreamDestroy(s);
    (void)hipFree(d_buf);
    return fail(kerr);
  }
  HIP_CHECK(hipStreamSynchronize(s));

  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("module launch result wrong at index " + std::to_string(bad_i) +
                ": got " + std::to_string(bad_v));
  return pass();
}

// Resolve a static device kernel with hipGetFuncBySymbol and launch it via
// hipModuleLaunchKernel.
ACCEPT_TEST(GetFuncBySymbol_Resolve, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int n = 256;
  const dim3 grid(1), block(256);

  hipFunction_t fn = nullptr;
  HIP_CHECK(hipGetFuncBySymbol(&fn, sk_fill_indices_ptr()));
  if (fn == nullptr)
    return fail("hipGetFuncBySymbol returned a null function");

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));
  HIP_CHECK(hipMemset(d_buf, 0xff, sizeof(int) * n));

  FillArgs kargs{d_buf, n};
  size_t ksize = sizeof(kargs);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kargs,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &ksize, HIP_LAUNCH_PARAM_END};
  hipError_t launch_err = hipModuleLaunchKernel(fn, grid.x, 1, 1, block.x, 1, 1,
                                                0, s, nullptr, config);
  if (launch_err != hipSuccess) {
    (void)hipStreamDestroy(s);
    (void)hipFree(d_buf);
    return fail(std::string("hipModuleLaunchKernel (resolved function): ") +
                hipGetErrorString(launch_err));
  }
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("resolved-function launch wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));
  return pass();
}

// Launch a single-block cooperative kernel through hipLaunchKernelExC with an
// attribute array on a static kernel symbol.
ACCEPT_TEST(LaunchKernelEx_ClusterAttrs, "hip/launch", Topology::Single,
            Kind::Functional, 30) {
  (void)ctx;
  const int n = 256;

  int* d_buf = nullptr;
  hipStream_t s = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, sizeof(int) * n));
  HIP_CHECK(hipStreamCreate(&s));

  hipLaunchAttribute attrs[1];
  attrs[0].id = hipLaunchAttributeCooperative;
  attrs[0].val.cooperative = 1;

  hipLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(256);
  config.dynamicSmemBytes = 0;
  config.stream = s;
  config.attrs = attrs;
  config.numAttrs = 1;

  int n_arg = n;
  void* args[] = {&d_buf, &n_arg};
  HIP_CHECK(hipLaunchKernelExC(&config, sk_fill_indices_ptr(), args));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(s));

  int bad_i = -1, bad_v = -1;
  const bool ok = hip_verify_indices(d_buf, n, &bad_i, &bad_v);

  (void)hipStreamDestroy(s);
  (void)hipFree(d_buf);

  if (!ok)
    return fail("hipLaunchKernelExC result wrong at index " +
                std::to_string(bad_i) + ": got " + std::to_string(bad_v));
  return pass();
}

}  // namespace acceptance
#endif  // ACCEPT_HAVE_HIP
