// cuda2hip.hpp — CUDA → HIP compatibility layer for OpenVM / stark-backend
//
// Force-included (via `-include`) before every .cu file when compiling with
// hipcc on AMD GPUs.  Maps CUDA Runtime API names, types, and constants to
// their HIP equivalents so that .cu source files can be compiled unmodified.
//
// Based on supranational/sppark (Apache 2.0) with extensions for the full API
// surface used in stark-backend (streams, events, VPMM, device attributes).

#ifdef __HIPCC__
#pragma once
#pragma clang diagnostic ignored "-Wdeprecated-pragma"

// ── Step 1: Include HIP runtime ─────────────────────────────────────
// Must come first so all hip* symbols are available for aliasing below.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
// Include hipCUB early so its bf16/fp16 __shfl_*_sync overloads are
// defined before our warp-function macros override those names.
#if __has_include(<hipcub/hipcub.hpp>)
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

// HIP defines __forceinline__ as `inline __attribute__((always_inline))`.
// sppark headers do `#define inline __host__ __device__ __forceinline__`.
// When __forceinline__ expands and re-introduces `inline`, the preprocessor
// stops recursion but produces an unparseable token sequence.
// Fix: use `__inline__` (Clang keyword not captured by the `inline` macro).
#ifdef __forceinline__
#undef __forceinline__
#define __forceinline__ __inline__ __attribute__((always_inline))
#endif

// ── Step 2: Runtime API — function aliases ──────────────────────────
// Using `static const auto` for non-overloaded functions and `#define`
// for overloaded ones or constants.

// Device management
static const auto cudaGetDevice           = hipGetDevice;
static const auto cudaSetDevice           = hipSetDevice;
static const auto cudaDeviceGetAttribute  = hipDeviceGetAttribute;

// Memory copy
using cudaMemcpyKind = hipMemcpyKind;
static const auto cudaMemcpy      = hipMemcpy;
static const auto cudaMemcpyAsync = hipMemcpyAsync;
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Symbol copy (overloaded in HIP — must use #define)
#define cudaMemcpyToSymbol      hipMemcpyToSymbol
#define cudaMemcpyToSymbolAsync hipMemcpyToSymbolAsync

// Error handling
using cudaError_t = hipError_t;
static const auto cudaGetLastError    = hipGetLastError;
static const auto cudaGetErrorString  = hipGetErrorString;
static const auto cudaGetErrorName    = hipGetErrorName;
#define cudaSuccess               hipSuccess
#define cudaErrorNotReady         hipErrorNotReady
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorInvalidValue         hipErrorInvalidValue
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration

// Streams
using cudaStream_t = hipStream_t;
static const auto cudaStreamCreateWithFlags = hipStreamCreateWithFlags;
static const auto cudaStreamDestroy         = hipStreamDestroy;
static const auto cudaStreamSynchronize     = hipStreamSynchronize;
static const auto cudaStreamWaitEvent       = hipStreamWaitEvent;

// Device attribute constants
#define cudaDevAttrMultiProcessorCount \
    hipDeviceAttributeMultiprocessorCount

// Kernel launch attributes — hipFuncSetAttribute takes `const void*` but
// CUDA implicitly converts a kernel function to `const void*`. Provide a
// template wrapper so callers can pass a bare kernel name.
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
    hipFuncAttributeMaxDynamicSharedMemorySize

template<typename F>
static inline hipError_t cudaFuncSetAttribute(
    F func, hipFuncAttribute attr, int value
) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

// ── Step 3: Device-code macros ──────────────────────────────────────
//
// We intentionally do NOT define __CUDA_ARCH__.  Code that guards PTX inline
// assembly with `#ifdef __CUDA_ARCH__` must not compile that asm on HIP.
// Non-asm fallback paths (typically in `#else` blocks) use portable C++ and
// work correctly under hipcc.  Code that needs a device-compilation guard
// should check `__HIP_DEVICE_COMPILE__` (or `__HIPCC__` for file-level).

// ── Step 4: Warp-level primitives ───────────────────────────────────
//
// HIP's __shfl_*_sync require a 64-bit mask, while CUDA code universally
// passes 32-bit `unsigned` masks (e.g. 0xFFFFFFFF).  We intercept the calls
// via macros that widen the mask to `unsigned long long` before forwarding
// to HIP's native template.  The value semantics are preserved: a 32-bit
// full-mask 0xFFFFFFFF becomes 0x00000000FFFFFFFF which is correct for
// wave32 GPUs (RDNA / gfx10+ / gfx11+).

// Save HIP's originals under prefixed names
template<typename MaskT, typename T>
__device__ inline T __hip_orig_shfl_xor_sync(MaskT mask, T var, int laneMask, int width = warpSize) {
    return ::__shfl_xor_sync(mask, var, laneMask, width);
}
template<typename MaskT, typename T>
__device__ inline T __hip_orig_shfl_down_sync(MaskT mask, T var, unsigned int delta, int width = warpSize) {
    return ::__shfl_down_sync(mask, var, delta, width);
}

// Macro wrappers that widen 32-bit masks to 64-bit
#undef __shfl_xor_sync
#define __shfl_xor_sync(mask, var, laneMask, ...) \
    __hip_orig_shfl_xor_sync(static_cast<unsigned long long>(mask), (var), (laneMask) __VA_OPT__(,) __VA_ARGS__)

#undef __shfl_down_sync
#define __shfl_down_sync(mask, var, delta, ...) \
    __hip_orig_shfl_down_sync(static_cast<unsigned long long>(mask), (var), (delta) __VA_OPT__(,) __VA_ARGS__)

// __syncwarp also requires a 64-bit mask in HIP.  We provide a
// function overload set that accepts either no args or a 32-bit mask.
// A compiler fence is added after the barrier to prevent LLVM from
// sinking LDS loads past the synchronization point (same rationale
// as the __syncthreads() hardening in Step 6).
__device__ inline void __hip_syncwarp_impl() { ::__syncwarp(); }
__device__ inline void __hip_syncwarp_impl(unsigned int mask) {
    ::__syncwarp(static_cast<unsigned long long>(mask));
}
#undef __syncwarp
#define __syncwarp(...) do { __hip_syncwarp_impl(__VA_ARGS__); asm volatile("" ::: "memory"); } while(0)

// ── Step 5: CUDA Driver API mappings (for VPMM) ────────────────────
//
// vpmm_shim.cu uses the CUDA Driver API (cu* prefix) for virtual memory
// management.  HIP provides equivalent functions under the hip* prefix
// with compatible semantics.

typedef hipDevice_t   CUdevice;
typedef hipError_t    CUresult;
typedef void*         CUdeviceptr;
typedef hipMemGenericAllocationHandle_t CUmemGenericAllocationHandle;
typedef hipMemAllocationProp            CUmemAllocationProp;
typedef hipMemAccessDesc                CUmemAccessDesc;

#define CUDA_SUCCESS              hipSuccess
#define CUDA_ERROR_NOT_SUPPORTED  hipErrorNotSupported
#define CUDA_ERROR_INVALID_VALUE  hipErrorInvalidValue

#define CU_MEM_ALLOCATION_TYPE_PINNED        hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE          hipMemLocationTypeDevice
#define CU_MEM_HANDLE_TYPE_NONE              hipMemHandleTypeNone
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM     hipMemAllocationGranularityMinimum
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE   hipMemAccessFlagsProtReadWrite
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED \
    hipDeviceAttributeVirtualMemoryManagementSupported

#define cuDeviceGet          hipDeviceGet
#define cuDeviceGetAttribute hipDeviceGetAttribute

#define cuMemCreate(h, sz, prop, fl) \
    hipMemCreate((h), (sz), (prop), (fl))

#define cuMemGetAllocationGranularity(out, prop, gran) \
    hipMemGetAllocationGranularity((out), (prop), (gran))

// CUdeviceptr is `void*` in our typedef but CUDA uses `unsigned long long`.
// The shim functions pass CUdeviceptr by value, so we cast to `void*` for HIP.
#define cuMemAddressReserve(va, sz, al, addr, fl) \
    hipMemAddressReserve(reinterpret_cast<void**>(va), (sz), (al), \
                         reinterpret_cast<void*>(addr), (fl))

#define cuMemAddressFree(va, sz) \
    hipMemAddressFree(reinterpret_cast<void*>(va), (sz))

#define cuMemMap(va, sz, off, h, fl) \
    hipMemMap(reinterpret_cast<void*>(va), (sz), (off), (h), (fl))

#define cuMemUnmap(va, sz) \
    hipMemUnmap(reinterpret_cast<void*>(va), (sz))

#define cuMemSetAccess(va, sz, acc, cnt) \
    hipMemSetAccess(reinterpret_cast<void*>(va), (sz), (acc), (cnt))

#define cuMemRelease hipMemRelease

// ── Step 6: __syncthreads() / __syncwarp() compiler fences ──────────
//
// AMD gfx1100 (RDNA3): LLVM can sink ds_load past S_BARRIER, reading
// stale LDS values (LLVM #181708).  Compiler fences prevent this.
// Disable with AMDGPU_BARRIER_FENCE=0 for debugging.
__device__ __attribute__((convergent)) inline
void __hip_syncthreads_fenced() {
    __syncthreads();
}
#define __syncthreads() do { __hip_syncthreads_fenced(); asm volatile("" ::: "memory"); } while(0)

#endif // __HIPCC__
