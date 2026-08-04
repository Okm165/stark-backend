#pragma once

#include <algorithm>
#include <cuda_runtime.h>
#ifdef CUDA_DEBUG
#include <cstdio>
#endif

// ROCm/HIP #2212: shared-memory inter-warp reductions miscompile at block
// sizes > 256 on gfx1100 (RDNA3).  Cap the default to 256 on AMD.
#if defined(__HIPCC__)
static const size_t MAX_THREADS = 256;
#else
static const size_t MAX_THREADS = 1024;
#endif
static const size_t WARP_SIZE = 32;

inline size_t div_ceil(size_t a, size_t b) { return (a + b - 1) / b; }

inline std::pair<dim3, dim3> kernel_launch_params(
    size_t count,
    size_t threads_per_block = MAX_THREADS
) {
    if (count == 0) {
        return std::make_pair(dim3(0, 1, 1), dim3(1, 1, 1));
    }
    size_t block = std::min(count, threads_per_block);
    // Round block size up to a WARP_SIZE multiple: on AMD, __shfl_down from
    // inactive lanes returns stale VGPR data (not 0 as on NVIDIA), corrupting
    // any warp-level reduction in a partial warp.
    if (threads_per_block >= WARP_SIZE) {
        block = ((block + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    }
    size_t grid = div_ceil(count, block);
    return std::make_pair(dim3(grid, 1, 1), dim3(block, 1, 1));
}

inline std::pair<dim3, dim3> kernel_launch_2d_params(size_t x, size_t y) {
    if (x == 0 || y == 0) {
        return std::make_pair(dim3(0, 1, 1), dim3(1, 1, 1));
    }
    dim3 block = dim3(std::min(x, WARP_SIZE), std::min(y, WARP_SIZE));
    dim3 grid = dim3(div_ceil(x, block.x), div_ceil(y, block.y));
    return std::make_pair(grid, block);
}

#define CUDA_OK(expr) do {                                  \
    cudaError_t err = expr;                                 \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err));   \
    }                                                       \
} while(0)

#ifdef CUDA_DEBUG
    inline int cuda_check_kernel(const char* kernel_name) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "[ERROR] Kernel '%s' failed: %s\n",
                    kernel_name, cudaGetErrorString(err));
        }
        return err;
    }
#   define CHECK_KERNEL() cuda_check_kernel(__func__)
#else
#   define CHECK_KERNEL() cudaGetLastError()
#endif
