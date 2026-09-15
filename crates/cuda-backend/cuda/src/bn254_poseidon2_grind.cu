/// BN254 sponge PoW grinding kernel — fully-inlined for maximum GPU throughput.
///
/// Architecture:
///   - All field arithmetic + Poseidon2 permutation is __forceinline__ within the
///     `bn254_grind_inline` namespace. This eliminates function call overhead and
///     scratch memory usage — the entire hot path stays in registers.
///   - On NVIDIA: nvcc lowers __uint128_t to native PTX (mad.hi.cc carry chains).
///   - On AMD: hipcc emulates __uint128_t via 32-bit ops; __forceinline__ ensures
///     the register allocator avoids scratch memory spills.
///
/// Translation unit isolation (device-link group: bn254):
///   bn254_constants.cu          - Round constant storage + init launchers
///   bn254_poseidon2_row_hash.cu - Merkle row-hash kernels (uses bn254_b32::*)
///   bn254_poseidon2_grind.cu    - (this) Grinding kernel + FFI

#include "fp.h"
#include "launcher.cuh"
#include "poseidon2_bn254_noinline.cuh" // for Bn254Fr, BN254_P, BN254_MU, BN254_R2, BABYBEAR_PRIME
#include <cstdint>

// ---------------------------------------------------------------------------
// Round constant device memory (filled by _init_bn254_poseidon2_rc)
// ---------------------------------------------------------------------------

/// External initial round constants: 4 rounds × 3 elements
extern __device__ __constant__ Bn254Fr g_initial_rc[4][3];

/// Internal (partial) round constants: 56 rounds × 1 element (for state[0] only)
extern __device__ __constant__ Bn254Fr g_partial_rc[56];

/// External terminal round constants: 4 rounds × 3 elements
extern __device__ __constant__ Bn254Fr g_terminal_rc[4][3];

struct Bn254PoseidonPermShared {
    Bn254Fr *initial_rc;
    Bn254Fr *partial_rc;
    Bn254Fr *terminal_rc;
};

// make sure to __syncthreads() before reading
static __device__ Bn254PoseidonPermShared load_shared() {
    __shared__ uint64_t buf[(4 * 3 + 56 + 4 * 3) * 4];
    for (int i = threadIdx.x; i < 12 * 4; i += blockDim.x) {
        buf[i] = ((uint64_t *)g_initial_rc)[i];
    }

    for (int i = threadIdx.x; i < 56 * 4; i += blockDim.x) {
        buf[i + 12 * 4] = ((uint64_t *)g_partial_rc)[i];
    }

    for (int i = threadIdx.x; i < 12 * 4; i += blockDim.x) {
        buf[i + 12 * 4 + 56 * 4] = ((uint64_t *)g_terminal_rc)[i];
    }
    auto ptr = (Bn254Fr *)buf;

    return {ptr, ptr + 12, ptr + 12 + 56};
}

// ---------------------------------------------------------------------------
// BN254 sponge state for GPU grinding
//
// Matches MultiFieldTranscript<BabyBear, Bn254Scalar, Perm, WIDTH=3, RATE=2>:
//   num_obs_per_word = SF::bits() / CF::bits() = 254/31 = 8
//   num_samples_per_word = 5  (base-p decomposition, ≥100 bits bias slack)
//
// The sponge uses overwrite-mode duplex with absorb_idx/sample_idx tracking.
// Rust DeviceBn254SpongeState must have identical layout (verified by size assert).
// ---------------------------------------------------------------------------

static const uint32_t BN254_NUM_OBS_PER_WORD = 8;
static const uint32_t BN254_SPONGE_RATE = 2;

struct DeviceBn254SpongeState {
    Bn254Fr sponge_state[3];  // 96 bytes
    uint32_t absorb_idx;      //  4 bytes
    uint32_t sample_idx;      //  4 bytes
    uint32_t observe_buf[8];  // 32 bytes
    uint32_t observe_buf_len; //  4 bytes
    // total = 140 + 4 padding = 144 bytes (aligned to 8)
    // Note: sample_buf is not needed on device — observe() clears it before grinding.
};

static_assert(
    sizeof(DeviceBn254SpongeState) == 144,
    "DeviceBn254SpongeState size mismatch with Rust"
);

// ===========================================================================
// bn254_grind_inline: __forceinline__ field arithmetic + permutation for the
// grinding kernel. Eliminates all function call overhead and scratch memory
// usage, keeping all computation in registers for maximum GPU throughput.
//
// Architecture: Uses __uint128_t directly (same algorithm as bn254_noinline).
// On AMD/HIP, hipcc emulates __uint128_t via 32-bit ops; the forceinline
// ensures the compiler can allocate registers across the full permutation
// without spilling to scratch memory (which __noinline__ forces).
// On NVIDIA, nvcc lowers __uint128_t to efficient PTX.
// ===========================================================================

namespace bn254_grind_inline {

static __device__ __forceinline__ uint64_t
add256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        __uint128_t t = (__uint128_t)a[i] + b[i] + carry;
        r[i] = (uint64_t)t;
        carry = (uint64_t)(t >> 64);
    }
    return carry;
}

static __device__ __forceinline__ uint64_t
sub256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        __uint128_t t = (__uint128_t)a[i] - b[i] - borrow;
        r[i] = (uint64_t)t;
        borrow = (t >> 127) ? 1 : 0;
    }
    return borrow;
}

static __device__ __forceinline__ uint64_t
mul_small(uint64_t high4[4], const uint64_t lhs[4], uint64_t rhs) {
    __uint128_t acc = (__uint128_t)lhs[0] * rhs;
    uint64_t low = (uint64_t)acc;
    acc >>= 64;
#pragma unroll
    for (int i = 1; i < 4; i++) {
        acc += (__uint128_t)lhs[i] * rhs;
        high4[i - 1] = (uint64_t)acc;
        acc >>= 64;
    }
    high4[3] = (uint64_t)acc;
    return low;
}

static __device__ __forceinline__ uint64_t
mul_small_and_acc(uint64_t high4[4], const uint64_t lhs[4], uint64_t rhs, const uint64_t add[4]) {
    __uint128_t acc = (__uint128_t)lhs[0] * rhs + add[0];
    uint64_t low = (uint64_t)acc;
    acc >>= 64;
#pragma unroll
    for (int i = 1; i < 4; i++) {
        acc += (__uint128_t)lhs[i] * rhs + add[i];
        high4[i - 1] = (uint64_t)acc;
        acc >>= 64;
    }
    high4[3] = (uint64_t)acc;
    return low;
}

static __device__ __forceinline__ void imr(uint64_t r[4], uint64_t acc0, const uint64_t acc[4]) {
    uint64_t t = acc0 * BN254_MU;
    uint64_t u[4];
    mul_small(u, BN254_P, t);
    uint64_t sub[4];
    uint64_t borrow = sub256(sub, acc, u);
    if (borrow) {
        add256(r, sub, BN254_P);
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++)
            r[i] = sub[i];
    }
}

static __device__ __forceinline__ void monty_mul(
    uint64_t r[4],
    const uint64_t lhs[4],
    const uint64_t rhs[4]
) {
    uint64_t acc0, acc[4], tmp[4];
    acc0 = mul_small(acc, lhs, rhs[0]);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[1], tmp);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[2], tmp);
    imr(tmp, acc0, acc);
    acc0 = mul_small_and_acc(acc, lhs, rhs[3], tmp);
    imr(r, acc0, acc);
}

static __device__ __forceinline__ Bn254Fr bn254_add(Bn254Fr a, Bn254Fr b) {
    Bn254Fr r;
    uint64_t sum[4];
    uint64_t overflow = add256(sum, a.limbs, b.limbs);
    uint64_t sub[4];
    uint64_t borrow = sub256(sub, sum, BN254_P);
    if (overflow || !borrow) {
#pragma unroll
        for (int i = 0; i < 4; i++)
            r.limbs[i] = sub[i];
    } else {
#pragma unroll
        for (int i = 0; i < 4; i++)
            r.limbs[i] = sum[i];
    }
    return r;
}

static __device__ __forceinline__ Bn254Fr bn254_mul(Bn254Fr a, Bn254Fr b) {
    Bn254Fr r;
    monty_mul(r.limbs, a.limbs, b.limbs);
    return r;
}

static __device__ __forceinline__ Bn254Fr bn254_sbox(Bn254Fr x) {
    Bn254Fr x2 = bn254_mul(x, x);
    Bn254Fr x4 = bn254_mul(x2, x2);
    return bn254_mul(x4, x);
}

static __device__ __forceinline__ Bn254Fr bn254_double(Bn254Fr a) { return bn254_add(a, a); }

static __device__ __forceinline__ Bn254Fr bn254_from_canonical(const uint64_t canonical[4]) {
    Bn254Fr r;
    monty_mul(r.limbs, BN254_R2, canonical);
    return r;
}

static __device__ __forceinline__ void bn254_to_canonical(uint64_t canonical[4], Bn254Fr x) {
    const uint64_t one[4] = {1, 0, 0, 0};
    monty_mul(canonical, x.limbs, one);
}

static __device__ __forceinline__ Bn254Fr bn254_pack_base_2_31(const uint32_t *bb, int count) {
    uint64_t canonical[4] = {0, 0, 0, 0};
    for (int i = 0; i < count; i++) {
        int bit_pos = i * 31;
        int limb = bit_pos >> 6;
        int shift = bit_pos & 63;
        canonical[limb] |= (uint64_t)(bb[i]) << shift;
        if (shift > 33 && limb < 3) {
            canonical[limb + 1] |= (uint64_t)(bb[i]) >> (64 - shift);
        }
    }
    return bn254_from_canonical(canonical);
}

static __device__ __forceinline__ uint32_t u256_mod_u32(const uint64_t x[4], uint32_t d) {
    uint64_t rem = 0;
    for (int i = 3; i >= 0; i--) {
        rem = ((rem << 32) | (x[i] >> 32)) % d;
        rem = ((rem << 32) | (x[i] & 0xFFFFFFFFULL)) % d;
    }
    return (uint32_t)rem;
}

template <int WIDTH> static __device__ __forceinline__ void bn254_mds_external(Bn254Fr s[WIDTH]) {
    Bn254Fr sum = s[0];
#pragma unroll
    for (int i = 1; i < WIDTH; i++)
        sum = bn254_add(sum, s[i]);
#pragma unroll
    for (int i = 0; i < WIDTH; i++)
        s[i] = bn254_add(s[i], sum);
}

template <int WIDTH> static __device__ __forceinline__ void bn254_mds_internal(Bn254Fr s[WIDTH]) {
    Bn254Fr sum = s[0];
#pragma unroll
    for (int i = 1; i < WIDTH; i++)
        sum = bn254_add(sum, s[i]);
#pragma unroll
    for (int i = 0; i < WIDTH - 1; i++)
        s[i] = bn254_add(s[i], sum);
    s[WIDTH - 1] = bn254_add(bn254_double(s[WIDTH - 1]), sum);
}

static __device__ __forceinline__ void bn254_poseidon2_permute_grind(
    Bn254Fr state[3],
    const Bn254Fr *initial_rc,
    const Bn254Fr *partial_rc,
    const Bn254Fr *terminal_rc
) {
    bn254_mds_external<3>(state);
#pragma unroll
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 3; i++) {
            state[i] = bn254_add(state[i], initial_rc[r * 3 + i]);
            state[i] = bn254_sbox(state[i]);
        }
        bn254_mds_external<3>(state);
    }
#pragma unroll 1
    for (int r = 0; r < 56; r++) {
        state[0] = bn254_add(state[0], partial_rc[r]);
        state[0] = bn254_sbox(state[0]);
        bn254_mds_internal<3>(state);
    }
#pragma unroll
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 3; i++) {
            state[i] = bn254_add(state[i], terminal_rc[r * 3 + i]);
            state[i] = bn254_sbox(state[i]);
        }
        bn254_mds_external<3>(state);
    }
}

} // namespace bn254_grind_inline

// ===========================================================================
// Grinding-specific sponge: uses bn254_grind_inline for all permutations.
// ===========================================================================

static __device__ __forceinline__ void bn254_sponge_absorb_grind(
    DeviceBn254SpongeState &s,
    Bn254Fr value,
    Bn254PoseidonPermShared shared_state
) {
    s.sponge_state[s.absorb_idx] = value;
    s.absorb_idx++;
    if (s.absorb_idx == BN254_SPONGE_RATE) {
        bn254_grind_inline::bn254_poseidon2_permute_grind(
            s.sponge_state,
            shared_state.initial_rc,
            shared_state.partial_rc,
            shared_state.terminal_rc
        );
        s.absorb_idx = 0;
        s.sample_idx = BN254_SPONGE_RATE;
    }
}

static __device__ __forceinline__ Bn254Fr
bn254_sponge_squeeze_grind(DeviceBn254SpongeState &s, Bn254PoseidonPermShared shared_state) {
    if (s.absorb_idx != 0 || s.sample_idx == 0) {
        bn254_grind_inline::bn254_poseidon2_permute_grind(
            s.sponge_state,
            shared_state.initial_rc,
            shared_state.partial_rc,
            shared_state.terminal_rc
        );
        s.absorb_idx = 0;
        s.sample_idx = BN254_SPONGE_RATE;
    }
    s.sample_idx--;
    return s.sponge_state[s.sample_idx];
}

static __device__ __forceinline__ void bn254_transcript_flush_observe_grind(
    DeviceBn254SpongeState &s,
    Bn254PoseidonPermShared shared_state
) {
    if (s.observe_buf_len > 0) {
        Bn254Fr packed = bn254_grind_inline::bn254_pack_base_2_31(s.observe_buf, s.observe_buf_len);
        bn254_sponge_absorb_grind(s, packed, shared_state);
        s.observe_buf_len = 0;
    }
}

static __device__ __forceinline__ void bn254_transcript_observe_grind(
    DeviceBn254SpongeState &s,
    uint32_t value,
    Bn254PoseidonPermShared shared_state
) {
    s.observe_buf[s.observe_buf_len++] = value;
    if (s.observe_buf_len == BN254_NUM_OBS_PER_WORD) {
        Bn254Fr packed =
            bn254_grind_inline::bn254_pack_base_2_31(s.observe_buf, BN254_NUM_OBS_PER_WORD);
        bn254_sponge_absorb_grind(s, packed, shared_state);
        s.observe_buf_len = 0;
    }
}

static __device__ __forceinline__ uint32_t
bn254_transcript_sample_grind(DeviceBn254SpongeState &s, Bn254PoseidonPermShared shared_state) {
    bn254_transcript_flush_observe_grind(s, shared_state);
    Bn254Fr squeezed = bn254_sponge_squeeze_grind(s, shared_state);
    uint64_t canonical[4];
    bn254_grind_inline::bn254_to_canonical(canonical, squeezed);
    return bn254_grind_inline::u256_mod_u32(canonical, (uint32_t)BABYBEAR_PRIME);
}

/// Returns true if check_witness(bits, witness) passes — using inline permutation.
static __device__ __forceinline__ bool bn254_sponge_check_witness_grind(
    DeviceBn254SpongeState &s,
    uint32_t bits,
    uint32_t witness,
    Bn254PoseidonPermShared shared_state
) {
    bn254_transcript_observe_grind(s, witness, shared_state);
    uint32_t sample = bn254_transcript_sample_grind(s, shared_state);
    return (sample & ((1u << bits) - 1)) == 0;
}

/// Grinding kernel: each thread checks ONE candidate w = min_witness + tid.
/// No inner loop — the Rust-side wrapper batches kernel launches to cover the
/// full witness range.  This avoids reliance on volatile cross-CU coherence
/// for early exit (which is unreliable on AMD RDNA).
__global__ void bn254_grind_kernel(
    const DeviceBn254SpongeState *init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t *result
) {
    uint32_t w = min_witness + blockIdx.x * blockDim.x + threadIdx.x;
    if (w > max_witness || *result != UINT32_MAX)
        return;

    __shared__ DeviceBn254SpongeState s_local_state[1];
    for (int i = threadIdx.x; i < sizeof(DeviceBn254SpongeState) / sizeof(uint32_t);
         i += blockDim.x) {
        ((uint32_t *)s_local_state)[i] = ((uint32_t *)init_state)[i];
    }

    Bn254PoseidonPermShared shared_state = load_shared();
    __syncthreads();

    DeviceBn254SpongeState local_state = s_local_state[0];
    if (bn254_sponge_check_witness_grind(local_state, bits, w, shared_state)) {
        atomicCAS(result, UINT32_MAX, w);
    }
}

extern "C" int _bn254_sponge_grind(
    const DeviceBn254SpongeState *init_state,
    uint32_t bits,
    uint32_t min_witness,
    uint32_t max_witness,
    uint32_t *result,
    cudaStream_t stream
) {
    if (bits >= 32 || (uint64_t{1} << bits) >= Fp::P) {
        return cudaErrorInvalidValue;
    }
    auto const [grid, block] = kernel_launch_params(1 << bits);

    bn254_grind_kernel<<<grid, block, 0, stream>>>(
        init_state, bits, min_witness, max_witness, result
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return (int)err;

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        return (int)err;

    return CHECK_KERNEL();
}
