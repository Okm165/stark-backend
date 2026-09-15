# openvm-cuda-backend

GPU-accelerated STARK prover backend for the SWIRL proof system. Supports both NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs through a unified codebase with a thin translation layer.

## Architecture

```
src/
├── cuda/
│   ├── stacked_reduction.rs   # FFI: sumcheck MLE round kernels
│   ├── whir.rs                # FFI: WHIR algebraic batch traces
│   └── ...
├── data_transporter.rs        # H2D matrix/tree transport
├── prove.rs                   # Top-level GPU prover (per-segment)
├── whir.rs                    # WHIR protocol GPU implementation
└── bn254_sponge.rs            # BN254 Poseidon2 sponge

cuda/
├── src/
│   ├── sumcheck.cu            # Sumcheck MLE round kernel (Horner accumulation)
│   ├── zerocheck.cu           # Zerocheck kernel
│   └── ...
└── supra/
    ├── ntt.cu                 # NTT forward/inverse (supranational-derived)
    └── ntt_bitrev.cu          # NTT bit-reversal permutation (COBRA padding)
```

## Key GPU Optimizations

### Vendor-Neutral (applied to both AMD + NVIDIA)

| Optimization | File | Impact |
|-------------|------|--------|
| Horner accumulation in sumcheck | `cuda/src/sumcheck.cu` | Replaces FpExt×FpExt mul with FpExt add |
| GPU_REGISTER_HEAVY macro | `cuda-common/include/launcher.cuh` | waves_per_eu(1,4) — fewer threads, more VGPRs |
| COBRA bank-conflict padding | `cuda/supra/ntt_bitrev.cu` | Z_STRIDE = Z_COUNT + 1, eliminates LDS bank conflicts |
| Active-thread masking in NTT | `cuda/supra/ntt.cu` | Prevents OOB when block padded to WARP_SIZE |
| Warp-deduplicated histogram | `primitives/histogram.cuh` | sm_70+ ballot → single atomicAdd per unique key |

### AMD-Specific (via HIP translation)

| Optimization | File | Impact |
|-------------|------|--------|
| `ds_bpermute` lane permute | `cuda-common/include/ff/mont32_t.hip` | Native RDNA instruction, no runtime overhead |
| Binary GCD inverse | `cuda-common/include/ff/mont_t.hip` | ~512 shift+add vs 254 modular multiplications |
| HIP launch bounds (256 max) | `cuda/supra/ntt_bitrev.cu`, `ntt.cu` | RDNA3 Wave32 optimal occupancy |
| VPMM release_free_pages | `cuda-common/src/memory_manager/` | Disabled on AMD (RDNA L1 coherence issue) |

## Building

```bash
# AMD (ROCm/HIP — auto-detected when ROCM_PATH is set):
cargo build --release -p openvm-cuda-backend

# NVIDIA (CUDA — auto-detected when nvcc is in PATH):
cargo build --release -p openvm-cuda-backend

# Check for warnings:
cargo clippy --release -p openvm-cuda-backend -p openvm-cuda-common
```

## Memory Constraints

Peak GPU memory usage:
- STARK segment proving: ~5.5 GiB
- Root proving + aggregation: ~6.5 GiB
- Halo2 outer proof: ~21 GiB (runs in-process AFTER STARK memory is freed)

**Rule**: Never run a STARK worker and Halo2 on the same GPU simultaneously.

## Formatting

CUDA/HIP files follow the project's `.clang-format` (LLVM-based, 100-col, 4-space indent):

```bash
clang-format -i cuda/src/sumcheck.cu cuda/supra/ntt_bitrev.cu cuda/supra/ntt.cu
```

Rust files use the project's `rustfmt.toml`:

```bash
cargo fmt -p openvm-cuda-backend -p openvm-cuda-common
```
