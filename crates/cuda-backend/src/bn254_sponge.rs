//! GPU-accelerated transcript for the BabyBear-BN254 Poseidon2 configuration.
//!
//! [`MultiFieldTranscriptGpu`] wraps the CPU `MultiFieldTranscript` and adds
//! host-to-device state synchronization for GPU proof-of-work grinding.

use std::{ffi::c_void, sync::Arc};

use openvm_cuda_common::{
    copy::cuda_memcpy_on, d_buffer::DeviceBuffer, error::MemCopyError, stream::GpuDeviceCtx,
};
use openvm_stark_backend::FiatShamirTranscript;
use openvm_stark_sdk::config::{
    baby_bear_bn254_poseidon2::{BabyBearBn254Poseidon2Config, Bn254Scalar, Transcript},
    bn254_poseidon2::default_bn254_poseidon2_width3,
};
use p3_baby_bear::BabyBear;
use p3_field::{PrimeCharacteristicRing, PrimeField32};

use crate::sponge::{validate_gpu_grind_bits, GpuFiatShamirTranscript, GrindError};

// ---------------------------------------------------------------------------
// Distributed grinding support
// ---------------------------------------------------------------------------

/// Trait for distributed PoW grinding helpers. Implementations send the sponge state
/// to remote GPUs and return the first valid witness found.
///
/// Security: this does NOT modify PoW difficulty. It only parallelizes the search
/// by having different GPUs search non-overlapping witness subsets (interleaving).
pub trait DistributedGrindHelper: Send + Sync {
    /// Number of remote GPUs available for grinding.
    /// Used by `grind_gpu` to compute the interleave step: step = 1 + num_workers().
    fn num_workers(&self) -> u32;

    /// Launch grinding on all remote workers. Each remote worker i (0-indexed)
    /// searches witnesses at offset (i+1) with the given `witness_step`.
    ///
    /// The caller (local GPU) searches offset=0 with the same step, so the
    /// full search space [0, max_witness] is partitioned into non-overlapping
    /// interleaved slices across all participants.
    ///
    /// Returns Ok(Some(witness)) from the first worker to find one, Ok(None)
    /// if no remote worker found a valid witness, Err on failure.
    fn grind_remote(
        &self,
        sponge_state: &DeviceBn254SpongeState,
        bits: u32,
        max_witness: u32,
        witness_step: u32,
    ) -> Result<Option<u32>, Box<dyn std::error::Error + Send + Sync>>;
}

std::thread_local! {
    static DISTRIBUTED_GRIND_HELPER: std::cell::RefCell<Option<Arc<dyn DistributedGrindHelper>>> =
        const { std::cell::RefCell::new(None) };
}

/// Install a distributed grind helper for the current thread.
/// The helper will be used by all subsequent `grind_gpu()` calls on this thread
/// to race local GPU grinding against remote worker(s).
pub fn set_distributed_grind_helper(helper: Arc<dyn DistributedGrindHelper>) {
    DISTRIBUTED_GRIND_HELPER.with(|h| *h.borrow_mut() = Some(helper));
}

/// Remove the distributed grind helper from the current thread.
pub fn clear_distributed_grind_helper() {
    DISTRIBUTED_GRIND_HELPER.with(|h| *h.borrow_mut() = None);
}

fn get_distributed_grind_helper() -> Option<Arc<dyn DistributedGrindHelper>> {
    DISTRIBUTED_GRIND_HELPER.with(|h| h.borrow().clone())
}

/// Bn254 digest type: one BN254 scalar element.
type Digest = [Bn254Scalar; 1];

// ---------------------------------------------------------------------------
// DeviceBn254SpongeState — must match `DeviceBn254SpongeState` in bn254_poseidon2.cu
// ---------------------------------------------------------------------------

/// `#[repr(C)]` mirror of a [`Transcript`]'s
/// [`snapshot`](openvm_stark_backend::transcript::multi_field::MultiFieldTranscript::snapshot)
/// for GPU grinding.
///
/// Populated in [`MultiFieldTranscriptGpu::sync_h2d`], then memcpy'd to the
/// device for CUDA grinding kernels.
///
/// Layout must exactly match the CUDA struct `DeviceBn254SpongeState`:
/// ```text
/// struct DeviceBn254SpongeState {
///     Bn254Fr  sponge_state[3];    // 96 bytes
///     uint32_t absorb_idx;         //  4 bytes
///     uint32_t sample_idx;         //  4 bytes
///     uint32_t observe_buf[8];     // 32 bytes
///     uint32_t observe_buf_len;    //  4 bytes
///     // total = 140 + 4 padding = 144 bytes (aligned to 8)
/// };
/// ```
#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct DeviceBn254SpongeState {
    pub sponge_state: [[u64; 4]; 3], // 96 bytes
    pub absorb_idx: u32,             // 4 bytes
    pub sample_idx: u32,             // 4 bytes
    pub observe_buf: [u32; 8],       // 32 bytes
    pub observe_buf_len: u32,        // 4 bytes + 4 padding = 144 total
}

// Compile-time FFI safety: `Bn254Scalar` ↔ `[u64; 4]` conversion is sound only if
// size and alignment match.  `p3_bn254::Bn254` is a newtype `{ value: [u64; 4] }`
// without `#[repr(C)]`, so we guard against upstream layout changes here.
const _: () = assert!(
    std::mem::size_of::<Bn254Scalar>() == std::mem::size_of::<[u64; 4]>(),
    "Bn254Scalar must be 32 bytes (same as [u64; 4])"
);
const _: () = assert!(
    std::mem::align_of::<Bn254Scalar>() == std::mem::align_of::<[u64; 4]>(),
    "Bn254Scalar alignment must match [u64; 4]"
);
const _: () = assert!(
    std::mem::size_of::<DeviceBn254SpongeState>() == 144,
    "DeviceBn254SpongeState must be 144 bytes to match CUDA struct"
);

/// Extract the Montgomery-form `[u64; 4]` limbs from a `Bn254Scalar`.
///
/// # Safety
///
/// This reinterprets `Bn254Scalar` memory as `[u64; 4]` via a pointer cast,
/// which depends on layout compatibility. `Bn254Scalar` (`p3_bn254::Bn254`) is
/// a single-field newtype `{ value: [u64; 4] }` with identical size and
/// alignment (guarded by the const assertions above).
fn bn254_scalar_to_raw(s: Bn254Scalar) -> [u64; 4] {
    unsafe { std::ptr::read((&s as *const Bn254Scalar).cast::<[u64; 4]>()) }
}

// ---------------------------------------------------------------------------
// MultiFieldTranscriptGpu
// ---------------------------------------------------------------------------

/// GPU-accelerated transcript for the BabyBear-BN254 Poseidon2 proving system.
///
/// Wraps the CPU [`Transcript`] (a `MultiFieldTranscript`) and adds a device
/// buffer for GPU grinding. All observe/sample operations delegate to the inner
/// CPU transcript. Only [`grind_gpu`](GpuFiatShamirTranscript::grind_gpu)
/// touches the GPU: it snapshots the transcript state to the device, runs the
/// CUDA grinding kernel, then updates the host transcript with the result.
///
/// The device snapshot is intentionally not a full serialization of
/// [`Transcript`]: it omits the transcript's buffered sampled values
/// (`sample_buf`). This wrapper is therefore intended for the grinding flow,
/// where device-side execution observes a witness and consumes a fresh sample,
/// not for arbitrary continuation from a host transcript with buffered samples.
#[derive(Debug)]
pub struct MultiFieldTranscriptGpu {
    inner: Transcript,
    device: DeviceBuffer<DeviceBn254SpongeState>,
}

impl Default for MultiFieldTranscriptGpu {
    fn default() -> Self {
        Self {
            inner: Transcript::from(default_bn254_poseidon2_width3()),
            device: DeviceBuffer::new(),
        }
    }
}

impl Clone for MultiFieldTranscriptGpu {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            device: DeviceBuffer::new(),
        }
    }
}

impl MultiFieldTranscriptGpu {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a `DeviceBn254SpongeState` from the current CPU transcript state.
    /// Used both for H2D sync (kernel launch) and for distributed grinding (network serialization).
    fn snapshot_sponge_state(&self) -> DeviceBn254SpongeState {
        let mut ds = DeviceBn254SpongeState::default();
        for (i, &s) in self.inner.sponge_state().iter().enumerate() {
            ds.sponge_state[i] = bn254_scalar_to_raw(s);
        }
        ds.absorb_idx = self.inner.absorb_idx() as u32;
        ds.sample_idx = self.inner.sample_idx() as u32;
        for (i, &bb) in self.inner.observe_buf().iter().enumerate() {
            ds.observe_buf[i] = bb.as_canonical_u32();
        }
        ds.observe_buf_len = self.inner.observe_buf().len() as u32;
        ds
    }

    fn ensure_device_allocated(&mut self, device_ctx: &GpuDeviceCtx) {
        if self.device.is_empty() {
            self.device = DeviceBuffer::with_capacity_on(1, device_ctx);
        }
    }

    /// Snapshot the CPU transcript state to the device buffer.
    ///
    /// This copies the sponge state, sponge indices, and pending `observe_buf`,
    /// but it does not copy the transcript's buffered sampled values
    /// (`sample_buf`).
    ///
    /// Call this before launching a GPU grinding kernel. If the inner
    /// [`Transcript`] still has buffered samples from a prior host-side
    /// `sample()`, device-side sampling after this snapshot can diverge from the
    /// host transcript.
    pub fn sync_h2d(&mut self, device_ctx: &GpuDeviceCtx) -> Result<(), MemCopyError> {
        self.ensure_device_allocated(device_ctx);

        let ds = self.snapshot_sponge_state();

        unsafe {
            cuda_memcpy_on::<false, true>(
                self.device.as_mut_ptr() as *mut c_void,
                &ds as *const DeviceBn254SpongeState as *const c_void,
                std::mem::size_of::<DeviceBn254SpongeState>(),
                device_ctx,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Trait impls — delegate to inner CPU transcript
// ---------------------------------------------------------------------------

impl FiatShamirTranscript<BabyBearBn254Poseidon2Config> for MultiFieldTranscriptGpu {
    fn observe(&mut self, value: BabyBear) {
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut self.inner, value);
    }

    fn sample(&mut self) -> BabyBear {
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut self.inner)
    }

    fn observe_commit(&mut self, digest: Digest) {
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe_commit(
            &mut self.inner,
            digest,
        );
    }
}

impl GpuFiatShamirTranscript<BabyBearBn254Poseidon2Config> for MultiFieldTranscriptGpu {
    fn grind_gpu(
        &mut self,
        bits: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<BabyBear, GrindError> {
        validate_gpu_grind_bits(bits)?;
        if bits == 0 {
            return Ok(BabyBear::ZERO);
        }

        // 1. Sync host state to device.
        self.sync_h2d(device_ctx)?;

        // 2. Run grinding (distributed-interleaved if available, else local-only).
        let max_witness = BabyBear::ORDER_U32 - 1;
        let helper = get_distributed_grind_helper();

        let witness_u32 = if let Some(helper) = helper {
            // Distributed-interleaved mode: partition [0, max_witness] across N+1
            // participants (1 local GPU + N remote workers). Each searches every
            // (N+1)-th candidate starting at its assigned offset.
            let num_remote = helper.num_workers();
            let step = 1 + num_remote; // total participants

            let ds = self.snapshot_sponge_state();
            let bits_u32 = bits as u32;

            let (tx, rx) = std::sync::mpsc::channel::<u32>();

            // Spawn local GPU thread (offset=0: candidates 0, step, 2*step, ...)
            let ds_local = ds.clone();
            let tx_local = tx.clone();
            std::thread::spawn(move || {
                if let Ok(w) = crate::cuda::bn254_merkle_tree::bn254_sponge_grind_on_new_stream(
                    &ds_local,
                    bits_u32,
                    0,
                    max_witness,
                    step,
                ) {
                    let _ = tx_local.send(w);
                }
            });

            // Spawn remote workers thread (offsets 1..=N, same step)
            let ds_remote = ds.clone();
            let tx_remote = tx.clone();
            std::thread::spawn(move || {
                match helper.grind_remote(&ds_remote, bits_u32, max_witness, step) {
                    Ok(Some(w)) => {
                        let _ = tx_remote.send(w);
                    }
                    _ => {}
                }
            });

            drop(tx);

            match rx.recv() {
                Ok(w) => w,
                Err(_) => {
                    // All threads exited without sending — fall back to single-GPU full search.
                    unsafe {
                        crate::cuda::bn254_merkle_tree::bn254_sponge_grind(
                            self.device.as_ptr(),
                            bits as u32,
                            max_witness,
                            device_ctx,
                        )?
                    }
                }
            }
        } else {
            // No distributed helper — local GPU only, full range, step=1.
            unsafe {
                crate::cuda::bn254_merkle_tree::bn254_sponge_grind(
                    self.device.as_ptr(),
                    bits as u32,
                    max_witness,
                    device_ctx,
                )?
            }
        };

        let witness = BabyBear::from_u32(witness_u32);

        // 3. Update host state: observe witness + consume one sample.
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut self.inner, witness);
        let _ = FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample(&mut self.inner);

        Ok(witness)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use openvm_cuda_common::{common::get_device, stream::GpuDeviceCtx};
    use openvm_stark_backend::FiatShamirTranscript;
    use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::default_transcript;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    /// Exercises the CUDA grinding kernel end-to-end: the kernel must correctly
    /// implement observe + sample (packing, sponge permutation, base-p decomposition)
    /// to find a valid witness. We verify the witness against the CPU transcript.
    #[test]
    fn test_grind_gpu_witness_valid_on_cpu() {
        let bits = 8;
        let ctx = GpuDeviceCtx::for_device(get_device().unwrap() as u32).unwrap();

        // Test with several different transcript states to exercise partial observe buffers,
        // different sponge positions, etc.
        for num_observed in [0, 1, 3, 7, 8, 9, 15, 16, 17] {
            let mut gpu = MultiFieldTranscriptGpu::new();
            let mut cpu = default_transcript();

            for i in 0..num_observed {
                let val = BabyBear::from_u32((i as u32).wrapping_mul(41).wrapping_add(7));
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, val);
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, val);
            }

            let witness = gpu
                .grind_gpu(bits, &ctx)
                .unwrap_or_else(|e| panic!("grind_gpu failed with {num_observed} observed: {e:?}"));
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, witness);
            let witness_bits =
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample_bits(&mut cpu, bits);
            // Verify the CUDA-found witness passes check_witness on the CPU transcript.
            assert_eq!(
                witness_bits, 0,
                "CUDA witness {witness:?} invalid on CPU (observed {num_observed} values, witness_bits {witness_bits})"
            );
        }
    }

    #[test]
    fn test_grind_gpu_15bit_timing() {
        let ctx = GpuDeviceCtx::for_device(get_device().unwrap() as u32).unwrap();

        // Warmup: first kernel launch triggers JIT compilation
        {
            let mut gpu = MultiFieldTranscriptGpu::new();
            let _ = gpu.grind_gpu(8, &ctx);
        }

        // Test correctness and timing at 15 bits
        let bits = 15;
        let mut gpu = MultiFieldTranscriptGpu::new();
        for i in 0..20u32 {
            let val = BabyBear::from_u32(i.wrapping_mul(37).wrapping_add(100));
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, val);
        }

        let start = std::time::Instant::now();
        let witness = gpu
            .grind_gpu(bits, &ctx)
            .unwrap_or_else(|e| panic!("grind_gpu failed at {bits} bits: {e:?}"));
        let elapsed = start.elapsed();
        eprintln!(
            "BN254 grind: bits={bits}, witness={}, elapsed={:?}",
            witness.as_canonical_u32(),
            elapsed
        );
        assert!(
            elapsed.as_secs() < 5,
            "15-bit grind took {elapsed:?} — expected < 5s (stride loop bug?)"
        );

        // Verify correctness
        let mut cpu = default_transcript();
        for i in 0..20u32 {
            let val = BabyBear::from_u32(i.wrapping_mul(37).wrapping_add(100));
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, val);
        }
        FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, witness);
        let witness_bits =
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample_bits(&mut cpu, bits);
        assert_eq!(
            witness_bits, 0,
            "CUDA witness invalid on CPU at {bits} bits"
        );
    }

    /// Diagnostic: verify a CPU-computed witness on GPU to detect sponge divergence.
    #[test]
    fn test_bn254_sponge_gpu_cpu_agreement() {
        use crate::cuda::bn254_merkle_tree::bn254_verify_witness;

        let ctx = GpuDeviceCtx::for_device(get_device().unwrap() as u32).unwrap();
        let bits = 15usize;

        // Build identical states on GPU and CPU
        let mut gpu = MultiFieldTranscriptGpu::new();
        let mut cpu = default_transcript();

        for i in 0..20u32 {
            let val = BabyBear::from_u32(i.wrapping_mul(37).wrapping_add(100));
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut gpu, val);
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu, val);
        }

        // Find the FIRST valid witness on CPU (brute force)
        let cpu_witness = {
            let mut w = 0u32;
            loop {
                let mut cpu_clone = cpu.clone();
                let val = BabyBear::from_u32(w);
                FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(&mut cpu_clone, val);
                let sample = FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample_bits(
                    &mut cpu_clone,
                    bits,
                );
                if sample == 0 {
                    break w;
                }
                w += 1;
                if w > 200_000 {
                    panic!("Could not find CPU witness in first 200K candidates");
                }
            }
        };
        eprintln!("CPU found valid witness: w={cpu_witness}");

        // Now sync GPU state and verify this CPU-found witness on GPU
        gpu.sync_h2d(&ctx).unwrap();
        let gpu_result = unsafe {
            bn254_verify_witness(gpu.device.as_ptr(), bits as u32, cpu_witness, &ctx)
                .expect("GPU verify kernel failed")
        };
        eprintln!("GPU verify of CPU witness {cpu_witness}: sample_bits={gpu_result} (expected 0)");

        // Also check witness=0 on both
        let cpu_check_0 = {
            let mut cpu_clone = cpu.clone();
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::observe(
                &mut cpu_clone,
                BabyBear::from_u32(0),
            );
            FiatShamirTranscript::<BabyBearBn254Poseidon2Config>::sample_bits(&mut cpu_clone, bits)
        };
        let gpu_check_0 = unsafe {
            bn254_verify_witness(gpu.device.as_ptr(), bits as u32, 0, &ctx)
                .expect("GPU verify kernel failed for w=0")
        };
        eprintln!("w=0: CPU sample_bits={cpu_check_0}, GPU sample_bits={gpu_check_0}");

        assert_eq!(
            gpu_result, 0,
            "GPU and CPU DISAGREE on witness {cpu_witness}: GPU says {gpu_result}, CPU says 0"
        );
        assert_eq!(
            gpu_check_0, cpu_check_0 as u32,
            "GPU/CPU disagree on w=0: GPU={gpu_check_0}, CPU={cpu_check_0}"
        );
    }
}
