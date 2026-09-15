//! Rust bindings for CUDA/HIP sponge grinding kernel (BabyBear Poseidon2).

use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    error::CudaError,
    stream::{cudaStream_t, GpuDeviceCtx},
};

use crate::sponge::{validate_gpu_grind_bits, DeviceSpongeState, GrindError};

extern "C" {
    fn _sponge_grind(
        init_state: *const DeviceSpongeState,
        bits: u32,
        min_witness: u32,
        max_witness: u32,
        result: *mut u32,
        stream: cudaStream_t,
    ) -> i32;
}

/// Launch the GPU grinding kernel to find a PoW witness.
///
/// Searches the full witness space `[0, max_witness]` in batches of `2^bits`.
/// Each kernel invocation checks one candidate per GPU thread.
///
/// # Safety
/// - `init_state` must point to valid device memory containing a `DeviceSpongeState`
pub unsafe fn sponge_grind(
    init_state: *const DeviceSpongeState,
    bits: u32,
    max_witness: u32,
    device_ctx: &GpuDeviceCtx,
) -> Result<u32, GrindError> {
    validate_gpu_grind_bits(bits as usize)?;
    let mut d_result = DeviceBuffer::with_capacity_on(1, device_ctx);
    [u32::MAX].copy_to_on(&mut d_result, device_ctx)?;

    let step = 1u32 << bits;

    for start in (0..=max_witness).step_by(step as usize) {
        CudaError::from_result(_sponge_grind(
            init_state,
            bits,
            start,
            max_witness,
            d_result.as_mut_ptr(),
            device_ctx.stream.as_raw(),
        ))?;

        let result = d_result.to_host_on(device_ctx)?[0];
        if result < u32::MAX {
            return Ok(result);
        }
    }
    Err(GrindError::WitnessNotFound)
}
