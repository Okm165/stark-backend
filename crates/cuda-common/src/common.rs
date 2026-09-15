use std::{
    ffi::c_void,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::error::{check, CudaError};

crate::gpu_link! {
    #[cfg_attr(gpu_vendor_amd, link_name = "hipFree")]
    fn cudaFree(dev_ptr: *mut c_void) -> i32;
    #[cfg_attr(gpu_vendor_amd, link_name = "hipGetDevice")]
    fn cudaGetDevice(device: *mut i32) -> i32;
    #[cfg_attr(gpu_vendor_amd, link_name = "hipSetDevice")]
    fn cudaSetDevice(device: i32) -> i32;
}

/// Monotonic counter for device-level resets. Kernel constant caches
/// (twiddle factors, round constants) use `(device_id, epoch)` as a key
/// to re-initialize after a hypothetical device reset. In normal operation
/// the epoch stays at 0 — we never call `hipDeviceReset`/`cudaDeviceReset`
/// because it invalidates static GPU state held by libraries like halo2-gpu.
/// Memory cleanup is done via `release_and_reinit_pool()` instead.
static DEVICE_RESET_EPOCH: AtomicU64 = AtomicU64::new(0);

pub fn get_device() -> Result<i32, CudaError> {
    let mut device = 0;
    unsafe {
        check(cudaGetDevice(&mut device))?;
    }
    assert!(device >= 0);
    Ok(device)
}

pub fn device_reset_epoch() -> u64 {
    DEVICE_RESET_EPOCH.load(Ordering::Acquire)
}

pub fn set_device_by_id(device: i32) -> Result<(), CudaError> {
    assert!(device >= 0);
    unsafe {
        check(cudaSetDevice(device))?;
        check(cudaFree(std::ptr::null_mut()))?;
    }
    Ok(())
}

pub fn set_device() -> Result<i32, CudaError> {
    let device = get_device()?;
    set_device_by_id(device)?;
    Ok(device)
}
