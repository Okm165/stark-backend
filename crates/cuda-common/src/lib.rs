pub mod common;
pub mod copy;
pub mod d_buffer;
pub mod error;
pub mod memory_manager;
pub mod pinned;
pub mod stream;

/// Links an `extern "C"` block against the GPU runtime library.
/// Resolves to `cudart` on NVIDIA or `amdhip64` on AMD (ROCm/HIP).
/// Use inside modules to avoid repeating the cfg_attr link annotations.
#[macro_export]
macro_rules! gpu_link {
    ( $($body:tt)* ) => {
        #[cfg_attr(not(gpu_vendor_amd), link(name = "cudart"))]
        #[cfg_attr(gpu_vendor_amd, link(name = "amdhip64"))]
        extern "C" { $($body)* }
    };
}
