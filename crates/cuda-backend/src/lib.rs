// cudaStream_t is an opaque CUDA handle (*mut c_void) passed through to FFI.
// Clippy's not_unsafe_ptr_arg_deref fires on functions that accept it, but the
// "pointer" is never dereferenced in Rust — it is just forwarded to CUDA runtime calls.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod base;
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub mod bn254_sponge;
pub mod data_transporter;
pub mod hash_scheme;
pub mod logup_zerocheck;
pub mod merkle_tree;
pub mod monomial;
pub mod ntt;
pub mod poly;
pub mod sponge;
pub mod stacked_pcs;
pub mod stacked_reduction;
pub mod utils;
pub mod whir;

/// Rust bindings for CUDA kernels
pub mod cuda;
mod device;
mod engine;
mod error;
mod gpu_backend;
mod pkey;
mod sumcheck;
mod types;
#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub use bn254_sponge::{DeviceBn254SpongeState, MultiFieldTranscriptGpu};
pub use device::*;
pub use engine::*;
pub use error::*;
pub use gpu_backend::*;
pub use hash_scheme::*;
pub use pkey::*;

#[cfg(test)]
mod tests;

pub mod prelude {
    pub use crate::types::*;
}

// ─── Distributed Grinding Interface ──────────────────────────────────────────
// Required by openvm-distributed for compilation compatibility.
// Per protocol design, grinding is local per worker and these are only called
// when --grind-workers are explicitly configured (rare).

#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub trait DistributedGrindHelper: Send + Sync {
    fn num_workers(&self) -> u32;
    fn grind_remote(
        &self,
        sponge_state: &DeviceBn254SpongeState,
        bits: u32,
        max_witness: u32,
        witness_step: u32,
    ) -> Result<Option<u32>, Box<dyn std::error::Error + Send + Sync>>;
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
static DISTRIBUTED_GRIND_HELPER: std::sync::OnceLock<
    std::sync::Mutex<Option<std::sync::Arc<dyn DistributedGrindHelper>>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub fn set_distributed_grind_helper(helper: std::sync::Arc<dyn DistributedGrindHelper>) {
    let lock = DISTRIBUTED_GRIND_HELPER.get_or_init(|| std::sync::Mutex::new(None));
    *lock.lock().unwrap() = Some(helper);
}

#[cfg(feature = "baby-bear-bn254-poseidon2")]
pub fn clear_distributed_grind_helper() {
    if let Some(lock) = DISTRIBUTED_GRIND_HELPER.get() {
        *lock.lock().unwrap() = None;
    }
}
