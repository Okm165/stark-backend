fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let include_dir = format!("{manifest_dir}/include");
    println!("cargo:rustc-env=CUDA_BUILDER_INCLUDE_DIR={include_dir}");
    println!("cargo:rerun-if-changed=include/cuda2hip.hpp");
}
