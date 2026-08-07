use std::{fs, path::Path};

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let include_dir = format!("{manifest_dir}/include");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let stubs_dir = format!("{out_dir}/hip-stubs");

    generate_hip_stubs(&stubs_dir);

    println!("cargo:rustc-env=CUDA_BUILDER_INCLUDE_DIR={include_dir}");
    println!("cargo:rustc-env=HIP_STUBS_DIR={stubs_dir}");
    println!("cargo:rerun-if-changed=include/cuda2hip.hpp");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Generate minimal header stubs that redirect CUDA includes to HIP equivalents.
/// These are placed on hipcc's include path so `#include <cuda.h>` etc. resolve
/// to either empty stubs or the corresponding HIP/hipCUB header.
fn generate_hip_stubs(dir: &str) {
    let empty_stubs = [
        "cooperative_groups.h",
        "cuda.h",
        "cuda_runtime.h",
        "cuda_runtime_api.h",
        "device_atomic_functions.h",
        "driver_types.h",
        "vector_types.h",
    ];

    let redirect_stubs: &[(&str, &str)] = &[
        ("cub/cub.cuh", "hipcub/hipcub.hpp"),
        (
            "cub/device/device_merge_sort.cuh",
            "hipcub/device/device_merge_sort.hpp",
        ),
        (
            "cub/device/device_reduce.cuh",
            "hipcub/device/device_reduce.hpp",
        ),
        (
            "cub/device/device_scan.cuh",
            "hipcub/device/device_scan.hpp",
        ),
        (
            "cub/device/device_select.cuh",
            "hipcub/device/device_select.hpp",
        ),
    ];

    for name in empty_stubs {
        let path = format!("{dir}/{name}");
        write_if_changed(&path, "/* HIP stub — provided by hipcc runtime */\n");
    }

    for (name, target) in redirect_stubs {
        let path = format!("{dir}/{name}");
        write_if_changed(&path, &format!("#include <{target}>\n"));
    }
}

/// Write file only if contents differ, to avoid unnecessary rebuilds.
fn write_if_changed(path: &str, contents: &str) {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    if p.exists() {
        if let Ok(existing) = fs::read_to_string(p) {
            if existing == contents {
                return;
            }
        }
    }
    fs::write(p, contents).unwrap();
}
