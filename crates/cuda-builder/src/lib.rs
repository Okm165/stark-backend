use std::{collections::BTreeSet, env, path::Path, process::Command, sync::OnceLock};

/// GPU vendor detected at build time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
}

/// Detect which GPU toolchain is available.
///
/// Resolution order:
/// 1. `GPU_VENDOR` env var (values: `nvidia`/`cuda` or `amd`/`hip`/`rocm`)
/// 2. Probe for `nvcc` on `$PATH`
/// 3. Probe for `hipcc` at `$ROCM_PATH/bin/hipcc` (default `/opt/rocm`)
///
/// Returns `None` if neither compiler is found.
pub fn detect_gpu_vendor() -> Option<GpuVendor> {
    static CACHED: OnceLock<Option<GpuVendor>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Ok(v) = env::var("GPU_VENDOR") {
            return match v.to_lowercase().as_str() {
                "nvidia" | "cuda" => Some(GpuVendor::Nvidia),
                "amd" | "hip" | "rocm" => Some(GpuVendor::Amd),
                _ => None,
            };
        }
        if Command::new("nvcc")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return Some(GpuVendor::Nvidia);
        }
        if Command::new(hipcc_path())
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
        {
            return Some(GpuVendor::Amd);
        }
        None
    })
}

/// Returns `true` if any GPU compiler (`nvcc` or `hipcc`) is available.
pub fn cuda_available() -> bool {
    detect_gpu_vendor().is_some()
}

/// Emit `cargo:rustc-cfg=gpu_vendor_amd` when building for AMD.
///
/// Call this from `build.rs` files that contain Rust code with
/// `#[cfg(gpu_vendor_amd)]` or `#[cfg_attr(gpu_vendor_amd, ...)]`.
pub fn emit_gpu_vendor_cfg() {
    println!("cargo:rerun-if-env-changed=GPU_VENDOR");
    if detect_gpu_vendor() == Some(GpuVendor::Amd) {
        println!("cargo:rustc-cfg=gpu_vendor_amd");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CudaBuilder
// ═══════════════════════════════════════════════════════════════════════

/// GPU kernel builder — compiles `.cu` files with `nvcc` (NVIDIA) or `hipcc` (AMD).
///
/// The public API is identical regardless of GPU vendor.  Vendor-specific
/// behavior (compiler selection, flags, architecture detection, link libraries)
/// is handled internally based on [`detect_gpu_vendor`].
///
/// # Example
///
/// ```ignore
/// CudaBuilder::new()
///     .library_name("my_kernels")
///     .include("cuda/include")
///     .file("cuda/src/kernel.cu")
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct CudaBuilder {
    include_paths: Vec<String>,
    source_files: Vec<String>,
    watch_paths: Vec<String>,
    watch_globs: Vec<String>,
    library_name: String,
    gpu_arch: Vec<String>,
    cuda_opt_level: Option<String>,
    lineinfo: bool,
    custom_flags: Vec<String>,
    link_libraries: Vec<String>,
    link_search_paths: Vec<String>,
    vendor: GpuVendor,
}

impl Default for CudaBuilder {
    fn default() -> Self {
        let vendor = detect_gpu_vendor().unwrap_or(GpuVendor::Nvidia);

        let (link_libraries, link_search_paths, custom_flags) = match vendor {
            GpuVendor::Nvidia => {
                let search = if let Ok(d) = env::var("CUDA_LIB_DIR") {
                    vec![d]
                } else {
                    vec!["/usr/local/cuda/lib64".to_string()]
                };
                (
                    vec!["cudart".to_string(), "cuda".to_string()],
                    search,
                    vec![
                        "--std=c++17".to_string(),
                        "--expt-relaxed-constexpr".to_string(),
                        "-Xfatbin=-compress-all".to_string(),
                    ],
                )
            }
            GpuVendor::Amd => {
                let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
                (
                    vec!["amdhip64".to_string(), "stdc++".to_string()],
                    vec![format!("{}/lib", rocm_path)],
                    vec!["--std=c++17".to_string()],
                )
            }
        };

        Self {
            include_paths: Vec::new(),
            source_files: Vec::new(),
            watch_paths: vec!["build.rs".to_string()],
            watch_globs: Vec::new(),
            library_name: String::new(),
            gpu_arch: Vec::new(),
            cuda_opt_level: None,
            lineinfo: false,
            custom_flags,
            link_libraries,
            link_search_paths,
            vendor,
        }
    }
}

impl CudaBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn library_name(mut self, name: &str) -> Self {
        self.library_name = name.to_string();
        self
    }

    pub fn include<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.include_paths.push(path_str.clone());
        self.watch_paths.push(path_str);
        self
    }

    pub fn include_from_dep(mut self, dep_env_var: &str) -> Self {
        if let Ok(path) = env::var(dep_env_var) {
            self.include_paths.push(path);
        }
        self
    }

    pub fn file<P: AsRef<Path>>(mut self, path: P) -> Self {
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.source_files.push(path_str.clone());
        self.watch_paths.push(path_str);
        self
    }

    pub fn files<P: AsRef<Path>, I: IntoIterator<Item = P>>(mut self, paths: I) -> Self {
        for path in paths {
            let path_str = path.as_ref().to_string_lossy().to_string();
            self.source_files.push(path_str.clone());
            self.watch_paths.push(path_str);
        }
        self
    }

    pub fn files_from_glob(mut self, pattern: &str) -> Self {
        self.watch_globs.push(pattern.to_string());
        for path in glob::glob(pattern).expect("Invalid glob pattern").flatten() {
            if path.is_file() && path.extension().is_some_and(|ext| ext == "cu") {
                self.source_files.push(path.to_string_lossy().to_string());
            }
        }
        self
    }

    pub fn watch<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.watch_paths
            .push(path.as_ref().to_string_lossy().to_string());
        self
    }

    pub fn watch_glob(mut self, pattern: &str) -> Self {
        self.watch_globs.push(pattern.to_string());
        self
    }

    /// Set GPU architecture.  For NVIDIA: e.g. `"75"`, `"80"`.
    /// For AMD: e.g. `"gfx1100"`.  When unset, architecture is auto-detected.
    pub fn cuda_arch(mut self, arch: &str) -> Self {
        self.gpu_arch = vec![arch.to_string()];
        self
    }

    pub fn cuda_archs(mut self, archs: Vec<&str>) -> Self {
        self.gpu_arch = archs.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn cuda_opt_level(mut self, level: u8) -> Self {
        self.cuda_opt_level = Some(level.to_string());
        self
    }

    /// Enable line info for profiling (NCU on NVIDIA).  Keeps other
    /// optimizations intact.
    pub fn lineinfo(mut self, enabled: bool) -> Self {
        self.lineinfo = enabled;
        self
    }

    /// Add a custom compiler flag.  nvcc-specific flags (e.g. `-Xcompiler`,
    /// `-gencode`) are silently filtered out when compiling with hipcc.
    pub fn flag(mut self, flag: &str) -> Self {
        self.custom_flags.push(flag.to_string());
        self
    }

    pub fn link_lib(mut self, lib: &str) -> Self {
        self.link_libraries.push(lib.to_string());
        self
    }

    pub fn link_search<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.link_search_paths
            .push(path.as_ref().to_string_lossy().to_string());
        self
    }

    /// Compile all registered `.cu` source files into a static library.
    pub fn build(self) {
        self.validate();
        self.setup_rerun_conditions();

        match self.vendor {
            GpuVendor::Nvidia => self.build_nvidia(),
            GpuVendor::Amd => self.build_amd(),
        }
    }

    /// Emit linker directives for the GPU runtime library and the
    /// `gpu_vendor_amd` cfg flag.
    pub fn emit_link_directives(&self) {
        emit_gpu_vendor_cfg();
        for path in &self.link_search_paths {
            println!("cargo:rustc-link-search=native={}", path);
        }
        for lib in &self.link_libraries {
            println!("cargo:rustc-link-lib={}", lib);
        }
    }

    // ── NVIDIA build path ───────────────────────────────────────────

    fn build_nvidia(self) {
        let archs = self.get_gpu_arch();
        let opt_level = self.get_opt_level();

        let mut builder = cc::Build::new();
        builder.cuda(true);

        self.apply_debug_flags_nvidia(&mut builder);

        for inc in &self.include_paths {
            builder.include(inc);
        }
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            builder.include(format!("{}/include", cuda_path));
        }
        for flag in &self.custom_flags {
            builder.flag(flag);
        }

        for arch in &archs {
            builder
                .flag("-gencode")
                .flag(format!("arch=compute_{arch},code=sm_{arch}"));
        }
        if let Some(max_arch) = archs.iter().max() {
            builder
                .flag("-gencode")
                .flag(format!("arch=compute_{max_arch},code=compute_{max_arch}"));
        }

        builder.flag(nvcc_parallel_jobs());

        if opt_level == "0" {
            builder.debug(true).flag("-O0");
        } else {
            builder
                .debug(false)
                .flag(format!("--ptxas-options=-O{opt_level}"));
        }
        if self.get_lineinfo() {
            builder.flag("-lineinfo");
        }
        for file in &self.source_files {
            builder.file(file);
        }

        builder.compile(&self.library_name);
    }

    // ── AMD HIP build path ─────────────────────────────────────────
    //
    // HIP's `__constant__` variables have protected (internal) visibility,
    // so `extern __constant__` references across translation units fail
    // without `-fgpu-rdc` (Relocatable Device Code).  We enable `-fgpu-rdc`
    // and use a three-phase build:
    //
    //  1. Compile each `.cu` → relocatable `.o`
    //  2. Device-link all `.o` → single `device_linked.o` (merges device code + host stubs,
    //     resolves cross-TU `__constant__` references)
    //  3. Archive `device_linked.o` into a static library
    //
    // Phase 2 uses `hipcc --hip-link -r` which performs a relocatable
    // partial link.  Header-defined `__forceinline__` functions may produce
    // multiple host definitions across TUs; we pass `--allow-multiple-definition`
    // to the host linker to let the partial link succeed.  The duplicate
    // definitions are identical (same header, same code), so this is safe.

    fn build_amd(self) {
        let archs = self.get_gpu_arch();
        let opt_level = self.get_opt_level();

        let compat_dir = cuda_builder_include_dir();
        let stubs_dir = format!("{compat_dir}/hip-stubs");
        let cuda2hip = format!("{compat_dir}/cuda2hip.hpp");

        let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");

        let mut common_flags: Vec<String> = Vec::new();
        common_flags.push("-fgpu-rdc".to_string());
        common_flags.push("-x".to_string());
        common_flags.push("hip".to_string());
        common_flags.push("-include".to_string());
        common_flags.push(cuda2hip.clone());
        common_flags.push(format!("-I{stubs_dir}"));

        for arch in &archs {
            common_flags.push(format!("--offload-arch={arch}"));
        }
        for flag in &self.custom_flags {
            if !is_nvcc_only_flag(flag) {
                common_flags.push(flag.clone());
            }
        }
        for inc in &self.include_paths {
            common_flags.push(format!("-I{inc}"));
        }
        common_flags.push(format!("-O{opt_level}"));

        // gfx1100 (RDNA3): LLVM's AMDGPU backend can emit >63 outstanding
        // loads without s_waitcnt, overflowing the 6-bit VMCNT counter and
        // causing stale reads (LLVM #172932).  +precise-memory forces a
        // s_waitcnt after every memory operation, preventing this.
        // Disable with AMDGPU_PRECISE_MEMORY=0.
        if env::var("AMDGPU_PRECISE_MEMORY").as_deref() != Ok("0") {
            common_flags.push("-Xarch_device".to_string());
            common_flags.push("-mattr=+precise-memory".to_string());
        }

        // Phase 1: compile each .cu file to a relocatable .o
        let hipcc = hipcc_path();
        let mut obj_files = Vec::new();
        let mut seen_sources: BTreeSet<String> = BTreeSet::new();

        for (idx, src) in self.source_files.iter().enumerate() {
            if !seen_sources.insert(src.clone()) {
                continue;
            }
            let stem = Path::new(src)
                .file_stem()
                .expect("source file has no stem")
                .to_string_lossy();
            let obj_path = format!("{out_dir}/hip_{idx}_{stem}.o");

            let mut cmd = Command::new(hipcc);
            cmd.args(&common_flags)
                .arg("-fPIC")
                .arg("-c")
                .arg(src)
                .arg("-o")
                .arg(&obj_path);

            let status = cmd
                .status()
                .unwrap_or_else(|e| panic!("failed to run hipcc for {src}: {e}"));
            if !status.success() {
                panic!("hipcc failed to compile {src} (exit: {status})");
            }
            obj_files.push(obj_path);
        }

        let lib_path = format!("{out_dir}/lib{}.a", self.library_name);
        let _ = std::fs::remove_file(&lib_path);

        // Phase 2: device-link all .o files into a single partially-linked
        // object.  `--hip-link -r` merges device code from all TUs and
        // resolves cross-TU __constant__ references.
        let device_linked = format!("{out_dir}/device_linked_{}.o", self.library_name);
        {
            let mut cmd = Command::new(hipcc);
            cmd.arg("-fgpu-rdc")
                .arg("--hip-link")
                .arg("-fPIC")
                .arg("-r")
                .arg("-Wl,--allow-multiple-definition");
            for arch in &archs {
                cmd.arg(format!("--offload-arch={arch}"));
            }
            if env::var("AMDGPU_PRECISE_MEMORY").as_deref() != Ok("0") {
                cmd.arg("-Xarch_device").arg("-mattr=+precise-memory");
            }
            cmd.args(&obj_files).arg("-o").arg(&device_linked);

            let status = cmd
                .status()
                .unwrap_or_else(|e| panic!("hipcc device-link failed: {e}"));
            if !status.success() {
                panic!("hipcc device-link failed (exit: {status})");
            }
        }

        println!("cargo:rustc-link-arg=-Wl,--allow-multiple-definition");

        // Phase 3: archive the device-linked object.
        let mut cmd = Command::new("ar");
        cmd.arg("rcs").arg(&lib_path).arg(&device_linked);
        let status = cmd.status().unwrap_or_else(|e| panic!("ar failed: {e}"));
        if !status.success() {
            panic!("ar failed (exit: {status})");
        }

        println!("cargo:rustc-link-search=native={out_dir}");
        println!("cargo:rustc-link-lib=static={}", self.library_name);
    }

    // ── Shared helpers ──────────────────────────────────────────────

    fn validate(&self) {
        if self.library_name.is_empty() {
            panic!(
                "Library name must be set using .library_name(\"name\") before calling .build()"
            );
        }
        if self.source_files.is_empty() {
            panic!("At least one source file must be added using .file() or .files() before calling .build()");
        }
        for file in &self.source_files {
            if !Path::new(file).exists() {
                eprintln!("cargo:warning=GPU source file does not exist: {file}");
            }
        }
        for include in &self.include_paths {
            if !Path::new(include).exists() {
                eprintln!("cargo:warning=Include path does not exist: {include}");
            }
        }
    }

    fn setup_rerun_conditions(&self) {
        for var in [
            "CUDA_ARCH",
            "CUDA_OPT_LEVEL",
            "CUDA_DEBUG",
            "CUDA_LINEINFO",
            "NVCC_THREADS",
            "GPU_VENDOR",
            "ROCM_PATH",
            "HIP_ARCH",
            "HIPCC_PATH",
            "AMDGPU_PRECISE_MEMORY",
        ] {
            println!("cargo:rerun-if-env-changed={var}");
        }

        for path in &self.watch_paths {
            println!("cargo:rerun-if-changed={path}");
        }
        for pattern in &self.watch_globs {
            watch_glob(pattern);
        }
    }

    fn get_gpu_arch(&self) -> Vec<String> {
        if !self.gpu_arch.is_empty() {
            return self.gpu_arch.clone();
        }
        if let Ok(env_archs) = env::var("CUDA_ARCH") {
            let archs: Vec<String> = env_archs
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if !archs.is_empty() {
                return archs;
            }
        }
        match self.vendor {
            GpuVendor::Nvidia => vec![detect_nvidia_arch()],
            GpuVendor::Amd => vec![detect_amd_arch()],
        }
    }

    fn get_opt_level(&self) -> String {
        self.cuda_opt_level
            .clone()
            .or_else(|| env::var("CUDA_OPT_LEVEL").ok())
            .unwrap_or_else(|| {
                // AMD gfx1100 (RDNA3): -O3 triggers LLVM miscompilations in
                // large shared-memory reductions, producing incorrect sumcheck
                // polynomials.  -O2 is equally fast and avoids the bug.
                if self.vendor == GpuVendor::Amd {
                    "2".to_string()
                } else {
                    "3".to_string()
                }
            })
    }

    fn get_lineinfo(&self) -> bool {
        self.lineinfo || env::var("CUDA_LINEINFO").is_ok_and(|v| v == "1")
    }

    fn apply_debug_flags_nvidia(&self, builder: &mut cc::Build) {
        if !env::var("CUDA_DEBUG").is_ok_and(|v| v == "1") {
            return;
        }

        env::set_var("CUDA_OPT_LEVEL", "0");
        env::set_var("CUDA_LAUNCH_BLOCKING", "1");
        env::set_var("RUST_BACKTRACE", "full");
        env::set_var("CUDA_ENABLE_COREDUMP_ON_EXCEPTION", "1");
        env::set_var("CUDA_DEVICE_WAITS_ON_EXCEPTION", "1");

        println!("cargo:warning=CUDA_DEBUG=1: O0, LAUNCH_BLOCKING=1, device debug symbols");

        builder.flag("-G");
        builder.flag("-Xcompiler=-fno-omit-frame-pointer");
        builder.flag("-Xptxas=-v");
        builder.define("CUDA_DEBUG", "1");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Architecture detection
// ═══════════════════════════════════════════════════════════════════════

/// Detect NVIDIA GPU compute capability via `nvidia-smi`.
pub fn detect_nvidia_arch() -> String {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .expect("Failed to run nvidia-smi — are NVIDIA drivers installed?");

    let stdout = String::from_utf8(output.stdout).expect("nvidia-smi output is not valid UTF-8");

    let arch = stdout
        .lines()
        .next()
        .expect("nvidia-smi returned no compute capability")
        .trim()
        .replace('.', "");

    println!("cargo:rustc-env=CUDA_ARCH={arch}");
    env::set_var("CUDA_ARCH", &arch);
    arch
}

/// Detect AMD GPU architecture via `rocminfo`.
///
/// Parses the first `gfx*` ISA name from `rocminfo` output (e.g. `gfx1100`).
/// The `HIP_ARCH` environment variable can override auto-detection.
pub fn detect_amd_arch() -> String {
    if let Ok(arch) = env::var("HIP_ARCH") {
        println!("cargo:rustc-env=CUDA_ARCH={arch}");
        return arch;
    }

    let output = Command::new("rocminfo")
        .output()
        .expect("Failed to run rocminfo — is ROCm installed?");

    let text = String::from_utf8(output.stdout).expect("rocminfo output is not valid UTF-8");

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(name) = trimmed.strip_prefix("Name:") {
            let name = name.trim();
            if name.starts_with("gfx") {
                println!("cargo:rustc-env=CUDA_ARCH={name}");
                return name.to_string();
            }
        }
    }

    panic!("rocminfo did not report any gfx target — is an AMD GPU present?");
}

// ═══════════════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════════════

/// Calculate optimal number of parallel nvcc jobs based on available CPUs.
pub fn nvcc_parallel_jobs() -> String {
    let threads = env::var("NVCC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });

    format!("-t{threads}")
}

/// Path to this crate's `include/` directory, embedded at compile time.
/// Contains `cuda2hip.hpp` and the `hip-stubs/` folder.
fn cuda_builder_include_dir() -> &'static str {
    env!("CUDA_BUILDER_INCLUDE_DIR")
}

/// Resolve the path to `hipcc`.
///
/// Checks `$HIPCC_PATH`, then falls back to `$ROCM_PATH/bin/hipcc`,
/// defaulting to `/opt/rocm/bin/hipcc`.
fn hipcc_path() -> &'static str {
    static CACHED: OnceLock<String> = OnceLock::new();
    CACHED.get_or_init(|| {
        if let Ok(p) = env::var("HIPCC_PATH") {
            return p;
        }
        let rocm = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        format!("{rocm}/bin/hipcc")
    })
}

/// Returns `true` for flags that are nvcc-specific and should be silently
/// dropped when compiling with hipcc.
fn is_nvcc_only_flag(flag: &str) -> bool {
    // -Xcompiler, -Xfatbin, -Xptxas: nvcc pass-through flags
    // --expt-relaxed-constexpr: nvcc extension
    // -gencode: nvcc multi-arch codegen
    // -t<N>: nvcc parallel compilation threads
    flag.starts_with("-Xcompiler")
        || flag.starts_with("-Xfatbin")
        || flag.starts_with("-Xptxas")
        || flag.starts_with("--expt-")
        || flag.starts_with("-gencode")
        || (flag.starts_with("-t")
            && flag.len() > 2
            && flag[2..].bytes().all(|b| b.is_ascii_digit()))
}

fn watch_glob(pattern: &str) {
    for path in glob::glob(pattern).expect("Invalid glob pattern").flatten() {
        if path.is_file() {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
