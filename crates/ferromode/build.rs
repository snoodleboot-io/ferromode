// Build script for ferromode crate.
//
// This script handles CUDA kernel compilation if the 'cuda' feature is enabled.

use std::env;
use std::path::PathBuf;

fn main() {
    // Only attempt CUDA compilation if the cuda feature is enabled
    if cfg!(feature = "cuda") {
        compile_cuda_kernels();
    }

    // Signal to Cargo that the build script should be re-run
    // if build configuration changes
    println!("cargo:rerun-if-changed=build.rs");
}

/// Compile CUDA kernels using NVCC.
///
/// This function:
/// 1. Checks if CUDA toolkit is installed and nvcc is available
/// 2. Compiles cuda_kernels.cu to a static library
/// 3. Links the library into the Rust crate
///
/// If CUDA is not available, this is a no-op (the FFI bindings will fail
/// at link time with a clear error message).
fn compile_cuda_kernels() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let out_path = PathBuf::from(&out_dir);

    // Path to the CUDA kernel source file
    let cuda_source = "src/adapters/gpu/cuda_kernels.cu";

    // Output library path
    let lib_path = out_path.join("libcuda_kernels.a");

    // Try to find nvcc compiler
    let nvcc_path = find_nvcc();

    if let Some(nvcc) = nvcc_path {
        println!("cargo:warning=Compiling CUDA kernels with: {}", nvcc.display());

        // CUDA compilation flags
        let nvcc_args = vec![
            // Input source file
            cuda_source.to_string(),
            // Output library
            format!("-o{}", lib_path.display()),
            // Compilation flags
            "-shared".to_string(),          // Create shared library
            "-Xcompiler=-fPIC".to_string(), // Position-independent code
            "-O3".to_string(),              // Optimization level 3
            // Architecture support (adjust for your target GPUs)
            "--gencode=arch=compute_70,code=sm_70".to_string(), // V100, Titan V
            "--gencode=arch=compute_80,code=sm_80".to_string(), // A100
            "--gencode=arch=compute_86,code=sm_86".to_string(), // RTX 30xx
            "--gencode=arch=compute_89,code=sm_89".to_string(), // RTX 40xx
            // Enable RNG support
            "-lcurand".to_string(),
        ];

        // Attempt compilation
        match std::process::Command::new(&nvcc).args(&nvcc_args).output() {
            Ok(output) => {
                if output.status.success() {
                    println!("cargo:rustc-link-search=native={}", out_dir);
                    println!("cargo:rustc-link-lib=static=cuda_kernels");
                    println!("cargo:rustc-link-lib=cuda");
                    println!("cargo:rustc-link-lib=curand");
                    println!("cargo:warning=CUDA kernels compiled successfully");
                } else {
                    eprintln!("CUDA compilation failed:");
                    eprintln!("stdout: {}", String::from_utf8_lossy(&output.stdout));
                    eprintln!("stderr: {}", String::from_utf8_lossy(&output.stderr));
                    println!(
                        "cargo:warning=CUDA kernel compilation failed. \
                        See build output above. FFI linking will fail."
                    );
                }
            }
            Err(e) => {
                println!(
                    "cargo:warning=Failed to execute nvcc ({}). \
                    CUDA kernels will not be compiled. \
                    Ensure CUDA toolkit is installed and nvcc is in PATH.",
                    e
                );
            }
        }
    } else {
        println!(
            "cargo:warning=CUDA toolkit (nvcc) not found. \
            Install CUDA toolkit to enable GPU kernel compilation. \
            For now, FFI stubs will be used."
        );
    }
}

/// Attempt to locate the NVCC compiler.
///
/// Searches in common CUDA installation paths and checks if nvcc is in PATH.
fn find_nvcc() -> Option<PathBuf> {
    // First, check if nvcc is in PATH
    if let Ok(_) = which::which("nvcc") {
        return Some(PathBuf::from("nvcc"));
    }

    // Search in common CUDA installation paths
    let cuda_paths = vec![
        "/usr/local/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin\\nvcc.exe",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin\\nvcc.exe",
    ];

    for path_str in cuda_paths {
        let path = PathBuf::from(path_str);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

// Helper module: which
// Provides which::which() to locate executables in PATH
// For a more robust solution, you would use the `which` crate:
//   Cargo.toml: which = "4.4"
//
// For now, we implement a minimal fallback:
#[allow(dead_code)]
mod which {
    use std::path::PathBuf;

    pub fn which(cmd: &str) -> Result<PathBuf, std::io::Error> {
        // Try to find command in PATH
        if let Ok(path_var) = std::env::var("PATH") {
            for path_str in path_var.split(':') {
                let mut cmd_path = PathBuf::from(path_str);
                cmd_path.push(cmd);
                if cmd_path.exists() {
                    return Ok(cmd_path);
                }
            }
        }

        // Not found
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Command '{}' not found in PATH", cmd),
        ))
    }
}

/*
 * Build Configuration Notes
 *
 * This build script:
 * 1. Detects if CUDA is installed (checks for nvcc compiler)
 * 2. If available, compiles cuda_kernels.cu to a static library
 * 3. Links the library and CUDA runtime libraries
 * 4. If CUDA is not available, warns the user but does not fail
 *
 * CUDA Toolkit Installation:
 * - Linux:   https://developer.nvidia.com/cuda-downloads
 * - macOS:   CUDA support is deprecated; use OpenCL or Metal
 * - Windows: https://developer.nvidia.com/cuda-downloads
 *
 * Environment Variables:
 * - CUDA_PATH: Manually specify CUDA toolkit location
 * - CC/CXX:    Compiler to use (important for NVCC compatibility)
 *
 * Troubleshooting:
 * If compilation fails, try:
 * 1. Verify CUDA toolkit is installed:    nvcc --version
 * 2. Check cuRAND is available:           ldconfig -p | grep curand
 * 3. Export CUDA path:                    export PATH=/usr/local/cuda/bin:$PATH
 * 4. Clean and rebuild:                   cargo clean && cargo build --features cuda
 */
