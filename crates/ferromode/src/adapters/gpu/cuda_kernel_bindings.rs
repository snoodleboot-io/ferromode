#![warn(missing_docs)]

//! FFI bindings to CUDA kernel functions.
//!
//! This module provides low-level C FFI declarations for the CUDA kernels
//! defined in `cuda_kernels.cu`. All functions are declared as `extern "C"`
//! and wrapped by safe Rust functions in `cuda_wrapper.rs`.
//!
//! # Safety
//!
//! All raw FFI calls are unsafe and must be wrapped in safe Rust functions
//! with proper validation and error handling. This module is internal and
//! should not be used directly by end users.

use std::os::raw::c_double;

/// Raw error code from CUDA API.
///
/// Corresponds to `cudaError_t` enum in CUDA runtime.
/// 0 = cudaSuccess, non-zero = error
pub type CudaErrorCode = u32;

/// Result of a CUDA kernel launch operation.
pub type CudaFFIResult = CudaErrorCode;

/// CUDA error code constants (matching cuda_runtime.h)
pub mod cuda_errors {
    //! CUDA error code constants

    /// No error
    pub const CUDA_SUCCESS: u32 = 0;
    /// Invalid device ordinal
    pub const CUDA_ERROR_INVALID_DEVICE: u32 = 1;
    /// Device not initialized
    pub const CUDA_ERROR_NOT_INITIALIZED: u32 = 3;
    /// Deinitialization error
    pub const CUDA_ERROR_DEINITIALIZED: u32 = 4;
    /// Memory allocation failed
    pub const CUDA_ERROR_MEMORY_ALLOCATION: u32 = 2;
    /// Memory operation failed
    pub const CUDA_ERROR_LAUNCH_FAILED: u32 = 719;
    /// Invalid value parameter
    pub const CUDA_ERROR_INVALID_VALUE: u32 = 1;
}

/// FFI binding to CUDA kernel: generate_noise
///
/// Generates Gaussian (normal distribution) noise on GPU device.
///
/// # Safety
///
/// - `output` must be a valid GPU device pointer allocated via cudaMalloc
/// - `output` must have at least `size * sizeof(f64)` bytes allocated
/// - `size` must match the allocated size
/// - GPU memory must remain valid until kernel completes
/// - Caller must synchronize device before reading output
///
/// # Arguments
///
/// * `seed` - RNG seed for reproducibility
/// * `scale` - Standard deviation of Gaussian noise
/// * `output` - GPU device pointer to output array (f64 elements)
/// * `size` - Number of noise samples to generate
///
/// # Returns
///
/// CUDA error code: 0 (cudaSuccess) on success, non-zero on error
#[link(name = "cuda")]
#[link(name = "curand")]
extern "C" {
    pub fn launch_generate_noise(
        seed: u64,
        scale: c_double,
        output: *mut c_double,
        size: u32,
    ) -> CudaFFIResult;
}

/// FFI binding to CUDA kernel: add_signal
///
/// Performs element-wise operation: output[i] = signal[i] + scale * noise[i]
///
/// # Safety
///
/// - `signal` must be a valid GPU device pointer to array of size `size`
/// - `noise` must be a valid GPU device pointer to array of size `size`
/// - `output` must be a valid GPU device pointer with size >= `size`
/// - All pointers must point to f64 (double) arrays
/// - GPU memory must remain valid until kernel completes
/// - Caller must synchronize device before reading output
///
/// # Arguments
///
/// * `signal` - GPU pointer to input signal array (f64)
/// * `noise` - GPU pointer to noise array (f64)
/// * `noise_scale` - Scaling factor applied to noise
/// * `output` - GPU pointer to output array (f64)
/// * `size` - Number of elements to process
///
/// # Returns
///
/// CUDA error code: 0 on success, non-zero on error
extern "C" {
    pub fn launch_add_signal(
        signal: *const c_double,
        noise: *const c_double,
        noise_scale: c_double,
        output: *mut c_double,
        size: u32,
    ) -> CudaFFIResult;
}

/// FFI binding to CUDA kernel: find_extrema
///
/// Finds local extrema (maxima and minima) in signal for spline basis.
///
/// The kernel detects points where:
/// - Maximum: signal[i-1] < signal[i] > signal[i+1]
/// - Minimum: signal[i-1] > signal[i] < signal[i+1]
///
/// Boundary points (first and last element) are never extrema.
///
/// # Safety
///
/// - All input pointers must be valid GPU device pointers
/// - Arrays must have sufficient allocated size
/// - `signal` must have at least `size * sizeof(f64)` bytes
/// - `max_indices` must have at least `size * sizeof(u32)` bytes
/// - `min_indices` must have at least `size * sizeof(u32)` bytes
/// - `max_count` must be a u32* with value 0 before kernel launch
/// - `min_count` must be a u32* with value 0 before kernel launch
/// - GPU memory must remain valid until kernel completes
/// - Caller must synchronize device before reading results
///
/// # Arguments
///
/// * `signal` - GPU pointer to input signal (f64 array, length `size`)
/// * `max_indices` - GPU pointer to output maxima indices (u32 array)
/// * `min_indices` - GPU pointer to output minima indices (u32 array)
/// * `max_count` - GPU pointer to u32 counter: number of maxima found
/// * `min_count` - GPU pointer to u32 counter: number of minima found
/// * `size` - Number of samples in signal
///
/// # Returns
///
/// CUDA error code: 0 on success, non-zero on error
///
/// # Post-Conditions
///
/// After successful execution and synchronization:
/// - `max_count` contains the number of maxima found (0 to size-2)
/// - `max_indices[0..max_count-1]` contains indices of maxima
/// - `min_count` contains the number of minima found (0 to size-2)
/// - `min_indices[0..min_count-1]` contains indices of minima
extern "C" {
    pub fn launch_find_extrema(
        signal: *const c_double,
        max_indices: *mut u32,
        min_indices: *mut u32,
        max_count: *mut u32,
        min_count: *mut u32,
        size: u32,
    ) -> CudaFFIResult;
}

/// Trait for types that can be converted to/from CUDA error codes
pub trait CudaErrorConvert {
    /// Convert from CUDA error code
    fn from_cuda_error(code: CudaErrorCode) -> Option<String>;

    /// Convert to description
    fn cuda_error_name(code: CudaErrorCode) -> &'static str;
}

impl CudaErrorConvert for CudaErrorCode {
    fn from_cuda_error(code: CudaErrorCode) -> Option<String> {
        if code == cuda_errors::CUDA_SUCCESS {
            return None; // No error
        }

        let name = match code {
            cuda_errors::CUDA_ERROR_INVALID_DEVICE => "Invalid device ordinal",
            cuda_errors::CUDA_ERROR_NOT_INITIALIZED => "CUDA runtime not initialized",
            cuda_errors::CUDA_ERROR_DEINITIALIZED => "CUDA runtime deinitialized",
            cuda_errors::CUDA_ERROR_MEMORY_ALLOCATION => "Memory allocation failed",
            cuda_errors::CUDA_ERROR_INVALID_VALUE => "Invalid parameter value",
            cuda_errors::CUDA_ERROR_LAUNCH_FAILED => "Kernel launch failed",
            _ => "Unknown CUDA error",
        };

        Some(name.to_string())
    }

    fn cuda_error_name(code: CudaErrorCode) -> &'static str {
        match code {
            cuda_errors::CUDA_SUCCESS => "CUDA_SUCCESS",
            cuda_errors::CUDA_ERROR_INVALID_DEVICE => "CUDA_ERROR_INVALID_DEVICE",
            cuda_errors::CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED",
            cuda_errors::CUDA_ERROR_DEINITIALIZED => "CUDA_ERROR_DEINITIALIZED",
            cuda_errors::CUDA_ERROR_MEMORY_ALLOCATION => "CUDA_ERROR_MEMORY_ALLOCATION",
            cuda_errors::CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE",
            cuda_errors::CUDA_ERROR_LAUNCH_FAILED => "CUDA_ERROR_LAUNCH_FAILED",
            _ => "CUDA_ERROR_UNKNOWN",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_error_convert() {
        let error_str = CudaErrorCode::from_cuda_error(cuda_errors::CUDA_ERROR_MEMORY_ALLOCATION);
        assert_eq!(error_str, Some("Memory allocation failed".to_string()));

        let error_str = CudaErrorCode::from_cuda_error(cuda_errors::CUDA_SUCCESS);
        assert_eq!(error_str, None);
    }

    #[test]
    fn test_cuda_error_name() {
        assert_eq!(CudaErrorCode::cuda_error_name(cuda_errors::CUDA_SUCCESS), "CUDA_SUCCESS");
        assert_eq!(
            CudaErrorCode::cuda_error_name(cuda_errors::CUDA_ERROR_INVALID_DEVICE),
            "CUDA_ERROR_INVALID_DEVICE"
        );
    }

    #[test]
    fn test_cuda_error_constants() {
        assert_eq!(cuda_errors::CUDA_SUCCESS, 0);
        assert!(cuda_errors::CUDA_ERROR_INVALID_DEVICE > 0);
    }
}
