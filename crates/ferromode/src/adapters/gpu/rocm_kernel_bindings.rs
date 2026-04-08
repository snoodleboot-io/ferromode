#![warn(missing_docs)]

//! FFI bindings to HIP kernel functions for AMD ROCm backend.
//!
//! This module provides low-level C FFI declarations for the HIP kernels
//! defined in `rocm_kernels.hip`. All functions are declared as `extern "C"`
//! and wrapped by safe Rust functions in `rocm_wrapper.rs`.
//!
//! HIP (Heterogeneous-compute Interface for Portability) kernels are functionally
//! equivalent to CUDA kernels but compiled for AMD GPUs using hipcc compiler.
//!
//! # Safety
//!
//! All raw FFI calls are unsafe and must be wrapped in safe Rust functions
//! with proper validation and error handling. This module is internal and
//! should not be used directly by end users.

use std::os::raw::c_double;

/// Raw error code from HIP API.
///
/// Corresponds to `hipError_t` enum in HIP runtime.
/// 0 = hipSuccess, non-zero = error
pub type HipErrorCode = u32;

/// Result of a HIP kernel launch operation.
pub type HipFFIResult = HipErrorCode;

/// HIP error code constants (matching hip_runtime.h)
pub mod hip_errors {
    //! HIP error code constants

    /// No error
    pub const HIP_SUCCESS: u32 = 0;
    /// Invalid device ordinal
    pub const HIP_ERROR_INVALID_DEVICE: u32 = 1;
    /// Device not initialized
    pub const HIP_ERROR_NOT_INITIALIZED: u32 = 3;
    /// Memory allocation failed
    pub const HIP_ERROR_MEMORY_ALLOCATION: u32 = 2;
    /// Memory operation failed / Launch failed
    pub const HIP_ERROR_LAUNCH_FAILED: u32 = 719;
    /// Invalid value parameter
    pub const HIP_ERROR_INVALID_VALUE: u32 = 1;
}

/// FFI binding to HIP kernel: generate_noise
///
/// Generates Gaussian (normal distribution) noise on GPU device.
/// Uses hiprand for parallel random number generation with Box-Muller transform.
///
/// # Safety
///
/// - `output` must be a valid GPU device pointer allocated via hipMalloc
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
/// HIP error code: 0 (hipSuccess) on success, non-zero on error
#[link(name = "amdhip64")]
extern "C" {
    /// Launch HIP generate_noise kernel
    pub fn launch_generate_noise_hip(
        seed: u64,
        scale: c_double,
        output: *mut c_double,
        size: u32,
    ) -> HipFFIResult;
}

/// FFI binding to HIP kernel: add_signal
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
/// HIP error code: 0 on success, non-zero on error
extern "C" {
    /// Launch HIP add_signal kernel
    pub fn launch_add_signal_hip(
        signal: *const c_double,
        noise: *const c_double,
        noise_scale: c_double,
        output: *mut c_double,
        size: u32,
    ) -> HipFFIResult;
}

/// FFI binding to HIP kernel: find_extrema
///
/// Finds local extrema (maxima and minima) in signal for spline basis.
///
/// The kernel detects points where:
/// - Maximum: signal[i-1] < signal[i] > signal[i+1]
/// - Minimum: signal[i-1] > signal[i] < signal[i+1]
///
/// Boundary points (first and last element) are never extrema.
/// Uses atomic operations for thread-safe result collection.
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
/// HIP error code: 0 on success, non-zero on error
///
/// # Post-Conditions
///
/// After successful execution and synchronization:
/// - `max_count` contains number of maxima found (0 to size)
/// - `min_count` contains number of minima found (0 to size)
/// - First N elements of `max_indices` contain indices of maxima
/// - First M elements of `min_indices` contain indices of minima
extern "C" {
    /// Launch HIP find_extrema kernel
    pub fn launch_find_extrema_hip(
        signal: *const c_double,
        max_indices: *mut u32,
        min_indices: *mut u32,
        max_count: *mut u32,
        min_count: *mut u32,
        size: u32,
    ) -> HipFFIResult;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_error_constants_defined() {
        // Verify error constants exist
        let _ = hip_errors::HIP_SUCCESS;
        let _ = hip_errors::HIP_ERROR_INVALID_DEVICE;
        let _ = hip_errors::HIP_ERROR_MEMORY_ALLOCATION;
    }

    #[test]
    fn test_hip_ffi_types() {
        // Verify type definitions
        let _: HipErrorCode = 0;
        let _: HipFFIResult = 0;
    }
}
