#![warn(missing_docs)]

//! CUDA kernel implementations for GPU-accelerated EMD algorithms.
//!
//! This module provides high-performance CUDA kernels for:
//! - Gaussian noise generation (parallel RNG)
//! - Signal + noise addition (elementwise operations)
//! - Extrema finding (local min/max detection)
//!
//! # Compilation Note
//!
//! In a production build, the actual kernel code would be in `cuda_kernels.cu`
//! and compiled via NVCC. For now, these are Rust wrappers that will integrate
//! with actual CUDA kernel binaries via FFI.

use crate::adapters::gpu::cuda_wrapper::{
    CudaDevice, CudaError, CudaKernelLauncher, CudaMemoryHandle, CudaResult, KernelLaunchConfig,
    KernelLaunchResult,
};
use std::time::{Duration, Instant};

/// Kernel parameters for noise generation.
#[derive(Debug, Clone)]
pub struct NoiseGenerationKernel {
    /// Number of samples to generate
    pub num_samples: u32,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Mean of Gaussian distribution
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
}

impl NoiseGenerationKernel {
    /// Create a noise generation kernel config.
    pub fn new(num_samples: u32, seed: u64) -> Self {
        NoiseGenerationKernel { num_samples, seed, mean: 0.0, std_dev: 1.0 }
    }

    /// Set mean and std_dev for the Gaussian distribution.
    pub fn with_distribution(mut self, mean: f32, std_dev: f32) -> Self {
        self.mean = mean;
        self.std_dev = std_dev;
        self
    }

    /// Calculate optimal grid/block configuration for this kernel.
    pub fn optimal_launch_config(&self) -> KernelLaunchConfig {
        // Thread block size: 256 threads (good balance on most GPUs)
        let block_size = 256u32;
        // Number of blocks needed to cover all samples
        let num_blocks = (self.num_samples + block_size - 1) / block_size;

        KernelLaunchConfig::grid_1d(num_blocks, block_size)
    }
}

/// Kernel parameters for signal + noise addition.
#[derive(Debug, Clone)]
pub struct SignalAdditionKernel {
    /// Number of samples
    pub num_samples: u32,
    /// Amplitude scaling factor for noise
    pub noise_scale: f32,
}

impl SignalAdditionKernel {
    /// Create a signal addition kernel config.
    pub fn new(num_samples: u32, noise_scale: f32) -> Self {
        SignalAdditionKernel { num_samples, noise_scale }
    }

    /// Calculate optimal launch configuration.
    pub fn optimal_launch_config(&self) -> KernelLaunchConfig {
        let block_size = 256u32;
        let num_blocks = (self.num_samples + block_size - 1) / block_size;
        KernelLaunchConfig::grid_1d(num_blocks, block_size)
    }
}

/// Kernel parameters for extrema finding.
#[derive(Debug, Clone)]
pub struct ExtremaKernel {
    /// Number of samples in signal
    pub num_samples: u32,
    /// Minimum distance between extrema (for spline basis)
    pub min_distance: u32,
}

impl ExtremaKernel {
    /// Create an extrema finding kernel config.
    pub fn new(num_samples: u32, min_distance: u32) -> Self {
        ExtremaKernel { num_samples, min_distance }
    }

    /// Calculate optimal launch configuration (2D grid for efficiency).
    pub fn optimal_launch_config(&self) -> KernelLaunchConfig {
        // Use 2D grid for better load distribution
        let block_dim = 16u32; // 16x16 = 256 threads per block
        let grid_dim = (self.num_samples + (block_dim * block_dim) - 1) / (block_dim * block_dim);

        KernelLaunchConfig::grid_2d(grid_dim, 1, block_dim, block_dim)
    }
}

/// High-level interface for CUDA kernel execution.
pub struct CudaKernelExecutor {
    launcher: CudaKernelLauncher,
}

impl CudaKernelExecutor {
    /// Create a new kernel executor for the specified device.
    pub fn new(device_id: u32) -> CudaResult<Self> {
        let launcher = CudaKernelLauncher::new(device_id)?;
        Ok(CudaKernelExecutor { launcher })
    }

    /// Get reference to the kernel launcher.
    pub fn launcher(&self) -> &CudaKernelLauncher {
        &self.launcher
    }

    /// Execute noise generation kernel.
    ///
    /// Generates independent Gaussian noise on the GPU device and stores
    /// the result in `output_device_mem`.
    ///
    /// # Arguments
    ///
    /// * `config` - Kernel configuration (samples, seed, distribution)
    /// * `output_device_mem` - Preallocated device memory for output
    ///
    /// # Returns
    ///
    /// `KernelLaunchResult` with execution status and timing.
    pub fn execute_generate_noise(
        &self,
        config: &NoiseGenerationKernel,
        output_device_mem: &mut CudaMemoryHandle,
    ) -> CudaResult<KernelLaunchResult> {
        // Validate output buffer size
        let required_bytes = (config.num_samples as u64) * std::mem::size_of::<f64>() as u64;
        if output_device_mem.size < required_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: required_bytes,
                available_memory: output_device_mem.size,
            });
        }

        let launch_config = config.optimal_launch_config();
        self.launcher.launch_generate_noise(
            config.num_samples,
            config.seed,
            output_device_mem,
            &launch_config,
        )
    }

    /// Execute signal + noise addition kernel.
    ///
    /// Adds noise to signal: output[i] = signal[i] + noise_scale * noise[i]
    ///
    /// # Arguments
    ///
    /// * `config` - Kernel configuration
    /// * `signal_device_mem` - Device memory containing input signal
    /// * `noise_device_mem` - Device memory containing noise samples
    /// * `output_device_mem` - Device memory for result
    pub fn execute_add_signal(
        &self,
        config: &SignalAdditionKernel,
        signal_device_mem: &CudaMemoryHandle,
        noise_device_mem: &CudaMemoryHandle,
        output_device_mem: &mut CudaMemoryHandle,
    ) -> CudaResult<KernelLaunchResult> {
        let required_bytes = (config.num_samples as u64) * std::mem::size_of::<f64>() as u64;

        // Validate all buffers
        if signal_device_mem.size < required_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: required_bytes,
                available_memory: signal_device_mem.size,
            });
        }
        if noise_device_mem.size < required_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: required_bytes,
                available_memory: noise_device_mem.size,
            });
        }
        if output_device_mem.size < required_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: required_bytes,
                available_memory: output_device_mem.size,
            });
        }

        let launch_config = config.optimal_launch_config();
        self.launcher.launch_add_signal(
            config.num_samples,
            signal_device_mem,
            noise_device_mem,
            output_device_mem,
            &launch_config,
        )
    }

    /// Execute extrema finding kernel.
    ///
    /// Finds local maxima and minima in signal for spline basis computation.
    /// Results are stored as indices in the output arrays.
    ///
    /// # Arguments
    ///
    /// * `config` - Kernel configuration with sample count and min_distance
    /// * `signal_device_mem` - Device memory containing the signal
    /// * `max_indices_device_mem` - Output buffer for maxima indices
    /// * `min_indices_device_mem` - Output buffer for minima indices
    /// * `max_count_device_mem` - Output scalar: number of maxima found
    /// * `min_count_device_mem` - Output scalar: number of minima found
    pub fn execute_find_extrema(
        &self,
        config: &ExtremaKernel,
        signal_device_mem: &CudaMemoryHandle,
        max_indices_device_mem: &mut CudaMemoryHandle,
        min_indices_device_mem: &mut CudaMemoryHandle,
        max_count_device_mem: &mut CudaMemoryHandle,
        min_count_device_mem: &mut CudaMemoryHandle,
    ) -> CudaResult<KernelLaunchResult> {
        let signal_bytes = (config.num_samples as u64) * std::mem::size_of::<f64>() as u64;
        let count_bytes = std::mem::size_of::<u32>() as u64;
        let indices_bytes = (config.num_samples as u64) * std::mem::size_of::<u32>() as u64;

        // Validate all buffers
        if signal_device_mem.size < signal_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: signal_bytes,
                available_memory: signal_device_mem.size,
            });
        }
        if max_indices_device_mem.size < indices_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: indices_bytes,
                available_memory: max_indices_device_mem.size,
            });
        }
        if min_indices_device_mem.size < indices_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: indices_bytes,
                available_memory: min_indices_device_mem.size,
            });
        }
        if max_count_device_mem.size < count_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: count_bytes,
                available_memory: max_count_device_mem.size,
            });
        }
        if min_count_device_mem.size < count_bytes {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: count_bytes,
                available_memory: min_count_device_mem.size,
            });
        }

        let launch_config = config.optimal_launch_config();
        self.launcher.launch_find_extrema(
            config.num_samples,
            signal_device_mem,
            max_indices_device_mem,
            min_indices_device_mem,
            max_count_device_mem,
            min_count_device_mem,
            &launch_config,
        )
    }

    /// Synchronize with the GPU device.
    pub fn synchronize(&self) -> CudaResult<Duration> {
        self.launcher.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_generation_kernel_new() {
        let kernel = NoiseGenerationKernel::new(1024, 42);
        assert_eq!(kernel.num_samples, 1024);
        assert_eq!(kernel.seed, 42);
    }

    #[test]
    fn test_noise_generation_optimal_config() {
        let kernel = NoiseGenerationKernel::new(1000, 42);
        let config = kernel.optimal_launch_config();
        assert_eq!(config.block_x, 256);
        assert!(config.grid_x > 0);
    }

    #[test]
    fn test_signal_addition_kernel_new() {
        let kernel = SignalAdditionKernel::new(1024, 0.1);
        assert_eq!(kernel.num_samples, 1024);
        assert_eq!(kernel.noise_scale, 0.1);
    }

    #[test]
    fn test_signal_addition_optimal_config() {
        let kernel = SignalAdditionKernel::new(512, 0.1);
        let config = kernel.optimal_launch_config();
        assert_eq!(config.block_x, 256);
        assert_eq!(config.grid_x, 2);
    }

    #[test]
    fn test_extrema_kernel_new() {
        let kernel = ExtremaKernel::new(2000, 10);
        assert_eq!(kernel.num_samples, 2000);
        assert_eq!(kernel.min_distance, 10);
    }

    #[test]
    fn test_extrema_optimal_config() {
        let kernel = ExtremaKernel::new(1024, 5);
        let config = kernel.optimal_launch_config();
        assert_eq!(config.block_x, 16);
        assert_eq!(config.block_y, 16);
    }

    #[test]
    fn test_cuda_kernel_executor_new() {
        let executor = CudaKernelExecutor::new(0);
        assert!(executor.is_ok());
    }
}
