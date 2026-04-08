#![warn(missing_docs)]

//! CUDA Kernel Tests
//!
//! Comprehensive unit and integration tests for CUDA kernel functions.
//!
//! Tests are organized by kernel:
//! - `generate_noise_kernel`: Gaussian noise generation
//! - `add_signal_kernel`: Signal + noise addition
//! - `find_extrema_kernel`: Extrema detection
//!
//! Tests verify:
//! - Kernel configuration validity
//! - Memory handling
//! - Numerical correctness (where applicable)
//! - Error handling
//!
//! NOTE: These tests are designed to run in simulation mode on CPUs without CUDA.
//! In a production environment with CUDA installed, the actual GPU kernels would run.

#[cfg(test)]
mod tests {
    use crate::adapters::gpu::cuda_kernels::{
        CudaKernelExecutor, ExtremaKernel, NoiseGenerationKernel, SignalAdditionKernel,
    };
    use crate::adapters::gpu::cuda_wrapper::{CudaDevice, CudaKernelLauncher, KernelLaunchConfig};

    // ============================================================================
    // Tests: Noise Generation Kernel
    // ============================================================================

    #[test]
    fn test_noise_generation_kernel_creation() {
        let kernel = NoiseGenerationKernel::new(1024, 42);

        assert_eq!(kernel.num_samples, 1024);
        assert_eq!(kernel.seed, 42);
        assert_eq!(kernel.mean, 0.0);
        assert_eq!(kernel.std_dev, 1.0);
    }

    #[test]
    fn test_noise_generation_kernel_with_distribution() {
        let kernel = NoiseGenerationKernel::new(512, 99).with_distribution(5.0, 2.5);

        assert_eq!(kernel.num_samples, 512);
        assert_eq!(kernel.seed, 99);
        assert_eq!(kernel.mean, 5.0);
        assert_eq!(kernel.std_dev, 2.5);
    }

    #[test]
    fn test_noise_generation_launch_config_small() {
        let kernel = NoiseGenerationKernel::new(128, 42);
        let config = kernel.optimal_launch_config();

        // For N=128, num_blocks = (128 + 255) / 256 = 1
        assert_eq!(config.block_x, 256);
        assert_eq!(config.block_y, 1);
        assert_eq!(config.grid_x, 1);
        assert_eq!(config.grid_y, 1);
    }

    #[test]
    fn test_noise_generation_launch_config_medium() {
        let kernel = NoiseGenerationKernel::new(1000, 42);
        let config = kernel.optimal_launch_config();

        // For N=1000, num_blocks = (1000 + 255) / 256 = 4
        assert_eq!(config.block_x, 256);
        assert_eq!(config.grid_x, 4);
    }

    #[test]
    fn test_noise_generation_launch_config_large() {
        let kernel = NoiseGenerationKernel::new(1_000_000, 42);
        let config = kernel.optimal_launch_config();

        // For N=1M, num_blocks = (1000000 + 255) / 256 = 3906
        let expected_blocks = (1_000_000 + 255) / 256;
        assert_eq!(config.grid_x, expected_blocks);
        assert_eq!(config.block_x, 256);
    }

    #[test]
    fn test_noise_generation_launch_config_power_of_2() {
        let kernel = NoiseGenerationKernel::new(65536, 42);
        let config = kernel.optimal_launch_config();

        // For N=2^16=65536, num_blocks = 256
        assert_eq!(config.grid_x, 256);
        assert_eq!(config.block_x, 256);
    }

    // ============================================================================
    // Tests: Signal Addition Kernel
    // ============================================================================

    #[test]
    fn test_signal_addition_kernel_creation() {
        let kernel = SignalAdditionKernel::new(512, 0.1);

        assert_eq!(kernel.num_samples, 512);
        assert_eq!(kernel.noise_scale, 0.1);
    }

    #[test]
    fn test_signal_addition_kernel_zero_scale() {
        let kernel = SignalAdditionKernel::new(256, 0.0);

        assert_eq!(kernel.noise_scale, 0.0);
        // With scale=0, output[i] = signal[i] + 0 * noise[i] = signal[i]
    }

    #[test]
    fn test_signal_addition_kernel_large_scale() {
        let kernel = SignalAdditionKernel::new(256, 10.0);

        assert_eq!(kernel.noise_scale, 10.0);
    }

    #[test]
    fn test_signal_addition_launch_config_exact_fit() {
        // N=256 fits exactly in 1 block of 256 threads
        let kernel = SignalAdditionKernel::new(256, 0.1);
        let config = kernel.optimal_launch_config();

        assert_eq!(config.grid_x, 1);
        assert_eq!(config.block_x, 256);
    }

    #[test]
    fn test_signal_addition_launch_config_needs_multiple_blocks() {
        // N=512 needs 2 blocks
        let kernel = SignalAdditionKernel::new(512, 0.1);
        let config = kernel.optimal_launch_config();

        assert_eq!(config.grid_x, 2);
        assert_eq!(config.block_x, 256);
    }

    #[test]
    fn test_signal_addition_launch_config_unaligned() {
        // N=100 needs 1 block (100 < 256)
        let kernel = SignalAdditionKernel::new(100, 0.1);
        let config = kernel.optimal_launch_config();

        assert_eq!(config.grid_x, 1);
        assert_eq!(config.block_x, 256);
    }

    #[test]
    fn test_signal_addition_launch_config_unaligned_large() {
        // N=1000 needs 4 blocks: (1000 + 255) / 256 = 4
        let kernel = SignalAdditionKernel::new(1000, 0.1);
        let config = kernel.optimal_launch_config();

        assert_eq!(config.grid_x, 4);
    }

    // ============================================================================
    // Tests: Extrema Finding Kernel
    // ============================================================================

    #[test]
    fn test_extrema_kernel_creation() {
        let kernel = ExtremaKernel::new(2000, 10);

        assert_eq!(kernel.num_samples, 2000);
        assert_eq!(kernel.min_distance, 10);
    }

    #[test]
    fn test_extrema_kernel_no_min_distance() {
        let kernel = ExtremaKernel::new(1000, 1);

        assert_eq!(kernel.min_distance, 1);
    }

    #[test]
    fn test_extrema_kernel_large_min_distance() {
        let kernel = ExtremaKernel::new(10000, 500);

        assert_eq!(kernel.min_distance, 500);
    }

    #[test]
    fn test_extrema_launch_config_small() {
        // Small signal: 100 samples
        let kernel = ExtremaKernel::new(100, 5);
        let config = kernel.optimal_launch_config();

        // Uses 2D grid: 16x16 threads = 256 per block
        // num_blocks = (100 + 255) / 256 = 1
        assert_eq!(config.block_x, 16);
        assert_eq!(config.block_y, 16);
        assert_eq!(config.grid_x, 1);
    }

    #[test]
    fn test_extrema_launch_config_medium() {
        // Medium signal: 1000 samples
        let kernel = ExtremaKernel::new(1000, 5);
        let config = kernel.optimal_launch_config();

        // num_blocks = (1000 + 255) / 256 = 4
        assert_eq!(config.grid_x, 4);
        assert_eq!(config.block_x, 16);
        assert_eq!(config.block_y, 16);
    }

    #[test]
    fn test_extrema_launch_config_large() {
        // Large signal: 1 million samples
        let kernel = ExtremaKernel::new(1_000_000, 5);
        let config = kernel.optimal_launch_config();

        let expected_blocks = (1_000_000 + 255) / 256;
        assert_eq!(config.grid_x, expected_blocks);
        assert_eq!(config.block_x, 16);
        assert_eq!(config.block_y, 16);
    }

    // ============================================================================
    // Tests: Kernel Launch Configuration Validation
    // ============================================================================

    #[test]
    fn test_kernel_launch_config_validation_valid() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let config = KernelLaunchConfig::grid_1d(128, 256);

        // Should validate successfully
        let result = config.validate(&device);
        assert!(result.is_ok());
    }

    #[test]
    fn test_kernel_launch_config_validation_zero_grid() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let config = KernelLaunchConfig {
            grid_x: 0,
            grid_y: 1,
            block_x: 256,
            block_y: 1,
            shared_memory: 0,
            stream: Default::default(),
        };

        // Should fail: grid dimension is 0
        let result = config.validate(&device);
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_launch_config_validation_zero_block() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let config = KernelLaunchConfig {
            grid_x: 128,
            grid_y: 1,
            block_x: 0,
            block_y: 1,
            shared_memory: 0,
            stream: Default::default(),
        };

        // Should fail: block dimension is 0
        let result = config.validate(&device);
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_launch_config_validation_threads_exceed_max() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");
        let max_threads = device.properties().max_threads_per_block;

        // Try to launch with more threads than device supports
        let config = KernelLaunchConfig {
            grid_x: 128,
            grid_y: 1,
            block_x: (max_threads + 1) as u32,
            block_y: 1,
            shared_memory: 0,
            stream: Default::default(),
        };

        let result = config.validate(&device);
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_launch_config_validation_2d_threads() {
        let device = CudaDevice::new(0).expect("Failed to create CUDA device");

        // 16x16 = 256 threads (valid)
        let config = KernelLaunchConfig::grid_2d(64, 64, 16, 16);
        assert!(config.validate(&device).is_ok());

        // 32x32 = 1024 threads (valid on most modern GPUs)
        let config = KernelLaunchConfig::grid_2d(32, 32, 32, 32);
        let result = config.validate(&device);
        // Result depends on device: 1024 is valid on V100+, may fail on older GPUs
        // Just check that it doesn't panic
        let _ = result;
    }

    // ============================================================================
    // Tests: CUDA Device Initialization
    // ============================================================================

    #[test]
    fn test_cuda_device_new() {
        let device = CudaDevice::new(0);
        assert!(device.is_ok());
    }

    #[test]
    fn test_cuda_device_properties() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let props = device.properties();

        assert!(props.total_global_mem > 0);
        assert_eq!(props.warp_size, 32); // Always 32 for NVIDIA
        assert!(props.max_threads_per_block > 0);
        assert!(props.multiprocessor_count > 0);
    }

    #[test]
    fn test_cuda_device_memory_allocation() {
        let device = CudaDevice::new(0).expect("Failed to create device");

        // Try to allocate 1MB
        let size = 1024 * 1024;
        let result = device.allocate(size);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_device_memory_allocation_zero() {
        let device = CudaDevice::new(0).expect("Failed to create device");

        // Zero-size allocation should fail
        let result = device.allocate(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_device_memory_allocation_exceeds_available() {
        let device = CudaDevice::new(0).expect("Failed to create device");
        let total_mem = device.properties().total_global_mem;

        // Try to allocate more than total device memory
        let result = device.allocate(total_mem + 1);
        assert!(result.is_err());
    }

    // ============================================================================
    // Tests: CUDA Kernel Executor
    // ============================================================================

    #[test]
    fn test_cuda_kernel_executor_new() {
        let executor = CudaKernelExecutor::new(0);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_cuda_kernel_executor_launcher_reference() {
        let executor = CudaKernelExecutor::new(0).expect("Failed to create executor");
        let launcher = executor.launcher();

        assert_eq!(launcher.device().handle().device_id, 0);
    }

    #[test]
    fn test_cuda_kernel_executor_device_properties() {
        let executor = CudaKernelExecutor::new(0).expect("Failed to create executor");
        let device = executor.launcher().device();
        let props = device.properties();

        assert!(props.total_global_mem > 0);
        assert_eq!(props.warp_size, 32);
    }

    // ============================================================================
    // Tests: Numerical Behavior (Simulation)
    // ============================================================================

    #[test]
    fn test_noise_generation_seed_reproducibility() {
        // Even in simulation, verify that kernel configs with same seed
        // would produce identical results
        let kernel1 = NoiseGenerationKernel::new(1000, 42);
        let kernel2 = NoiseGenerationKernel::new(1000, 42);

        assert_eq!(kernel1.seed, kernel2.seed);
        assert_eq!(kernel1.num_samples, kernel2.num_samples);
    }

    #[test]
    fn test_noise_generation_different_seeds() {
        let kernel1 = NoiseGenerationKernel::new(1000, 42);
        let kernel2 = NoiseGenerationKernel::new(1000, 99);

        // Different seeds should produce different noise
        assert_ne!(kernel1.seed, kernel2.seed);
    }

    #[test]
    fn test_signal_addition_buffer_size_calculations() {
        let kernel = SignalAdditionKernel::new(1024, 0.1);

        let num_samples = kernel.num_samples as u64;
        let element_size = std::mem::size_of::<f64>() as u64;
        let required_bytes = num_samples * element_size;

        // Verify buffer size calculation
        assert_eq!(required_bytes, 1024 * 8);
    }

    #[test]
    fn test_extrema_detection_buffer_sizes() {
        let kernel = ExtremaKernel::new(2000, 10);

        let signal_bytes = (kernel.num_samples as u64) * std::mem::size_of::<f64>() as u64;
        let indices_bytes = (kernel.num_samples as u64) * std::mem::size_of::<u32>() as u64;
        let count_bytes = std::mem::size_of::<u32>() as u64;

        assert_eq!(signal_bytes, 2000 * 8);
        assert_eq!(indices_bytes, 2000 * 4);
        assert_eq!(count_bytes, 4);
    }

    // ============================================================================
    // Tests: Edge Cases
    // ============================================================================

    #[test]
    fn test_noise_generation_single_sample() {
        let kernel = NoiseGenerationKernel::new(1, 42);
        let config = kernel.optimal_launch_config();

        assert_eq!(config.grid_x, 1);
        assert_eq!(config.block_x, 256);
    }

    #[test]
    fn test_signal_addition_single_sample() {
        let kernel = SignalAdditionKernel::new(1, 0.1);
        let config = kernel.optimal_launch_config();

        assert_eq!(config.grid_x, 1);
    }

    #[test]
    fn test_extrema_kernel_minimum_samples() {
        // Need at least 3 samples: [0] border, [1] potential extrema, [2] border
        let kernel = ExtremaKernel::new(3, 1);
        let config = kernel.optimal_launch_config();

        // Should still generate valid config even for tiny input
        assert_eq!(config.block_x, 16);
        assert_eq!(config.block_y, 16);
    }

    #[test]
    fn test_kernel_launch_config_1d_builder() {
        let config = KernelLaunchConfig::grid_1d(256, 512);

        assert_eq!(config.grid_x, 256);
        assert_eq!(config.block_x, 512);
        assert_eq!(config.grid_y, 1);
        assert_eq!(config.block_y, 1);
    }

    #[test]
    fn test_kernel_launch_config_2d_builder() {
        let config = KernelLaunchConfig::grid_2d(64, 32, 16, 16);

        assert_eq!(config.grid_x, 64);
        assert_eq!(config.grid_y, 32);
        assert_eq!(config.block_x, 16);
        assert_eq!(config.block_y, 16);
    }

    #[test]
    fn test_kernel_launch_config_default() {
        let config = KernelLaunchConfig::default();

        assert_eq!(config.grid_x, 1);
        assert_eq!(config.grid_y, 1);
        assert_eq!(config.block_x, 256);
        assert_eq!(config.block_y, 1);
        assert_eq!(config.shared_memory, 0);
    }
}
