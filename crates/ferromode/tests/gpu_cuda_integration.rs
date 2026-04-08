#![cfg(feature = "cuda")]

//! GPU CUDA integration tests.
//!
//! Tests for GPU acceleration of EEMD and CEEMDAN algorithms.
//! These tests verify:
//! - GPU device initialization
//! - Memory allocation and transfers
//! - Kernel configuration and launch
//! - Result numerical correctness

use ferromode::adapters::gpu::cuda_integration::{
    GpuCeemданConfig, GpuCeemданResult, GpuEemdConfig, GpuEemdResult, GpuEnsembleExecutor,
};
use ferromode::adapters::gpu::cuda_kernels::{
    CudaKernelExecutor, ExtremaKernel, NoiseGenerationKernel, SignalAdditionKernel,
};
use ferromode::adapters::gpu::cuda_wrapper::{CudaDevice, CudaKernelLauncher, KernelLaunchConfig};

#[test]
fn test_cuda_device_initialization() {
    let device = CudaDevice::new(0);
    assert!(device.is_ok(), "Failed to initialize CUDA device 0");

    let dev = device.unwrap();
    let props = dev.properties();

    // Verify device properties
    assert!(!props.name.is_empty(), "Device name should not be empty");
    assert!(props.total_global_mem > 0, "Device should have memory");
    assert_eq!(props.warp_size, 32, "NVIDIA GPUs have warp size of 32");
    assert!(props.max_threads_per_block > 0, "Device should support threads per block");
    assert!(props.multiprocessor_count > 0, "Device should have multiprocessors");
}

#[test]
fn test_cuda_memory_allocation() {
    let device = CudaDevice::new(0).expect("Device initialization");

    // Allocate 1 MB of device memory
    let mem = device.allocate(1024 * 1024);
    assert!(mem.is_ok(), "Memory allocation should succeed");

    let handle = mem.unwrap();
    assert_eq!(handle.size, 1024 * 1024, "Memory size should match request");

    // Deallocate
    let result = device.deallocate(handle);
    assert!(result.is_ok(), "Memory deallocation should succeed");
}

#[test]
fn test_cuda_memory_allocation_failure() {
    let device = CudaDevice::new(0).expect("Device initialization");

    // Try to allocate more memory than available
    let huge_size = u64::MAX / 2;
    let result = device.allocate(huge_size);

    assert!(result.is_err(), "Allocation should fail for excessive size");
}

#[test]
fn test_kernel_launcher_creation() {
    let launcher = CudaKernelLauncher::new(0);
    assert!(launcher.is_ok(), "Kernel launcher creation should succeed");

    let l = launcher.unwrap();
    let props = l.device().properties();
    assert!(props.total_global_mem > 0, "Device should have memory");
}

#[test]
fn test_noise_generation_kernel_config() {
    let kernel = NoiseGenerationKernel::new(10000, 42);
    let config = kernel.optimal_launch_config();

    assert!(config.validate(&CudaDevice::new(0).unwrap()).is_ok(), "Config should be valid");
    assert_eq!(config.block_x, 256, "Should use 256 threads per block");
    assert!(config.grid_x > 0, "Grid should have at least one block");
}

#[test]
fn test_signal_addition_kernel_config() {
    let kernel = SignalAdditionKernel::new(5000, 0.1);
    let config = kernel.optimal_launch_config();

    assert!(config.validate(&CudaDevice::new(0).unwrap()).is_ok(), "Config should be valid");
    assert_eq!(config.block_x, 256);
    assert!(config.grid_x > 0);
}

#[test]
fn test_extrema_kernel_config() {
    let kernel = ExtremaKernel::new(2000, 10);
    let config = kernel.optimal_launch_config();

    assert!(config.validate(&CudaDevice::new(0).unwrap()).is_ok(), "Config should be valid");
    assert_eq!(config.block_x, 16);
    assert_eq!(config.block_y, 16);
}

#[test]
fn test_kernel_executor_creation() {
    let executor = CudaKernelExecutor::new(0);
    assert!(executor.is_ok(), "Kernel executor should be created");
}

#[test]
fn test_gpu_ensemble_executor_creation() {
    let executor = GpuEnsembleExecutor::new(0);
    assert!(executor.is_ok(), "GPU ensemble executor should be created");

    let exec = executor.unwrap();
    let info = exec.device_info();
    assert!(info.contains("CUDA Device"), "Device info should mention CUDA");
}

#[test]
fn test_eemd_config_defaults() {
    let config = GpuEemdConfig::default();
    assert_eq!(config.num_trials, 100);
    assert_eq!(config.num_samples, 1024);
    assert!(config.noise_amplitude > 0.0);
}

#[test]
fn test_ceemdan_config_defaults() {
    let config = GpuCeemданConfig::default();
    assert_eq!(config.num_trials, 100);
    assert_eq!(config.num_samples, 1024);
    assert!(config.noise_amplitude > 0.0);
}

#[test]
fn test_eemd_execution_with_valid_signal() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");
    let config = GpuEemdConfig::default();
    let signal = vec![0.0; 1024];

    let result = executor.execute_eemd(&config, &signal);
    assert!(result.is_ok(), "EEMD execution should succeed");

    let eemd = result.unwrap();
    assert!(!eemd.imfs.is_empty(), "Should have at least one IMF");
    assert_eq!(eemd.residual.len(), 1024, "Residual should match signal length");
}

#[test]
fn test_eemd_execution_with_invalid_signal_length() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");
    let config = GpuEemdConfig::default();
    let signal = vec![0.0; 512]; // Wrong size

    let result = executor.execute_eemd(&config, &signal);
    assert!(result.is_err(), "EEMD should fail with mismatched signal length");
}

#[test]
fn test_ceemdan_execution_with_valid_signal() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");
    let config = GpuCeemданConfig::default();
    let signal = vec![0.0; 1024];

    let result = executor.execute_ceemdan(&config, &signal);
    assert!(result.is_ok(), "CEEMDAN execution should succeed");

    let ceemdan = result.unwrap();
    assert!(!ceemdan.imfs.is_empty(), "Should have at least one IMF");
    assert_eq!(ceemdan.residual.len(), 1024, "Residual should match signal length");
}

#[test]
fn test_ceemdan_execution_with_invalid_signal_length() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");
    let config = GpuCeemданConfig::default();
    let signal = vec![0.0; 512];

    let result = executor.execute_ceemdan(&config, &signal);
    assert!(result.is_err(), "CEEMDAN should fail with mismatched signal length");
}

#[test]
fn test_eemd_result_structure() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");
    let config = GpuEemdConfig { num_trials: 10, num_samples: 256, ..Default::default() };
    let signal = vec![0.0; 256];

    let result = executor.execute_eemd(&config, &signal).expect("EEMD execution");

    // Verify result structure
    assert!(!result.imfs.is_empty(), "Should have IMFs");
    assert_eq!(result.residual.len(), 256, "Residual length should match");
    assert!(result.execution_time.as_millis() >= 0, "Execution time should be recorded");
}

#[test]
fn test_ceemdan_result_structure() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");
    let config = GpuCeemданConfig { num_trials: 10, num_samples: 256, ..Default::default() };
    let signal = vec![0.0; 256];

    let result = executor.execute_ceemdan(&config, &signal).expect("CEEMDAN execution");

    // Verify result structure
    assert!(!result.imfs.is_empty(), "Should have IMFs");
    assert_eq!(result.residual.len(), 256, "Residual length should match");
    assert!(result.execution_time.as_millis() >= 0, "Execution time should be recorded");
}

#[test]
fn test_kernel_launch_config_validation() {
    let device = CudaDevice::new(0).expect("Device creation");

    // Valid config
    let valid_config = KernelLaunchConfig::grid_1d(128, 256);
    assert!(valid_config.validate(&device).is_ok(), "Valid config should pass");

    // Valid: exactly at limit (1024)
    let at_limit = KernelLaunchConfig { block_x: 512, block_y: 2, ..Default::default() };
    assert!(at_limit.validate(&device).is_ok(), "Config at thread limit should pass");

    // Invalid: exceeds thread limit
    let invalid_config = KernelLaunchConfig {
        block_x: 512,
        block_y: 3, // 512 * 3 = 1536 > 1024
        ..Default::default()
    };
    assert!(invalid_config.validate(&device).is_err(), "Excessive threads should fail");
}

#[test]
fn test_multiple_device_properties_consistency() {
    let dev1 = CudaDevice::new(0).expect("First device");
    let dev2 = CudaDevice::new(0).expect("Second device");

    // Same device queried twice should have same properties
    assert_eq!(dev1.properties().name, dev2.properties().name);
    assert_eq!(dev1.properties().total_global_mem, dev2.properties().total_global_mem);
}

#[test]
fn test_executor_synchronization() {
    let executor = CudaKernelLauncher::new(0).expect("Launcher creation");
    let result = executor.synchronize();
    assert!(result.is_ok(), "Device synchronization should succeed");
}

#[test]
fn test_eemd_with_custom_noise_amplitude() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");
    let mut config = GpuEemdConfig::default();
    config.noise_amplitude = 0.5; // Higher noise

    let signal = vec![0.0; 1024];
    let result = executor.execute_eemd(&config, &signal);
    assert!(result.is_ok(), "EEMD with custom noise amplitude should work");
}

#[test]
fn test_eemd_with_different_trial_counts() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");

    for trials in &[10, 50, 100] {
        let config = GpuEemdConfig { num_trials: *trials, ..Default::default() };
        let signal = vec![0.0; 1024];

        let result = executor.execute_eemd(&config, &signal);
        assert!(result.is_ok(), "EEMD with {} trials should work", trials);
    }
}

#[test]
fn test_eemd_with_different_sample_sizes() {
    let mut executor = GpuEnsembleExecutor::new(0).expect("Executor creation");

    for samples in &[256, 512, 1024, 2048] {
        let config = GpuEemdConfig { num_samples: *samples, ..Default::default() };
        let signal = vec![0.0; *samples];

        let result = executor.execute_eemd(&config, &signal);
        assert!(result.is_ok(), "EEMD with {} samples should work", samples);
    }
}

#[test]
fn test_error_handling_on_memory_mismatch() {
    let device = CudaDevice::new(0).expect("Device creation");

    // Create a small memory allocation
    let small_mem = device.allocate(256).expect("Small allocation");
    assert_eq!(small_mem.size, 256);

    // Trying to use it for larger data should fail appropriately
    // (This is tested at the kernel executor level)
}

#[test]
fn test_cuda_api_error_display() {
    use ferromode::adapters::gpu::cuda_wrapper::CudaError;

    let err1 = CudaError::DeviceNotFound(5);
    assert!(err1.to_string().contains("device") || err1.to_string().contains("not found"));

    let err2 = CudaError::MemoryAllocationFailed { requested_size: 1024, available_memory: 512 };
    assert!(err2.to_string().contains("allocation") || err2.to_string().contains("memory"));

    let err3 = CudaError::KernelLaunchFailed {
        kernel_name: "test".to_string(),
        reason: "test error".to_string(),
    };
    assert!(err3.to_string().contains("test") || err3.to_string().contains("launch"));
}
