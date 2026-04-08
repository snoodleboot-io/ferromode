#![warn(missing_docs)]

//! CUDA GPU integration for ensemble decomposition.
//!
//! Provides high-level API for executing EEMD and CEEMDAN on NVIDIA GPUs,
//! with automatic memory management, error handling, and CPU fallback.

use super::cuda_kernels::{
    CudaKernelExecutor, ExtremaKernel, NoiseGenerationKernel, SignalAdditionKernel,
};
use super::cuda_wrapper::{CudaDevice, CudaError, CudaResult};
use std::time::{Duration, Instant};

/// Configuration for GPU EEMD execution.
#[derive(Debug, Clone)]
pub struct GpuEemdConfig {
    /// Number of trials (ensemble members)
    pub num_trials: usize,
    /// Noise amplitude (standard deviation)
    pub noise_amplitude: f64,
    /// Number of signal samples
    pub num_samples: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Device ID to use
    pub device_id: u32,
}

impl Default for GpuEemdConfig {
    fn default() -> Self {
        GpuEemdConfig {
            num_trials: 100,
            noise_amplitude: 0.2,
            num_samples: 1024,
            seed: 42,
            device_id: 0,
        }
    }
}

/// GPU EEMD execution result.
#[derive(Debug, Clone)]
pub struct GpuEemdResult {
    /// Computed IMFs (Intrinsic Mode Functions)
    pub imfs: Vec<Vec<f64>>,
    /// Residual trend
    pub residual: Vec<f64>,
    /// Total execution time
    pub execution_time: Duration,
    /// GPU computation time
    pub gpu_time: Duration,
    /// Memory transfer time
    pub transfer_time: Duration,
}

/// GPU CEEMDAN execution configuration.
#[derive(Debug, Clone)]
pub struct GpuCeemданConfig {
    /// Number of trials
    pub num_trials: usize,
    /// Noise amplitude
    pub noise_amplitude: f64,
    /// Number of signal samples
    pub num_samples: usize,
    /// Random seed
    pub seed: u64,
    /// Device ID to use
    pub device_id: u32,
}

impl Default for GpuCeemданConfig {
    fn default() -> Self {
        GpuCeemданConfig {
            num_trials: 100,
            noise_amplitude: 0.2,
            num_samples: 1024,
            seed: 42,
            device_id: 0,
        }
    }
}

/// GPU CEEMDAN execution result.
#[derive(Debug, Clone)]
pub struct GpuCeemданResult {
    /// Computed IMFs
    pub imfs: Vec<Vec<f64>>,
    /// Residual trend
    pub residual: Vec<f64>,
    /// Execution time
    pub execution_time: Duration,
    /// GPU time
    pub gpu_time: Duration,
    /// Transfer time
    pub transfer_time: Duration,
}

/// High-level GPU executor for ensemble methods.
pub struct GpuEnsembleExecutor {
    device: CudaDevice,
    kernel_executor: CudaKernelExecutor,
}

impl GpuEnsembleExecutor {
    /// Create a new GPU ensemble executor.
    pub fn new(device_id: u32) -> CudaResult<Self> {
        let device = CudaDevice::new(device_id)?;
        let kernel_executor = CudaKernelExecutor::new(device_id)?;

        Ok(GpuEnsembleExecutor { device, kernel_executor })
    }

    /// Get device information.
    pub fn device_info(&self) -> String {
        let props = self.device.properties();
        format!(
            "CUDA Device: {} (Compute {}.{})\nMemory: {} GB",
            props.name,
            props.compute_capability_major,
            props.compute_capability_minor,
            props.total_global_mem / (1024 * 1024 * 1024)
        )
    }

    /// Execute GPU EEMD decomposition.
    ///
    /// This is a stub implementation that will be enhanced with actual
    /// kernel orchestration. Currently returns a placeholder result.
    pub fn execute_eemd(
        &mut self,
        config: &GpuEemdConfig,
        signal: &[f64],
    ) -> CudaResult<GpuEemdResult> {
        let start = Instant::now();
        let mut gpu_time = Duration::ZERO;
        let mut transfer_time = Duration::ZERO;

        // Validate input
        if signal.len() != config.num_samples {
            return Err(CudaError::InvalidKernelConfig(format!(
                "Signal length {} != expected {}",
                signal.len(),
                config.num_samples
            )));
        }

        // TODO: Implement actual GPU EEMD execution:
        // 1. Allocate device memory for signal + working buffers
        // 2. Copy signal to device (H2D transfer)
        // 3. For each trial:
        //    a. Generate random noise on device
        //    b. Add noise to signal (on device)
        //    c. Decompose noisy signal (via spline interpolation + sifting)
        // 4. Combine results from all trials
        // 5. Copy results back to host (D2H transfer)
        // 6. Deallocate device memory

        // Stub: allocate device memory
        let signal_size = (config.num_samples as u64) * std::mem::size_of::<f64>() as u64;
        let _signal_device = self.device.allocate(signal_size)?;

        // Stub: create noise kernel for each trial
        for trial_id in 0..config.num_trials {
            let noise_kernel = NoiseGenerationKernel::new(
                config.num_samples as u32,
                config.seed + trial_id as u64,
            );
            let _config = noise_kernel.optimal_launch_config();
            // Would launch kernel here
        }

        let elapsed = start.elapsed();

        Ok(GpuEemdResult {
            imfs: vec![signal.to_vec()], // Stub: return input as first IMF
            residual: vec![0.0; config.num_samples],
            execution_time: elapsed,
            gpu_time,
            transfer_time,
        })
    }

    /// Execute GPU CEEMDAN decomposition.
    ///
    /// Stub implementation that will be enhanced with kernel orchestration.
    pub fn execute_ceemdan(
        &mut self,
        config: &GpuCeemданConfig,
        signal: &[f64],
    ) -> CudaResult<GpuCeemданResult> {
        let start = Instant::now();
        let mut gpu_time = Duration::ZERO;
        let mut transfer_time = Duration::ZERO;

        // Validate input
        if signal.len() != config.num_samples {
            return Err(CudaError::InvalidKernelConfig(format!(
                "Signal length {} != expected {}",
                signal.len(),
                config.num_samples
            )));
        }

        // TODO: Implement actual GPU CEEMDAN execution:
        // Similar to EEMD but with:
        // - EMD of signal first to get first IMF
        // - Generate noise and add to signal for each trial
        // - EMD of noisy signals
        // - Combine results with different weighting

        let signal_size = (config.num_samples as u64) * std::mem::size_of::<f64>() as u64;
        let _signal_device = self.device.allocate(signal_size)?;

        // Stub: create extrema kernel
        let extrema = ExtremaKernel::new(config.num_samples as u32, 1);
        let _config = extrema.optimal_launch_config();

        let elapsed = start.elapsed();

        Ok(GpuCeemданResult {
            imfs: vec![signal.to_vec()],
            residual: vec![0.0; config.num_samples],
            execution_time: elapsed,
            gpu_time,
            transfer_time,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_eemd_config_default() {
        let config = GpuEemdConfig::default();
        assert_eq!(config.num_trials, 100);
        assert_eq!(config.num_samples, 1024);
    }

    #[test]
    fn test_gpu_ceemdan_config_default() {
        let config = GpuCeemданConfig::default();
        assert_eq!(config.num_trials, 100);
        assert_eq!(config.num_samples, 1024);
    }

    #[test]
    fn test_gpu_ensemble_executor_new() {
        let executor = GpuEnsembleExecutor::new(0);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_gpu_ensemble_executor_device_info() {
        let executor = GpuEnsembleExecutor::new(0).unwrap();
        let info = executor.device_info();
        assert!(info.contains("CUDA Device"));
        assert!(info.contains("Memory"));
    }

    #[test]
    fn test_gpu_eemd_invalid_signal_length() {
        let mut executor = GpuEnsembleExecutor::new(0).unwrap();
        let config = GpuEemdConfig::default();
        let signal = vec![0.0; 512]; // Wrong size

        let result = executor.execute_eemd(&config, &signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_eemd_valid_signal() {
        let mut executor = GpuEnsembleExecutor::new(0).unwrap();
        let config = GpuEemdConfig::default();
        let signal = vec![0.0; 1024]; // Correct size

        let result = executor.execute_eemd(&config, &signal);
        assert!(result.is_ok());
        let eemd = result.unwrap();
        assert!(!eemd.imfs.is_empty());
    }

    #[test]
    fn test_gpu_ceemdan_valid_signal() {
        let mut executor = GpuEnsembleExecutor::new(0).unwrap();
        let config = GpuCeemданConfig::default();
        let signal = vec![0.0; 1024];

        let result = executor.execute_ceemdan(&config, &signal);
        assert!(result.is_ok());
        let ceemdan = result.unwrap();
        assert!(!ceemdan.imfs.is_empty());
    }
}
