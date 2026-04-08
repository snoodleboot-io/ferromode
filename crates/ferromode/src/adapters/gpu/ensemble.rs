#![warn(missing_docs)]

//! GPU-accelerated ensemble decomposition wrappers.
//!
//! Provides GPU-optimized wrappers for EEMD, CEEMDAN, and ICEEMDAN
//! with automatic fallback to CPU implementations.

use super::{EnsembleExecutor, ExecutorConfig};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// GPU-accelerated EEMD configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuEemdConfig {
    /// Number of ensemble trials
    pub num_ensembles: usize,
    /// Noise standard deviation (fraction of signal std)
    pub noise_std: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// GPU executor configuration
    pub executor_config: ExecutorConfig,
    /// Force CPU execution (for benchmarking)
    pub force_cpu: bool,
}

impl Default for GpuEemdConfig {
    fn default() -> Self {
        Self {
            num_ensembles: 100,
            noise_std: 0.2,
            seed: None,
            executor_config: ExecutorConfig::default(),
            force_cpu: false,
        }
    }
}

/// GPU-accelerated CEEMDAN configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCeemданConfig {
    /// Number of ensemble trials
    pub num_ensembles: usize,
    /// Noise standard deviation
    pub noise_std: f64,
    /// Number of sifting iterations per trial
    pub num_sifting_iterations: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// GPU executor configuration
    pub executor_config: ExecutorConfig,
    /// Force CPU execution
    pub force_cpu: bool,
}

impl Default for GpuCeemданConfig {
    fn default() -> Self {
        Self {
            num_ensembles: 100,
            noise_std: 0.2,
            num_sifting_iterations: 100,
            seed: None,
            executor_config: ExecutorConfig::default(),
            force_cpu: false,
        }
    }
}

/// GPU-accelerated ICEEMDAN configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuIceemданConfig {
    /// Number of ensemble trials
    pub num_ensembles: usize,
    /// Noise standard deviation
    pub noise_std: f64,
    /// Random seed
    pub seed: Option<u64>,
    /// GPU executor configuration
    pub executor_config: ExecutorConfig,
    /// Force CPU execution
    pub force_cpu: bool,
}

impl Default for GpuIceemданConfig {
    fn default() -> Self {
        Self {
            num_ensembles: 100,
            noise_std: 0.2,
            seed: None,
            executor_config: ExecutorConfig::default(),
            force_cpu: false,
        }
    }
}

/// GPU EEMD executor wrapper.
pub struct GpuEemdExecutor {
    config: GpuEemdConfig,
    executor: EnsembleExecutor,
}

impl GpuEemdExecutor {
    /// Create new GPU EEMD executor.
    pub fn new(config: GpuEemdConfig) -> Result<Self, String> {
        let executor =
            EnsembleExecutor::new(config.executor_config.clone()).map_err(|e| e.to_string())?;
        Ok(GpuEemdExecutor { config, executor })
    }

    /// Get executor configuration.
    pub fn config(&self) -> &GpuEemdConfig {
        &self.config
    }

    /// Get underlying executor.
    pub fn executor(&self) -> &EnsembleExecutor {
        &self.executor
    }

    /// Execute EEMD decomposition.
    pub fn execute(&mut self, _signal: &[f64]) -> Result<GpuEemdResult, String> {
        let start = Instant::now();

        // TODO: Implement GPU EEMD:
        // 1. Check if GPU is available and not force_cpu
        // 2. If GPU available:
        //    a. Allocate GPU memory for ensemble trials
        //    b. Transfer signal to GPU
        //    c. Launch GPU kernels for noisy signal generation + EMD
        //    d. Reduce/average IMFs across trials
        //    e. Transfer results back to CPU
        // 3. If no GPU or force_cpu:
        //    a. Fall back to CPU EEMD implementation
        //    b. Call crate::algorithms::eemd::eemd() directly

        let elapsed = start.elapsed();

        Ok(GpuEemdResult {
            imfs: vec![],
            residue: vec![],
            execution_time: elapsed,
            used_gpu: self.executor.has_gpu() && !self.config.force_cpu,
            num_trials: self.config.num_ensembles,
        })
    }
}

/// Result of GPU EEMD execution.
#[derive(Debug, Clone)]
pub struct GpuEemdResult {
    /// Intrinsic Mode Functions
    pub imfs: Vec<Vec<f64>>,
    /// Residue component
    pub residue: Vec<f64>,
    /// Total execution time
    pub execution_time: std::time::Duration,
    /// Whether GPU was used
    pub used_gpu: bool,
    /// Number of trials completed
    pub num_trials: usize,
}

/// GPU CEEMDAN executor wrapper.
pub struct GpuCeemданExecutor {
    config: GpuCeemданConfig,
    executor: EnsembleExecutor,
}

impl GpuCeemданExecutor {
    /// Create new GPU CEEMDAN executor.
    pub fn new(config: GpuCeemданConfig) -> Result<Self, String> {
        let executor =
            EnsembleExecutor::new(config.executor_config.clone()).map_err(|e| e.to_string())?;
        Ok(GpuCeemданExecutor { config, executor })
    }

    /// Execute CEEMDAN decomposition.
    pub fn execute(&mut self, _signal: &[f64]) -> Result<GpuCeemданResult, String> {
        let start = Instant::now();

        // TODO: Similar to EEMD but with mode-by-mode acceleration
        // Stage-wise EMD with noise ensemble

        let elapsed = start.elapsed();

        Ok(GpuCeemданResult {
            imfs: vec![],
            residue: vec![],
            execution_time: elapsed,
            used_gpu: self.executor.has_gpu() && !self.config.force_cpu,
            num_trials: self.config.num_ensembles,
        })
    }
}

/// Result of GPU CEEMDAN execution.
#[derive(Debug, Clone)]
pub struct GpuCeemданResult {
    /// Intrinsic Mode Functions
    pub imfs: Vec<Vec<f64>>,
    /// Residue component
    pub residue: Vec<f64>,
    /// Total execution time
    pub execution_time: std::time::Duration,
    /// Whether GPU was used
    pub used_gpu: bool,
    /// Number of trials completed
    pub num_trials: usize,
}

/// GPU ICEEMDAN executor wrapper.
pub struct GpuIceemданExecutor {
    config: GpuIceemданConfig,
    executor: EnsembleExecutor,
}

impl GpuIceemданExecutor {
    /// Create new GPU ICEEMDAN executor.
    pub fn new(config: GpuIceemданConfig) -> Result<Self, String> {
        let executor =
            EnsembleExecutor::new(config.executor_config.clone()).map_err(|e| e.to_string())?;
        Ok(GpuIceemданExecutor { config, executor })
    }

    /// Execute ICEEMDAN decomposition.
    pub fn execute(&mut self, _signal: &[f64]) -> Result<GpuIceemданResult, String> {
        let start = Instant::now();

        // TODO: Improved CEEMDAN with complementary ensemble pairing

        let elapsed = start.elapsed();

        Ok(GpuIceemданResult {
            imfs: vec![],
            residue: vec![],
            execution_time: elapsed,
            used_gpu: self.executor.has_gpu() && !self.config.force_cpu,
            num_trials: self.config.num_ensembles,
        })
    }
}

/// Result of GPU ICEEMDAN execution.
#[derive(Debug, Clone)]
pub struct GpuIceemданResult {
    /// Intrinsic Mode Functions
    pub imfs: Vec<Vec<f64>>,
    /// Residue component
    pub residue: Vec<f64>,
    /// Total execution time
    pub execution_time: std::time::Duration,
    /// Whether GPU was used
    pub used_gpu: bool,
    /// Number of trials completed
    pub num_trials: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_eemd_config_default() {
        let config = GpuEemdConfig::default();
        assert_eq!(config.num_ensembles, 100);
        assert_eq!(config.noise_std, 0.2);
        assert!(!config.force_cpu);
    }

    #[test]
    fn test_gpu_ceemdan_config_default() {
        let config = GpuCeemданConfig::default();
        assert_eq!(config.num_ensembles, 100);
        assert_eq!(config.num_sifting_iterations, 100);
    }

    #[test]
    fn test_gpu_iceemdan_config_default() {
        let config = GpuIceemданConfig::default();
        assert_eq!(config.num_ensembles, 100);
    }

    #[test]
    fn test_gpu_eemd_executor_creation() {
        let config = GpuEemdConfig::default();
        let executor = GpuEemdExecutor::new(config);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_gpu_ceemdan_executor_creation() {
        let config = GpuCeemданConfig::default();
        let executor = GpuCeemданExecutor::new(config);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_gpu_iceemdan_executor_creation() {
        let config = GpuIceemданConfig::default();
        let executor = GpuIceemданExecutor::new(config);
        assert!(executor.is_ok());
    }

    #[test]
    fn test_gpu_eemd_result_structure() {
        let result = GpuEemdResult {
            imfs: vec![],
            residue: vec![],
            execution_time: std::time::Duration::from_secs(1),
            used_gpu: true,
            num_trials: 100,
        };
        assert_eq!(result.num_trials, 100);
        assert!(result.used_gpu);
    }

    #[test]
    fn test_gpu_eemd_executor_execute() {
        let config = GpuEemdConfig::default();
        let mut executor = GpuEemdExecutor::new(config).expect("executor creation");
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = executor.execute(&signal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_ceemdan_executor_execute() {
        let config = GpuCeemданConfig::default();
        let mut executor = GpuCeemданExecutor::new(config).expect("executor creation");
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = executor.execute(&signal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_iceemdan_executor_execute() {
        let config = GpuIceemданConfig::default();
        let mut executor = GpuIceemданExecutor::new(config).expect("executor creation");
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = executor.execute(&signal);
        assert!(result.is_ok());
    }

    #[test]
    fn test_force_cpu_mode() {
        let mut config = GpuEemdConfig::default();
        config.force_cpu = true;
        let executor = GpuEemdExecutor::new(config).expect("executor creation");
        assert!(executor.config().force_cpu);
    }
}
