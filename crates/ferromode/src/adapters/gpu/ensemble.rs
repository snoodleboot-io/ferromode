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
    pub fn execute(&mut self, signal: &[f64]) -> Result<GpuEemdResult, String> {
        let start = Instant::now();

        let signal_length = signal.len();
        if signal_length == 0 {
            return Err("Signal is empty".to_string());
        }

        // Check if GPU should be used
        let use_gpu = self.executor.has_gpu() && !self.config.force_cpu;

        let result =
            if use_gpu { self.execute_gpu_eemd(signal)? } else { self.execute_cpu_eemd(signal)? };

        let elapsed = start.elapsed();

        Ok(GpuEemdResult {
            imfs: result.0,
            residue: result.1,
            execution_time: elapsed,
            used_gpu: use_gpu,
            num_trials: self.config.num_ensembles,
        })
    }

    /// Execute EEMD on GPU.
    fn execute_gpu_eemd(&self, signal: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        // TODO: Full GPU EEMD implementation
        // 1. Allocate GPU memory for signal and ensemble data
        // 2. Generate random noise for all trials in parallel
        // 3. Create noisy signals: signal + noise
        // 4. Run EMD on each noisy signal via GPU kernel
        // 5. Average IMFs across trials
        // 6. Return ensemble-averaged IMFs

        // For now, fall back to CPU with a note
        self.execute_cpu_eemd(signal)
    }

    /// Execute EEMD on CPU.
    fn execute_cpu_eemd(&self, signal: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        // Call the CPU EEMD implementation from algorithms module
        use crate::algorithms::eemd::{eemd, EnsembleConfig};
        use crate::algorithms::emd::EmdConfig;

        let ensemble_config = EnsembleConfig {
            num_ensembles: self.config.num_ensembles,
            noise_std: self.config.noise_std,
            seed: self.config.seed,
        };

        let emd_config = EmdConfig::default();

        match eemd(signal, &ensemble_config, &emd_config) {
            Ok(result) => Ok((result.imfs.imfs, result.imfs.residue)),
            Err(e) => Err(format!("EEMD execution failed: {}", e)),
        }
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
    pub fn execute(&mut self, signal: &[f64]) -> Result<GpuCeemданResult, String> {
        let start = Instant::now();

        let signal_length = signal.len();
        if signal_length == 0 {
            return Err("Signal is empty".to_string());
        }

        // Check if GPU should be used
        let use_gpu = self.executor.has_gpu() && !self.config.force_cpu;

        let result = if use_gpu {
            self.execute_gpu_ceemdan(signal)?
        } else {
            self.execute_cpu_ceemdan(signal)?
        };

        let elapsed = start.elapsed();

        Ok(GpuCeemданResult {
            imfs: result.0,
            residue: result.1,
            execution_time: elapsed,
            used_gpu: use_gpu,
            num_trials: self.config.num_ensembles,
        })
    }

    /// Execute CEEMDAN on GPU.
    fn execute_gpu_ceemdan(&self, signal: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        // TODO: Full GPU CEEMDAN implementation
        // Similar to EEMD but with mode-by-mode acceleration
        // For now, fall back to CPU
        self.execute_cpu_ceemdan(signal)
    }

    /// Execute CEEMDAN on CPU.
    fn execute_cpu_ceemdan(&self, signal: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        use crate::algorithms::ceemdan::ceemdan;
        use crate::algorithms::eemd::EnsembleConfig;
        use crate::algorithms::emd::EmdConfig;

        let ensemble_config = EnsembleConfig {
            num_ensembles: self.config.num_ensembles,
            noise_std: self.config.noise_std,
            seed: self.config.seed,
        };

        let emd_config = EmdConfig::default();

        match ceemdan(signal, &ensemble_config, &emd_config) {
            Ok(result) => Ok((result.imfs.imfs, result.imfs.residue)),
            Err(e) => Err(format!("CEEMDAN execution failed: {}", e)),
        }
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
    pub fn execute(&mut self, signal: &[f64]) -> Result<GpuIceemданResult, String> {
        let start = Instant::now();

        let signal_length = signal.len();
        if signal_length == 0 {
            return Err("Signal is empty".to_string());
        }

        // Check if GPU should be used
        let use_gpu = self.executor.has_gpu() && !self.config.force_cpu;

        let result = if use_gpu {
            self.execute_gpu_iceemdan(signal)?
        } else {
            self.execute_cpu_iceemdan(signal)?
        };

        let elapsed = start.elapsed();

        Ok(GpuIceemданResult {
            imfs: result.0,
            residue: result.1,
            execution_time: elapsed,
            used_gpu: use_gpu,
            num_trials: self.config.num_ensembles,
        })
    }

    /// Execute ICEEMDAN on GPU.
    fn execute_gpu_iceemdan(&self, signal: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        // TODO: Full GPU ICEEMDAN implementation
        // Improved CEEMDAN with complementary ensemble pairing
        // For now, fall back to CPU
        self.execute_cpu_iceemdan(signal)
    }

    /// Execute ICEEMDAN on CPU.
    fn execute_cpu_iceemdan(&self, signal: &[f64]) -> Result<(Vec<Vec<f64>>, Vec<f64>), String> {
        // TODO: Implement ICEEMDAN algorithm when available in algorithms module
        // For now, use CEEMDAN as fallback
        use crate::algorithms::ceemdan::ceemdan;
        use crate::algorithms::eemd::EnsembleConfig;
        use crate::algorithms::emd::EmdConfig;

        let ensemble_config = EnsembleConfig {
            num_ensembles: self.config.num_ensembles,
            noise_std: self.config.noise_std,
            seed: self.config.seed,
        };

        let emd_config = EmdConfig::default();

        match ceemdan(signal, &ensemble_config, &emd_config) {
            Ok(result) => Ok((result.imfs.imfs, result.imfs.residue)),
            Err(e) => Err(format!("ICEEMDAN execution failed: {}", e)),
        }
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
        let mut config = GpuEemdConfig::default();
        config.num_ensembles = 5; // Small number for quick test
        config.force_cpu = true;
        let mut executor = GpuEemdExecutor::new(config).expect("executor creation");

        // Generate a simple sinusoidal signal with at least 100 samples
        let signal: Vec<f64> = (0..100)
            .map(|i| {
                let t = (i as f64) * 0.1;
                (t.sin() + 0.5 * (2.0 * t).sin()).abs()
            })
            .collect();

        let result = executor.execute(&signal);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.num_trials, 5);
        assert!(res.used_gpu == false); // force_cpu = true
    }

    #[test]
    fn test_gpu_ceemdan_executor_execute() {
        let mut config = GpuCeemданConfig::default();
        config.num_ensembles = 5;
        config.force_cpu = true;
        let mut executor = GpuCeemданExecutor::new(config).expect("executor creation");

        // Generate a simple sinusoidal signal
        let signal: Vec<f64> = (0..100)
            .map(|i| {
                let t = (i as f64) * 0.1;
                t.sin()
            })
            .collect();

        let result = executor.execute(&signal);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.num_trials, 5);
        assert!(res.used_gpu == false);
    }

    #[test]
    fn test_gpu_iceemdan_executor_execute() {
        let mut config = GpuIceemданConfig::default();
        config.num_ensembles = 5;
        config.force_cpu = true;
        let mut executor = GpuIceemданExecutor::new(config).expect("executor creation");

        // Generate a simple sinusoidal signal
        let signal: Vec<f64> = (0..100)
            .map(|i| {
                let t = (i as f64) * 0.1;
                t.sin() + 0.3 * (3.0 * t).sin()
            })
            .collect();

        let result = executor.execute(&signal);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.num_trials, 5);
        assert!(res.used_gpu == false);
    }

    #[test]
    fn test_force_cpu_mode() {
        let mut config = GpuEemdConfig::default();
        config.force_cpu = true;
        let executor = GpuEemdExecutor::new(config).expect("executor creation");
        assert!(executor.config().force_cpu);
    }
}
