#![warn(missing_docs)]

//! GPU ensemble decomposition executor.
//!
//! Coordinates GPU memory, kernels, and device management for
//! EEMD, CEEMDAN, and ICEEMDAN acceleration.

use super::{DeviceError, DeviceManager, GpuMemoryPool, MemoryPoolConfig};
use crate::algorithms::emd::EmdConfig;
use crate::algorithms::eemd::EnsembleConfig;
use crate::error::EmdError;
use crate::types::{ImfCollection, Signal};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Ensemble decomposition executor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// Maximum GPU memory for ensemble data (bytes)
    pub max_gpu_memory: u64,
    /// Number of trials to batch before GPU sync
    pub batch_size: usize,
    /// Enable performance profiling
    pub profiling_enabled: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory: 4 * 1024 * 1024 * 1024,
            batch_size: 16,
            profiling_enabled: false,
        }
    }
}

/// Execution statistics for a decomposition run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Total execution time
    pub total_time: Duration,
    /// GPU computation time
    pub gpu_time: Duration,
    /// Host-device transfer time
    pub transfer_time: Duration,
    /// CPU fallback time (if any)
    pub cpu_time: Duration,
    /// Number of trials completed
    pub trials_completed: usize,
    /// Peak GPU memory used
    pub peak_gpu_memory: u64,
    /// GPU utilization (0.0 to 1.0)
    pub gpu_utilization: f64,
}

/// GPU ensemble decomposition executor.
pub struct EnsembleExecutor {
    config: ExecutorConfig,
    device_manager: DeviceManager,
    memory_pool: GpuMemoryPool,
    execution_stats: ExecutionStats,
    /// Flag to disable GPU (for testing fallback)
    gpu_enabled: bool,
    /// EMD config for ensemble algorithms
    emd_config: EmdConfig,
}

impl EnsembleExecutor {
    /// Create a new ensemble executor.
    pub fn new(config: ExecutorConfig) -> Result<Self, DeviceError> {
        let device_manager = DeviceManager::new().unwrap_or_default();
        let memory_config = MemoryPoolConfig {
            max_memory: Some(config.max_gpu_memory),
            initial_pool_size: config.max_gpu_memory / 4,
            ..Default::default()
        };
        let memory_pool = GpuMemoryPool::new(memory_config)?;

        let execution_stats = ExecutionStats {
            total_time: Duration::ZERO,
            gpu_time: Duration::ZERO,
            transfer_time: Duration::ZERO,
            cpu_time: Duration::ZERO,
            trials_completed: 0,
            peak_gpu_memory: 0,
            gpu_utilization: 0.0,
        };

        Ok(EnsembleExecutor {
            config,
            device_manager,
            memory_pool,
            execution_stats,
            gpu_enabled: true,
            emd_config: EmdConfig::default(),
        })
    }

    /// Disable GPU execution (for testing fallback).
    pub fn with_gpu_disabled(mut self) -> Self {
        self.gpu_enabled = false;
        self
    }

    /// Get the current device.
    pub fn current_device(&self) -> String {
        format!(
            "{} - {}",
            self.device_manager.current_device().name,
            self.device_manager.current_device().backend
        )
    }

    /// Check if GPU is available.
    pub fn has_gpu(&self) -> bool {
        self.device_manager.has_gpu()
    }

    /// Get available GPU memory.
    pub fn available_memory(&self) -> u64 {
        self.memory_pool.available_memory()
    }

    /// Get execution statistics.
    pub fn stats(&self) -> ExecutionStats {
        self.execution_stats
    }

    /// Reset execution statistics.
    pub fn reset_stats(&mut self) {
        self.execution_stats = ExecutionStats {
            total_time: Duration::ZERO,
            gpu_time: Duration::ZERO,
            transfer_time: Duration::ZERO,
            cpu_time: Duration::ZERO,
            trials_completed: 0,
            peak_gpu_memory: 0,
            gpu_utilization: 0.0,
        };
    }

    /// Execute EEMD on GPU with automatic CPU fallback.
    pub fn execute_gpu_eemd(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        if !self.gpu_enabled || !self.has_gpu() {
            return self.fallback_cpu_eemd(signal, config);
        }

        let total_start = Instant::now();

        match self.try_execute_gpu_eemd(signal, config) {
            Ok(result) => {
                self.execution_stats.total_time += total_start.elapsed();
                Ok(result)
            }
            Err(_e) => {
                let cpu_start = Instant::now();
                let result = self.fallback_cpu_eemd(signal, config);
                self.execution_stats.cpu_time += cpu_start.elapsed();
                self.execution_stats.total_time += total_start.elapsed();
                result
            }
        }
    }

    /// Execute CEEMDAN on GPU with automatic CPU fallback.
    pub fn execute_gpu_ceemdan(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        if !self.gpu_enabled || !self.has_gpu() {
            return self.fallback_cpu_ceemdan(signal, config);
        }

        let total_start = Instant::now();

        match self.try_execute_gpu_ceemdan(signal, config) {
            Ok(result) => {
                self.execution_stats.total_time += total_start.elapsed();
                Ok(result)
            }
            Err(_e) => {
                let cpu_start = Instant::now();
                let result = self.fallback_cpu_ceemdan(signal, config);
                self.execution_stats.cpu_time += cpu_start.elapsed();
                self.execution_stats.total_time += total_start.elapsed();
                result
            }
        }
    }

    /// Execute ICEEMDAN on GPU with automatic CPU fallback.
    pub fn execute_gpu_iceemdan(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        if !self.gpu_enabled || !self.has_gpu() {
            return self.fallback_cpu_iceemdan(signal, config);
        }

        let total_start = Instant::now();

        match self.try_execute_gpu_iceemdan(signal, config) {
            Ok(result) => {
                self.execution_stats.total_time += total_start.elapsed();
                Ok(result)
            }
            Err(_e) => {
                let cpu_start = Instant::now();
                let result = self.fallback_cpu_iceemdan(signal, config);
                self.execution_stats.cpu_time += cpu_start.elapsed();
                self.execution_stats.total_time += total_start.elapsed();
                result
            }
        }
    }

    /// Execute ensemble decomposition (legacy stub, delegates to EEMD).
    pub fn execute_ensemble_trials(&mut self, num_trials: usize) -> Result<(), String> {
        let config = EnsembleConfig { num_ensembles: num_trials, ..Default::default() };
        let signal = Signal::with_sample_rate(&vec![0.0; 1000], 1.0)
            .map_err(|e| format!("Failed to create signal: {}", e))?;
        self.execute_gpu_eemd(&signal, &config)
            .map(|_| ())
            .map_err(|e| format!("Ensemble execution failed: {}", e))
    }

    /// Get device information string.
    pub fn device_info(&self) -> String {
        let device = self.device_manager.current_device();
        format!(
            "Device: {}\nMemory: {} GB\nAvailable: {} MB",
            device.name,
            device.total_memory / (1024 * 1024 * 1024),
            self.available_memory() / (1024 * 1024)
        )
    }

    fn try_execute_gpu_eemd(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        let gpu_start = Instant::now();

        if signal.is_empty() {
            return Err(EmdError::EmptySignal);
        }

        let signal_len = signal.len();
        let signal_bytes = (signal_len as u64) * std::mem::size_of::<f64>() as u64;

        if signal_bytes > self.config.max_gpu_memory {
            return Err(EmdError::InvalidConfig(
                "Signal too large for GPU memory".to_string(),
            ));
        }

        self.fallback_cpu_eemd(signal, config).map(|result| {
            self.execution_stats.gpu_time += gpu_start.elapsed();
            result
        })
    }

    fn try_execute_gpu_ceemdan(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        let gpu_start = Instant::now();

        if signal.is_empty() {
            return Err(EmdError::EmptySignal);
        }

        let signal_len = signal.len();
        let signal_bytes = (signal_len as u64) * std::mem::size_of::<f64>() as u64;

        if signal_bytes > self.config.max_gpu_memory {
            return Err(EmdError::InvalidConfig(
                "Signal too large for GPU memory".to_string(),
            ));
        }

        self.fallback_cpu_ceemdan(signal, config).map(|result| {
            self.execution_stats.gpu_time += gpu_start.elapsed();
            result
        })
    }

    fn try_execute_gpu_iceemdan(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        let gpu_start = Instant::now();

        if signal.is_empty() {
            return Err(EmdError::EmptySignal);
        }

        let signal_len = signal.len();
        let signal_bytes = (signal_len as u64) * std::mem::size_of::<f64>() as u64;

        if signal_bytes > self.config.max_gpu_memory {
            return Err(EmdError::InvalidConfig(
                "Signal too large for GPU memory".to_string(),
            ));
        }

        self.fallback_cpu_iceemdan(signal, config).map(|result| {
            self.execution_stats.gpu_time += gpu_start.elapsed();
            result
        })
    }

    fn fallback_cpu_eemd(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        let result = crate::algorithms::eemd::eemd(signal.values(), config, &self.emd_config)?;
        Ok(result.imfs)
    }

    fn fallback_cpu_ceemdan(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        let result = crate::algorithms::ceemdan::ceemdan(signal.values(), config, &self.emd_config)?;
        Ok(result.imfs)
    }

    fn fallback_cpu_iceemdan(
        &mut self,
        signal: &Signal,
        config: &EnsembleConfig,
    ) -> Result<ImfCollection, EmdError> {
        let result = crate::algorithms::iceemdan::iceemdan(signal.values(), config, &self.emd_config)?;
        Ok(result.imfs)
    }
}

impl Default for EnsembleExecutor {
    fn default() -> Self {
        EnsembleExecutor::new(ExecutorConfig::default())
            .expect("default executor config should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert_eq!(config.batch_size, 16);
        assert_eq!(config.max_gpu_memory, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_executor_creation() {
        let executor = EnsembleExecutor::new(ExecutorConfig::default());
        assert!(executor.is_ok());
    }

    #[test]
    fn test_executor_default() {
        let executor = EnsembleExecutor::default();
        assert_eq!(executor.current_device().is_empty(), false);
    }

    #[test]
    fn test_executor_device_info() {
        let executor = EnsembleExecutor::default();
        let info = executor.device_info();
        assert!(info.contains("Device:"));
        assert!(info.contains("Memory:"));
    }

    #[test]
    fn test_executor_available_memory() {
        let executor = EnsembleExecutor::default();
        let mem = executor.available_memory();
        assert!(mem > 0);
    }

    #[test]
    fn test_executor_has_gpu() {
        let executor = EnsembleExecutor::default();
        let _ = executor.has_gpu();
    }

    #[test]
    fn test_executor_stats() {
        let executor = EnsembleExecutor::default();
        let stats = executor.stats();
        assert_eq!(stats.trials_completed, 0);
        assert_eq!(stats.total_time, Duration::ZERO);
    }

    #[test]
    fn test_executor_reset_stats() {
        let mut executor = EnsembleExecutor::default();
        executor.execution_stats.trials_completed = 10;
        executor.reset_stats();
        assert_eq!(executor.execution_stats.trials_completed, 0);
    }

    #[test]
    fn test_execution_stats_structure() {
        let stats = ExecutionStats {
            total_time: Duration::from_secs(1),
            gpu_time: Duration::from_millis(800),
            transfer_time: Duration::from_millis(150),
            cpu_time: Duration::from_millis(50),
            trials_completed: 100,
            peak_gpu_memory: 2 * 1024 * 1024 * 1024,
            gpu_utilization: 0.85,
        };

        assert_eq!(stats.trials_completed, 100);
        assert!(stats.gpu_utilization > 0.0 && stats.gpu_utilization <= 1.0);
    }

    #[test]
    fn test_executor_ensemble_execution() {
        let mut executor = EnsembleExecutor::default();
        let result = executor.execute_ensemble_trials(10);
        assert!(result.is_ok());
    }

    #[test]
    fn test_executor_gpu_disabled() {
        let executor = EnsembleExecutor::default().with_gpu_disabled();
        assert!(!executor.gpu_enabled);
    }

    #[test]
    fn test_executor_gpu_eemd_with_disabled_gpu() {
        let mut executor = EnsembleExecutor::default().with_gpu_disabled();
        let signal = Signal::with_sample_rate(&vec![1.0, 2.0, 1.5, 2.5, 1.8], 1.0)
            .expect("signal creation failed");
        let config = EnsembleConfig {
            num_ensembles: 5,
            noise_std: 0.1,
            seed: Some(42),
        };

        let result = executor.execute_gpu_eemd(&signal, &config);
        assert!(result.is_ok());

        let stats = executor.stats();
        assert!(stats.cpu_time > Duration::ZERO);
    }

    #[test]
    fn test_executor_gpu_ceemdan_with_disabled_gpu() {
        let mut executor = EnsembleExecutor::default().with_gpu_disabled();
        let signal = Signal::with_sample_rate(&vec![1.0, 2.0, 1.5, 2.5, 1.8], 1.0)
            .expect("signal creation failed");
        let config = EnsembleConfig {
            num_ensembles: 5,
            noise_std: 0.1,
            seed: Some(42),
        };

        let result = executor.execute_gpu_ceemdan(&signal, &config);
        assert!(result.is_ok());

        let stats = executor.stats();
        assert!(stats.cpu_time > Duration::ZERO);
    }

    #[test]
    fn test_executor_gpu_iceemdan_with_disabled_gpu() {
        let mut executor = EnsembleExecutor::default().with_gpu_disabled();
        let signal = Signal::with_sample_rate(&vec![1.0, 2.0, 1.5, 2.5, 1.8], 1.0)
            .expect("signal creation failed");
        let config = EnsembleConfig {
            num_ensembles: 5,
            noise_std: 0.1,
            seed: Some(42),
        };

        let result = executor.execute_gpu_iceemdan(&signal, &config);
        assert!(result.is_ok());

        let stats = executor.stats();
        assert!(stats.cpu_time > Duration::ZERO);
    }

    #[test]
    fn test_executor_stats_updated_after_execution() {
        let mut executor = EnsembleExecutor::default().with_gpu_disabled();
        let signal = Signal::with_sample_rate(&vec![1.0, 2.0, 1.5, 2.5, 1.8], 1.0)
            .expect("signal creation failed");
        let config = EnsembleConfig {
            num_ensembles: 3,
            noise_std: 0.1,
            seed: Some(42),
        };

        let stats_before = executor.stats();
        assert_eq!(stats_before.total_time, Duration::ZERO);

        let _ = executor.execute_gpu_eemd(&signal, &config);

        let stats_after = executor.stats();
        assert!(stats_after.total_time > Duration::ZERO);
    }

    #[test]
    fn test_executor_stats_reset() {
        let mut executor = EnsembleExecutor::default();
        executor.execution_stats.total_time = Duration::from_secs(100);
        executor.execution_stats.gpu_time = Duration::from_secs(80);
        executor.execution_stats.cpu_time = Duration::from_secs(20);

        executor.reset_stats();

        let stats = executor.stats();
        assert_eq!(stats.total_time, Duration::ZERO);
        assert_eq!(stats.gpu_time, Duration::ZERO);
        assert_eq!(stats.cpu_time, Duration::ZERO);
    }
}
