#![warn(missing_docs)]

//! GPU acceleration module for ensemble decomposition methods.
//!
//! Provides optimized CUDA/ROCm implementations of EEMD, CEEMDAN, and ICEEMDAN
//! with automatic fallback to CPU if GPU is unavailable.

pub mod device;
pub mod executor;
pub mod kernels;
pub mod memory;

pub use device::{
    DeviceError, DeviceId, DeviceInfo, DeviceManager, DeviceSelectionStrategy, GpuBackend,
};
pub use executor::{EnsembleExecutor, ExecutionStats, ExecutorConfig};
pub use kernels::{
    CpuKernelLauncher, DataTransfer, KernelConfig, KernelLaunchResult, KernelLauncher,
    TransferDirection,
};
pub use memory::{AllocationId, GpuMemoryPool, MemoryPoolConfig, MemoryStats};

use serde::{Deserialize, Serialize};

/// Configuration for GPU acceleration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Device selection strategy
    pub device_strategy: DeviceSelectionStrategy,
    /// Explicit device to use (if strategy is Explicit)
    pub explicit_device: Option<DeviceId>,
    /// Maximum GPU memory to allocate (in bytes), None = use all available
    pub max_gpu_memory: Option<u64>,
    /// Number of trials to batch on GPU before syncing
    pub batch_size: usize,
    /// Whether to fall back to CPU if GPU operations fail
    pub allow_cpu_fallback: bool,
    /// Enable performance profiling
    pub enable_profiling: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_strategy: DeviceSelectionStrategy::MostMemory,
            explicit_device: None,
            max_gpu_memory: None,
            batch_size: 16,
            allow_cpu_fallback: true,
            enable_profiling: false,
        }
    }
}

/// GPU adapter for ensemble decomposition methods.
pub struct GpuAdapter {
    config: GpuConfig,
    device_manager: DeviceManager,
}

impl GpuAdapter {
    /// Create a new GPU adapter with the given configuration.
    pub fn new(config: GpuConfig) -> Result<Self, DeviceError> {
        let mut device_manager = DeviceManager::new().unwrap_or_default();

        // Apply device selection strategy
        if config.device_strategy == DeviceSelectionStrategy::Explicit {
            if let Some(device_id) = config.explicit_device {
                device_manager.set_device(device_id)?;
            }
        } else {
            device_manager.select_by_strategy(config.device_strategy)?;
        }

        Ok(GpuAdapter { config, device_manager })
    }

    /// Get the current device manager.
    pub fn device_manager(&self) -> &DeviceManager {
        &self.device_manager
    }

    /// Get the current device info.
    pub fn current_device(&self) -> &DeviceInfo {
        self.device_manager.current_device()
    }

    /// Check if GPU is available.
    pub fn has_gpu(&self) -> bool {
        self.device_manager.has_gpu()
    }

    /// Get adapter configuration.
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
}

impl Default for GpuAdapter {
    fn default() -> Self {
        GpuAdapter::new(GpuConfig::default()).unwrap_or_else(|_| GpuAdapter {
            config: GpuConfig::default(),
            device_manager: DeviceManager::default(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.device_strategy, DeviceSelectionStrategy::MostMemory);
        assert_eq!(config.batch_size, 16);
    }

    #[test]
    fn test_gpu_adapter_default() {
        let adapter = GpuAdapter::default();
        assert_eq!(adapter.device_manager.device_count() >= 1, true);
    }

    #[test]
    fn test_gpu_adapter_new() {
        let config = GpuConfig::default();
        let adapter = GpuAdapter::new(config);
        assert!(adapter.is_ok());
    }
}
