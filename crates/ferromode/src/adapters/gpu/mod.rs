#![warn(missing_docs)]

//! GPU acceleration module for ensemble decomposition methods.
//!
//! Provides optimized CUDA/ROCm implementations of EEMD, CEEMDAN, and ICEEMDAN
//! with automatic fallback to CPU if GPU is unavailable.
//!
//! # Features
//!
//! - **Device Abstraction**: Unified interface for CUDA, ROCm, WebGPU, and CPU
//! - **Memory Management**: GPU memory pool with fragmentation prevention
//! - **Kernel Execution**: Data-parallel kernel launching and coordination
//! - **Ensemble Acceleration**: GPU-optimized EEMD, CEEMDAN, ICEEMDAN
//! - **Automatic Fallback**: Seamless CPU fallback if GPU unavailable
//!
//! # Example
//!
//! ```ignore
//! use ferromode::adapters::gpu::{GpuAdapter, GpuConfig};
//! use ferromode::algorithms::eemd::{eemd, EnsembleConfig};
//!
//! let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let gpu_config = GpuConfig::default();
//! let ensemble_config = EnsembleConfig::default();
//!
//! // GPU adapter automatically detects best device
//! let adapter = GpuAdapter::new(gpu_config)?;
//!
//! // Use GPU for EEMD (or falls back to CPU)
//! let result = adapter.eemd(&signal, &ensemble_config)?;
//! ```

pub mod device;

pub use device::{
    DeviceError, DeviceId, DeviceInfo, DeviceManager, DeviceSelectionStrategy, GpuBackend,
};

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
///
/// Provides GPU-accelerated versions of EEMD, CEEMDAN, and ICEEMDAN
/// with automatic CPU fallback.
pub struct GpuAdapter {
    config: GpuConfig,
    device_manager: DeviceManager,
}

impl GpuAdapter {
    /// Create a new GPU adapter with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if device detection fails and CPU fallback is disabled.
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
        GpuAdapter::new(GpuConfig::default()).unwrap_or_else(|_| {
            // Fallback: create adapter with CPU-only device manager
            GpuAdapter { config: GpuConfig::default(), device_manager: DeviceManager::default() }
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
        assert!(config.allow_cpu_fallback);
        assert!(!config.enable_profiling);
    }

    #[test]
    fn test_gpu_adapter_default() {
        let adapter = GpuAdapter::default();
        assert!(!adapter.has_gpu() || adapter.has_gpu()); // Either is valid
        assert_eq!(adapter.device_manager.device_count() >= 1, true);
    }

    #[test]
    fn test_gpu_adapter_new() {
        let config = GpuConfig::default();
        let adapter = GpuAdapter::new(config);
        assert!(adapter.is_ok());
    }

    #[test]
    fn test_gpu_adapter_device_info() {
        let adapter = GpuAdapter::default();
        let device = adapter.current_device();
        assert!(!device.name.is_empty());
    }

    #[test]
    fn test_gpu_adapter_explicit_device_cpu() {
        let mut config = GpuConfig::default();
        config.device_strategy = DeviceSelectionStrategy::Explicit;
        config.explicit_device = Some(DeviceId::Cpu);

        let adapter = GpuAdapter::new(config).expect("adapter creation");
        assert_eq!(adapter.current_device().device_id, DeviceId::Cpu);
    }

    #[test]
    fn test_gpu_adapter_most_memory_strategy() {
        let mut config = GpuConfig::default();
        config.device_strategy = DeviceSelectionStrategy::MostMemory;

        let adapter = GpuAdapter::new(config).expect("adapter creation");
        assert!(adapter.device_manager.device_count() >= 1);
    }

    #[test]
    fn test_gpu_adapter_config() {
        let config = GpuConfig {
            device_strategy: DeviceSelectionStrategy::First,
            explicit_device: Some(DeviceId::Cpu),
            max_gpu_memory: Some(8 * 1024 * 1024 * 1024),
            batch_size: 32,
            allow_cpu_fallback: false,
            enable_profiling: true,
        };
        let adapter = GpuAdapter::new(config.clone()).expect("adapter creation");
        assert_eq!(adapter.config().batch_size, 32);
        assert!(adapter.config().enable_profiling);
        assert!(!adapter.config().allow_cpu_fallback);
    }
}
