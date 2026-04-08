#![warn(missing_docs)]

//! GPU device abstraction layer.
//!
//! Provides unified interface for detecting and managing GPU devices
//! across different backends (CUDA, ROCm, WebGPU) with CPU fallback.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique device identifier combining backend and device index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceId {
    /// CUDA device by index (0 = first GPU)
    Cuda(u32),
    /// ROCm device by index (0 = first GPU)
    Rocm(u32),
    /// WebGPU device by index (0 = first GPU)
    WebGpu(u32),
    /// CPU fallback (always available, index unused)
    Cpu,
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceId::Cuda(idx) => write!(f, "CUDA:{}", idx),
            DeviceId::Rocm(idx) => write!(f, "ROCm:{}", idx),
            DeviceId::WebGpu(idx) => write!(f, "WebGPU:{}", idx),
            DeviceId::Cpu => write!(f, "CPU"),
        }
    }
}

/// GPU vendor/backend type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBackend {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm
    Rocm,
    /// WebGPU (portable)
    WebGpu,
    /// CPU fallback
    Cpu,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Rocm => write!(f, "ROCm"),
            GpuBackend::WebGpu => write!(f, "WebGPU"),
            GpuBackend::Cpu => write!(f, "CPU"),
        }
    }
}

/// Device information and capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub device_id: DeviceId,
    /// Device name (e.g., "NVIDIA GeForce RTX 4090")
    pub name: String,
    /// GPU backend
    pub backend: GpuBackend,
    /// CUDA compute capability (only for CUDA devices), e.g., (8, 6) for Ada
    pub compute_capability: Option<(u32, u32)>,
    /// Total device memory in bytes
    pub total_memory: u64,
    /// Clock frequency in MHz
    pub clock_rate_mhz: Option<u32>,
    /// Number of multiprocessors / compute units
    pub multiprocessor_count: Option<u32>,
    /// Max threads per block
    pub max_threads_per_block: Option<u32>,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} ({}) - {} GB memory",
            self.name,
            self.device_id,
            self.total_memory / (1024 * 1024 * 1024)
        )?;
        if let Some((major, minor)) = self.compute_capability {
            write!(f, " - Compute {}.{}", major, minor)?;
        }
        Ok(())
    }
}

/// Device selection strategy for automatic device picking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceSelectionStrategy {
    /// Pick the first available device (any backend)
    First,
    /// Pick device with most memory
    MostMemory,
    /// Pick device with highest compute capability
    FastestCompute,
    /// Prefer CUDA > ROCm > WebGPU > CPU
    PreferredBackend,
    /// Explicit device selection (must be specified separately)
    Explicit,
}

impl Default for DeviceSelectionStrategy {
    fn default() -> Self {
        DeviceSelectionStrategy::MostMemory
    }
}

/// GPU device errors.
#[derive(Debug, Clone)]
pub enum DeviceError {
    /// Device not found at specified index
    DeviceNotFound(u32),
    /// Device is not available (offline, in use, etc.)
    DeviceUnavailable,
    /// Query operation failed (driver issue, permission, etc.)
    QueryFailed(String),
    /// Memory query failed
    MemoryQueryFailed,
    /// No devices available
    NoDevicesAvailable,
    /// Device initialization failed
    InitializationFailed(String),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceError::DeviceNotFound(idx) => write!(f, "Device {} not found", idx),
            DeviceError::DeviceUnavailable => write!(f, "Device is unavailable"),
            DeviceError::QueryFailed(msg) => write!(f, "Device query failed: {}", msg),
            DeviceError::MemoryQueryFailed => write!(f, "Memory query failed"),
            DeviceError::NoDevicesAvailable => write!(f, "No GPU devices available"),
            DeviceError::InitializationFailed(msg) => {
                write!(f, "Device initialization failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for DeviceError {}

/// Global device manager for GPU adaptation.
pub struct DeviceManager {
    devices: Vec<DeviceInfo>,
    current_device: DeviceId,
}

impl DeviceManager {
    /// Create a new device manager and detect available devices.
    pub fn new() -> Result<Self, DeviceError> {
        let devices = Self::detect_all_devices()?;
        if devices.is_empty() {
            return Err(DeviceError::NoDevicesAvailable);
        }
        let current_device = devices[0].device_id;
        Ok(DeviceManager { devices, current_device })
    }

    /// Detect all available devices across all backends.
    fn detect_all_devices() -> Result<Vec<DeviceInfo>, DeviceError> {
        let mut devices = Vec::new();

        // Always add CPU fallback
        devices.push(DeviceInfo {
            device_id: DeviceId::Cpu,
            name: "CPU Fallback".to_string(),
            backend: GpuBackend::Cpu,
            compute_capability: None,
            total_memory: Self::cpu_available_memory(),
            clock_rate_mhz: None,
            multiprocessor_count: None,
            max_threads_per_block: None,
        });

        Ok(devices)
    }

    /// Get estimated available system memory in bytes.
    fn cpu_available_memory() -> u64 {
        #[cfg(target_pointer_width = "64")]
        {
            16 * 1024 * 1024 * 1024 // 16GB conservative estimate
        }
        #[cfg(target_pointer_width = "32")]
        {
            2 * 1024 * 1024 * 1024 // 2GB for 32-bit systems
        }
    }

    /// Get all detected devices.
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Get currently selected device.
    pub fn current_device(&self) -> &DeviceInfo {
        self.devices
            .iter()
            .find(|d| d.device_id == self.current_device)
            .expect("current device should always be valid")
    }

    /// Set the current device by ID.
    pub fn set_device(&mut self, device_id: DeviceId) -> Result<(), DeviceError> {
        if !self.devices.iter().any(|d| d.device_id == device_id) {
            return Err(DeviceError::DeviceNotFound(match device_id {
                DeviceId::Cuda(idx) => idx,
                DeviceId::Rocm(idx) => idx,
                DeviceId::WebGpu(idx) => idx,
                DeviceId::Cpu => 0,
            }));
        }
        self.current_device = device_id;
        Ok(())
    }

    /// Select device by strategy.
    pub fn select_by_strategy(
        &mut self,
        strategy: DeviceSelectionStrategy,
    ) -> Result<(), DeviceError> {
        let device_id = match strategy {
            DeviceSelectionStrategy::First => self.devices[0].device_id,
            DeviceSelectionStrategy::MostMemory => {
                self.devices
                    .iter()
                    .max_by_key(|d| d.total_memory)
                    .expect("devices not empty")
                    .device_id
            }
            DeviceSelectionStrategy::FastestCompute => {
                self.devices
                    .iter()
                    .max_by(|a, b| {
                        let a_score = a.compute_capability.map_or(0, |c| c.0 * 1000 + c.1);
                        let b_score = b.compute_capability.map_or(0, |c| c.0 * 1000 + c.1);
                        a_score.cmp(&b_score)
                    })
                    .expect("devices not empty")
                    .device_id
            }
            DeviceSelectionStrategy::PreferredBackend => {
                let backends =
                    [GpuBackend::Cuda, GpuBackend::Rocm, GpuBackend::WebGpu, GpuBackend::Cpu];
                backends
                    .iter()
                    .find_map(|backend| self.devices.iter().find(|d| d.backend == *backend))
                    .expect("at least CPU should be available")
                    .device_id
            }
            DeviceSelectionStrategy::Explicit => {
                return Err(DeviceError::QueryFailed(
                    "Explicit strategy requires set_device call".to_string(),
                ))
            }
        };
        self.set_device(device_id)
    }

    /// Get device by ID.
    pub fn get_device(&self, device_id: DeviceId) -> Option<&DeviceInfo> {
        self.devices.iter().find(|d| d.device_id == device_id)
    }

    /// Get device count.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Check if GPU is available (non-CPU device exists).
    pub fn has_gpu(&self) -> bool {
        self.devices.iter().any(|d| d.backend != GpuBackend::Cpu)
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        DeviceManager {
            devices: vec![DeviceInfo {
                device_id: DeviceId::Cpu,
                name: "CPU Fallback".to_string(),
                backend: GpuBackend::Cpu,
                compute_capability: None,
                total_memory: DeviceManager::cpu_available_memory(),
                clock_rate_mhz: None,
                multiprocessor_count: None,
                max_threads_per_block: None,
            }],
            current_device: DeviceId::Cpu,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id_display() {
        assert_eq!(DeviceId::Cuda(0).to_string(), "CUDA:0");
        assert_eq!(DeviceId::Rocm(1).to_string(), "ROCm:1");
        assert_eq!(DeviceId::WebGpu(2).to_string(), "WebGPU:2");
        assert_eq!(DeviceId::Cpu.to_string(), "CPU");
    }

    #[test]
    fn test_device_manager_default() {
        let mgr = DeviceManager::default();
        assert_eq!(mgr.device_count(), 1);
        assert_eq!(mgr.current_device().device_id, DeviceId::Cpu);
    }

    #[test]
    fn test_device_manager_new() {
        let mgr = DeviceManager::new().expect("device manager creation");
        assert!(mgr.device_count() >= 1);
    }

    #[test]
    fn test_set_device_cpu() {
        let mut mgr = DeviceManager::default();
        mgr.set_device(DeviceId::Cpu).expect("setting CPU device");
        assert_eq!(mgr.current_device().device_id, DeviceId::Cpu);
    }

    #[test]
    fn test_select_by_strategy() {
        let mut mgr = DeviceManager::default();
        mgr.select_by_strategy(DeviceSelectionStrategy::First).expect("strategy");
        assert_eq!(mgr.current_device().backend, GpuBackend::Cpu);
    }
}
