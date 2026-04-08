#![warn(missing_docs)]

//! GPU device abstraction layer.
//!
//! Provides unified interface for detecting and managing GPU devices
//! across different backends (CUDA, ROCm, WebGPU) with CPU fallback.
//!
//! # Device Detection
//!
//! Device detection happens at runtime via platform-specific queries.
//! The library supports:
//! - CUDA (NVIDIA GPUs via NVIDIA CUDA Toolkit)
//! - ROCm (AMD GPUs via AMD ROCm)
//! - WebGPU (Browser/portable via WebGPU)
//! - CPU (Fallback for all platforms)
//!
//! # Device Selection
//!
//! Users can:
//! - Allow automatic detection (picks most capable available device)
//! - Explicitly select a device (CUDA device 0, ROCm device 1, etc.)
//! - Fall back to CPU if GPU unavailable
//!
//! # Thread Safety
//!
//! Device handles may be shared across threads, but actual kernel execution
//! is not thread-safe. Synchronization is application responsibility.

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
///
/// Provides metadata about a GPU device including compute capability,
/// memory size, and performance characteristics.
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

/// Device detection result. May fail if no devices available.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Available devices in order of detection
    pub devices: Vec<DeviceInfo>,
    /// Recommended device (best choice based on strategy)
    pub recommended: DeviceInfo,
}

/// Device detection and query trait.
///
/// Implemented by platform-specific device managers.
pub trait DeviceDetector: Send + Sync {
    /// Detect all available devices of this type.
    /// Returns empty vec if no devices found (not an error).
    fn detect_devices(&self) -> Result<Vec<DeviceInfo>, DeviceError>;

    /// Query specific device by index.
    fn get_device(&self, index: u32) -> Result<DeviceInfo, DeviceError>;

    /// Get count of available devices.
    fn device_count(&self) -> Result<u32, DeviceError>;
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
///
/// Singleton managing device detection and selection across the library.
/// Thread-safe access to device info and current device.
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

        // Default to first device
        let current_device = devices[0].device_id;

        Ok(DeviceManager { devices, current_device })
    }

    /// Detect all available devices across all backends.
    fn detect_all_devices() -> Result<Vec<DeviceInfo>, DeviceError> {
        let mut devices = Vec::new();

        // Try CUDA detection if feature enabled
        #[cfg(feature = "cuda")]
        {
            match Self::detect_cuda_devices() {
                Ok(cuda_devices) => devices.extend(cuda_devices),
                Err(_) => {} // CUDA not available, continue
            }
        }

        // Try ROCm detection if feature enabled
        #[cfg(feature = "rocm")]
        {
            match Self::detect_rocm_devices() {
                Ok(rocm_devices) => devices.extend(rocm_devices),
                Err(_) => {} // ROCm not available, continue
            }
        }

        // Try WebGPU detection if feature enabled
        #[cfg(feature = "webgpu")]
        {
            match Self::detect_webgpu_devices() {
                Ok(webgpu_devices) => devices.extend(webgpu_devices),
                Err(_) => {} // WebGPU not available, continue
            }
        }

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

    /// Detect CUDA devices (no-op if feature not enabled).
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices() -> Result<Vec<DeviceInfo>, DeviceError> {
        // TODO: Implement actual CUDA detection via CUDA Runtime API
        // This is a stub that would call cuDeviceGetCount, cuDeviceGet, cuDeviceGetAttribute, etc.
        Ok(vec![])
    }

    /// Stub for CUDA when feature is disabled.
    #[cfg(not(feature = "cuda"))]
    fn detect_cuda_devices() -> Result<Vec<DeviceInfo>, DeviceError> {
        Ok(vec![])
    }

    /// Detect ROCm devices (no-op if feature not enabled).
    #[cfg(feature = "rocm")]
    fn detect_rocm_devices() -> Result<Vec<DeviceInfo>, DeviceError> {
        // TODO: Implement actual ROCm detection via ROCm Runtime API
        // This would call hipGetDeviceCount, hipGetDeviceProperties, etc.
        Ok(vec![])
    }

    /// Stub for ROCm when feature is disabled.
    #[cfg(not(feature = "rocm"))]
    fn detect_rocm_devices() -> Result<Vec<DeviceInfo>, DeviceError> {
        Ok(vec![])
    }

    /// Detect WebGPU devices (no-op if feature not enabled).
    #[cfg(feature = "webgpu")]
    fn detect_webgpu_devices() -> Result<Vec<DeviceInfo>, DeviceError> {
        // TODO: Implement actual WebGPU detection
        // This would enumerate available WebGPU adapters
        Ok(vec![])
    }

    /// Stub for WebGPU when feature is disabled.
    #[cfg(not(feature = "webgpu"))]
    fn detect_webgpu_devices() -> Result<Vec<DeviceInfo>, DeviceError> {
        Ok(vec![])
    }

    /// Get estimated available system memory in bytes.
    fn cpu_available_memory() -> u64 {
        // Estimate: use 70% of system memory or 16GB, whichever is less
        // This prevents OOM from naive allocations
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
        // Fallback to CPU-only if detection fails
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
    fn test_gpu_backend_display() {
        assert_eq!(GpuBackend::Cuda.to_string(), "CUDA");
        assert_eq!(GpuBackend::Rocm.to_string(), "ROCm");
        assert_eq!(GpuBackend::WebGpu.to_string(), "WebGPU");
        assert_eq!(GpuBackend::Cpu.to_string(), "CPU");
    }

    #[test]
    fn test_device_manager_default() {
        let mgr = DeviceManager::default();
        assert_eq!(mgr.device_count(), 1);
        assert_eq!(mgr.current_device().device_id, DeviceId::Cpu);
        assert!(!mgr.has_gpu());
    }

    #[test]
    fn test_device_manager_new() {
        // Should not fail - at least CPU is always available
        let mgr = DeviceManager::new().expect("device manager creation");
        assert!(mgr.device_count() >= 1);
        assert_eq!(mgr.current_device().backend, GpuBackend::Cpu);
    }

    #[test]
    fn test_set_device_cpu() {
        let mut mgr = DeviceManager::default();
        mgr.set_device(DeviceId::Cpu).expect("setting CPU device");
        assert_eq!(mgr.current_device().device_id, DeviceId::Cpu);
    }

    #[test]
    fn test_set_device_not_found() {
        let mut mgr = DeviceManager::default();
        let result = mgr.set_device(DeviceId::Cuda(0));
        // Will fail if CUDA not available
        if result.is_err() {
            assert!(matches!(result.unwrap_err(), DeviceError::DeviceNotFound(_)));
        }
    }

    #[test]
    fn test_device_selection_strategy_default() {
        assert_eq!(DeviceSelectionStrategy::default(), DeviceSelectionStrategy::MostMemory);
    }

    #[test]
    fn test_device_info_display() {
        let info = DeviceInfo {
            device_id: DeviceId::Cpu,
            name: "Test CPU".to_string(),
            backend: GpuBackend::Cpu,
            compute_capability: None,
            total_memory: 16 * 1024 * 1024 * 1024,
            clock_rate_mhz: None,
            multiprocessor_count: None,
            max_threads_per_block: None,
        };
        let display = info.to_string();
        assert!(display.contains("Test CPU"));
        assert!(display.contains("16 GB"));
    }

    #[test]
    fn test_device_info_display_with_compute_capability() {
        let info = DeviceInfo {
            device_id: DeviceId::Cuda(0),
            name: "NVIDIA RTX 4090".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some((8, 9)),
            total_memory: 24 * 1024 * 1024 * 1024,
            clock_rate_mhz: Some(2520),
            multiprocessor_count: Some(128),
            max_threads_per_block: Some(1024),
        };
        let display = info.to_string();
        assert!(display.contains("RTX 4090"));
        assert!(display.contains("24 GB"));
        assert!(display.contains("Compute 8.9"));
    }

    #[test]
    fn test_select_by_strategy_cpu_only() {
        let mut mgr = DeviceManager::default();
        mgr.select_by_strategy(DeviceSelectionStrategy::First).expect("strategy");
        assert_eq!(mgr.current_device().backend, GpuBackend::Cpu);
    }

    #[test]
    fn test_select_by_strategy_most_memory() {
        let mut mgr = DeviceManager::default();
        mgr.select_by_strategy(DeviceSelectionStrategy::MostMemory).expect("strategy");
        assert_eq!(mgr.current_device().backend, GpuBackend::Cpu);
    }

    #[test]
    fn test_cpu_available_memory_reasonable() {
        let mem = DeviceManager::cpu_available_memory();
        assert!(mem > 0);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(mem, 16 * 1024 * 1024 * 1024);
        #[cfg(target_pointer_width = "32")]
        assert_eq!(mem, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_device_error_display() {
        let err = DeviceError::DeviceNotFound(5);
        assert!(err.to_string().contains("5"));

        let err = DeviceError::NoDevicesAvailable;
        assert!(err.to_string().contains("No GPU"));

        let err = DeviceError::QueryFailed("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_device_id_equality() {
        assert_eq!(DeviceId::Cuda(0), DeviceId::Cuda(0));
        assert_ne!(DeviceId::Cuda(0), DeviceId::Cuda(1));
        assert_ne!(DeviceId::Cuda(0), DeviceId::Rocm(0));
    }

    #[test]
    fn test_get_device() {
        let mgr = DeviceManager::default();
        let cpu_info = mgr.get_device(DeviceId::Cpu);
        assert!(cpu_info.is_some());
        assert_eq!(cpu_info.unwrap().backend, GpuBackend::Cpu);

        let none_info = mgr.get_device(DeviceId::Cuda(99));
        assert!(none_info.is_none());
    }
}
