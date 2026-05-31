#![warn(missing_docs)]

//! GPU kernel execution abstraction.
//!
//! Provides abstraction over GPU kernel launches, data transfers,
//! and synchronization across different backends.

use crate::adapters::gpu::AllocationId;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// GPU kernel launch configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KernelConfig {
    /// Grid dimension X: number of thread blocks along the X axis.
    pub grid_x: u32,
    /// Grid dimension Y: number of thread blocks along the Y axis.
    pub grid_y: u32,
    /// Grid dimension Z: number of thread blocks along the Z axis.
    pub grid_z: u32,
    /// Block dimension X: number of threads per block along the X axis.
    pub block_x: u32,
    /// Block dimension Y: number of threads per block along the Y axis.
    pub block_y: u32,
    /// Block dimension Z: number of threads per block along the Z axis.
    pub block_z: u32,
    /// Shared memory size in bytes to allocate per block.
    pub shared_memory: usize,
    /// Stream ID for async kernel execution; 0 uses the default stream.
    pub stream_id: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            grid_x: 1,
            grid_y: 1,
            grid_z: 1,
            block_x: 256,
            block_y: 1,
            block_z: 1,
            shared_memory: 0,
            stream_id: 0,
        }
    }
}

impl KernelConfig {
    /// Create a 1D grid/block configuration.
    pub fn grid_1d(grid_size: u32, block_size: u32) -> Self {
        Self { grid_x: grid_size, block_x: block_size, ..Default::default() }
    }

    /// Create a 2D grid/block configuration.
    pub fn grid_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self { grid_x, grid_y, block_x, block_y, ..Default::default() }
    }

    /// Validate kernel config.
    pub fn validate(&self) -> Result<(), String> {
        if self.grid_x == 0 || self.block_x == 0 {
            return Err("Grid and block dimensions must be > 0".to_string());
        }
        let total_threads = (self.block_x as u64) * (self.block_y as u64) * (self.block_z as u64);
        if total_threads > 1024 {
            return Err(format!("Total threads per block {} exceeds max 1024", total_threads));
        }
        Ok(())
    }
}

/// Result of a kernel launch.
#[derive(Debug, Clone)]
pub struct KernelLaunchResult {
    /// Whether the launch succeeded
    pub success: bool,
    /// Execution time (None if async)
    pub execution_time: Option<Duration>,
    /// Kernel name (for debugging)
    pub kernel_name: String,
    /// Error message if failed
    pub error: Option<String>,
}

/// Data transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferDirection {
    /// Host to Device
    HostToDevice,
    /// Device to Host
    DeviceToHost,
    /// Device to Device
    DeviceToDevice,
}

/// Data transfer operation.
#[derive(Debug, Clone)]
pub struct DataTransfer {
    /// Transfer direction
    pub direction: TransferDirection,
    /// Size in bytes
    pub size: u64,
    /// Source allocation (for device transfers)
    pub src_alloc: Option<AllocationId>,
    /// Destination allocation
    pub dst_alloc: Option<AllocationId>,
    /// Async transfer (non-blocking)
    pub async_transfer: bool,
}

/// Kernel launcher trait for platform-independent kernel execution.
pub trait KernelLauncher: Send + Sync {
    /// Launch a kernel with given configuration.
    fn launch_kernel(
        &self,
        kernel_name: &str,
        config: &KernelConfig,
    ) -> Result<KernelLaunchResult, String>;

    /// Transfer data between host and device.
    fn transfer_data(&self, transfer: &DataTransfer) -> Result<Duration, String>;

    /// Synchronize with GPU (wait for all pending operations).
    fn synchronize(&self) -> Result<(), String>;

    /// Get kernel execution time estimate (for performance optimization).
    fn estimated_kernel_time(&self, kernel_name: &str) -> Option<Duration>;
}

/// Stub implementation of KernelLauncher for CPU fallback.
pub struct CpuKernelLauncher;

impl KernelLauncher for CpuKernelLauncher {
    fn launch_kernel(
        &self,
        kernel_name: &str,
        config: &KernelConfig,
    ) -> Result<KernelLaunchResult, String> {
        config.validate()?;

        Ok(KernelLaunchResult {
            success: true,
            execution_time: Some(Duration::from_millis(1)),
            kernel_name: kernel_name.to_string(),
            error: None,
        })
    }

    fn transfer_data(&self, _transfer: &DataTransfer) -> Result<Duration, String> {
        // CPU transfers are instantaneous
        Ok(Duration::from_nanos(0))
    }

    fn synchronize(&self) -> Result<(), String> {
        // CPU has no async ops
        Ok(())
    }

    fn estimated_kernel_time(&self, _kernel_name: &str) -> Option<Duration> {
        Some(Duration::from_millis(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_config_default() {
        let config = KernelConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.grid_x, 1);
        assert_eq!(config.block_x, 256);
    }

    #[test]
    fn test_kernel_config_1d() {
        let config = KernelConfig::grid_1d(128, 256);
        assert!(config.validate().is_ok());
        assert_eq!(config.grid_x, 128);
        assert_eq!(config.block_x, 256);
    }

    #[test]
    fn test_kernel_config_2d() {
        let config = KernelConfig::grid_2d(64, 64, 16, 16);
        assert!(config.validate().is_ok());
        assert_eq!(config.grid_x, 64);
        assert_eq!(config.grid_y, 64);
        assert_eq!(config.block_x, 16);
        assert_eq!(config.block_y, 16);
    }

    #[test]
    fn test_kernel_config_validation_grid_zero() {
        let config = KernelConfig { grid_x: 0, ..Default::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_kernel_config_validation_block_zero() {
        let config = KernelConfig { block_x: 0, ..Default::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_kernel_config_validation_too_many_threads() {
        let config = KernelConfig { block_x: 512, block_y: 4, ..Default::default() };
        // 512 * 4 = 2048 > 1024
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_transfer_direction() {
        assert_eq!(TransferDirection::HostToDevice, TransferDirection::HostToDevice);
        assert_ne!(TransferDirection::HostToDevice, TransferDirection::DeviceToHost);
    }

    #[test]
    fn test_cpu_kernel_launcher() {
        let launcher = CpuKernelLauncher;
        let config = KernelConfig::default();

        let result = launcher.launch_kernel("test_kernel", &config).expect("launch");

        assert!(result.success);
        assert_eq!(result.kernel_name, "test_kernel");
        assert!(result.execution_time.is_some());
        assert!(result.error.is_none());
    }

    #[test]
    fn test_cpu_kernel_launcher_sync() {
        let launcher = CpuKernelLauncher;
        let result = launcher.synchronize();
        assert!(result.is_ok());
    }

    #[test]
    fn test_cpu_kernel_launcher_transfer() {
        let launcher = CpuKernelLauncher;
        let transfer = DataTransfer {
            direction: TransferDirection::HostToDevice,
            size: 1024,
            src_alloc: None,
            dst_alloc: None,
            async_transfer: false,
        };

        let result = launcher.transfer_data(&transfer);
        assert!(result.is_ok());
    }
}
