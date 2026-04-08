#![warn(missing_docs)]

//! Safe Rust wrappers for CUDA kernel execution.
//!
//! Provides FFI bindings to CUDA kernels with error handling, memory management,
//! and synchronization primitives. All unsafe code is confined to this module.

use std::ffi::CString;
use std::time::{Duration, Instant};

/// CUDA device handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceHandle {
    /// Device ordinal (0 = first GPU)
    pub device_id: u32,
}

/// CUDA memory allocation handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaMemoryHandle {
    /// Pointer to device memory (opaque to Rust)
    device_ptr: *mut std::ffi::c_void,
    /// Allocation size in bytes
    pub size: u64,
}

impl CudaMemoryHandle {
    /// Create a memory handle from device pointer and size.
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid GPU memory allocated via cudaMalloc.
    pub unsafe fn new(device_ptr: *mut std::ffi::c_void, size: u64) -> Self {
        CudaMemoryHandle { device_ptr, size }
    }

    /// Get the raw device pointer (for FFI use only).
    pub fn as_mut_ptr(&mut self) -> *mut std::ffi::c_void {
        self.device_ptr
    }

    /// Get the raw device pointer (for FFI use only).
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.device_ptr
    }
}

/// CUDA stream for async operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStream {
    /// Stream handle (opaque to Rust)
    stream_id: u32,
}

impl CudaStream {
    /// Create a new CUDA stream.
    pub fn new(stream_id: u32) -> Self {
        CudaStream { stream_id }
    }

    /// Get the stream ID for FFI use.
    pub fn id(&self) -> u32 {
        self.stream_id
    }
}

/// CUDA error type.
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Device not found or invalid
    DeviceNotFound(u32),
    /// Device initialization failed
    InitializationFailed(String),
    /// CUDA API call returned an error code
    ApiError(String),
    /// Memory allocation failed
    MemoryAllocationFailed {
        /// Requested size in bytes
        requested_size: u64,
        /// Available memory in bytes
        available_memory: u64,
    },
    /// Memory transfer failed
    MemoryTransferFailed(String),
    /// Kernel launch failed
    KernelLaunchFailed {
        /// Kernel name
        kernel_name: String,
        /// Error description
        reason: String,
    },
    /// Synchronization failed
    SynchronizationFailed(String),
    /// Invalid kernel configuration
    InvalidKernelConfig(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceNotFound(id) => write!(f, "CUDA device {} not found", id),
            CudaError::InitializationFailed(msg) => {
                write!(f, "CUDA initialization failed: {}", msg)
            }
            CudaError::ApiError(msg) => write!(f, "CUDA API error: {}", msg),
            CudaError::MemoryAllocationFailed { requested_size, available_memory } => {
                write!(
                    f,
                    "CUDA memory allocation failed: requested {} bytes, {} available",
                    requested_size, available_memory
                )
            }
            CudaError::MemoryTransferFailed(msg) => {
                write!(f, "CUDA memory transfer failed: {}", msg)
            }
            CudaError::KernelLaunchFailed { kernel_name, reason } => {
                write!(f, "CUDA kernel '{}' launch failed: {}", kernel_name, reason)
            }
            CudaError::SynchronizationFailed(msg) => {
                write!(f, "CUDA synchronization failed: {}", msg)
            }
            CudaError::InvalidKernelConfig(msg) => write!(f, "Invalid CUDA kernel config: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}

/// Result type for CUDA operations.
pub type CudaResult<T> = Result<T, CudaError>;

/// CUDA device management interface.
pub struct CudaDevice {
    handle: CudaDeviceHandle,
    properties: DeviceProperties,
}

/// CUDA device properties (query results).
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name
    pub name: String,
    /// Total global memory in bytes
    pub total_global_mem: u64,
    /// Warp size (always 32 for NVIDIA)
    pub warp_size: u32,
    /// Max threads per block
    pub max_threads_per_block: u32,
    /// Number of multiprocessors
    pub multiprocessor_count: u32,
    /// Compute capability major version
    pub compute_capability_major: u32,
    /// Compute capability minor version
    pub compute_capability_minor: u32,
}

impl CudaDevice {
    /// Initialize CUDA and get the device.
    pub fn new(device_id: u32) -> CudaResult<Self> {
        // TODO: Implement actual CUDA initialization
        // This is a stub that returns success with default properties

        let properties = DeviceProperties {
            name: format!("CUDA Device {}", device_id),
            total_global_mem: 12 * 1024 * 1024 * 1024, // 12 GB
            warp_size: 32,
            max_threads_per_block: 1024,
            multiprocessor_count: 128,
            compute_capability_major: 8,
            compute_capability_minor: 6,
        };

        Ok(CudaDevice { handle: CudaDeviceHandle { device_id }, properties })
    }

    /// Get device handle.
    pub fn handle(&self) -> CudaDeviceHandle {
        self.handle
    }

    /// Get device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Allocate device memory.
    pub fn allocate(&self, size: u64) -> CudaResult<CudaMemoryHandle> {
        if size == 0 {
            return Err(CudaError::InvalidKernelConfig("Cannot allocate 0 bytes".to_string()));
        }

        // Check if allocation would exceed device memory
        if size > self.properties.total_global_mem {
            return Err(CudaError::MemoryAllocationFailed {
                requested_size: size,
                available_memory: self.properties.total_global_mem,
            });
        }

        // TODO: Implement actual cudaMalloc call
        // This is a stub that returns a dummy pointer
        unsafe {
            let ptr = (size as usize) as *mut std::ffi::c_void;
            Ok(CudaMemoryHandle::new(ptr, size))
        }
    }

    /// Deallocate device memory.
    pub fn deallocate(&self, _mem: CudaMemoryHandle) -> CudaResult<()> {
        // TODO: Implement actual cudaFree call
        Ok(())
    }

    /// Copy data from host to device.
    pub fn copy_host_to_device(
        &self,
        host_data: &[f64],
        device_mem: &mut CudaMemoryHandle,
    ) -> CudaResult<Duration> {
        let start = Instant::now();

        let data_size = std::mem::size_of_val(host_data) as u64;
        if data_size > device_mem.size {
            return Err(CudaError::MemoryTransferFailed(format!(
                "Data size {} exceeds device memory {}",
                data_size, device_mem.size
            )));
        }

        // TODO: Implement actual cudaMemcpy (HostToDevice)
        // This is a stub operation
        let _src_ptr = host_data.as_ptr() as *const std::ffi::c_void;
        let _dst_ptr = device_mem.as_mut_ptr();
        let _copy_size = data_size;

        Ok(start.elapsed())
    }

    /// Copy data from device to host.
    pub fn copy_device_to_host(
        &self,
        device_mem: &CudaMemoryHandle,
        host_data: &mut [f64],
    ) -> CudaResult<Duration> {
        let start = Instant::now();

        let data_size = std::mem::size_of_val(host_data) as u64;
        if data_size > device_mem.size {
            return Err(CudaError::MemoryTransferFailed(format!(
                "Data size {} exceeds device memory {}",
                data_size, device_mem.size
            )));
        }

        // TODO: Implement actual cudaMemcpy (DeviceToHost)
        // This is a stub operation
        let _src_ptr = device_mem.as_ptr();
        let _dst_ptr = host_data.as_mut_ptr() as *mut std::ffi::c_void;
        let _copy_size = data_size;

        Ok(start.elapsed())
    }

    /// Synchronize with device (wait for all operations to complete).
    pub fn synchronize(&self) -> CudaResult<()> {
        // TODO: Implement actual cudaDeviceSynchronize
        Ok(())
    }
}

/// CUDA kernel configuration for launch.
#[derive(Debug, Clone, Copy)]
pub struct KernelLaunchConfig {
    /// Number of blocks in grid (x dimension)
    pub grid_x: u32,
    /// Number of blocks in grid (y dimension)
    pub grid_y: u32,
    /// Number of threads per block (x dimension)
    pub block_x: u32,
    /// Number of threads per block (y dimension)
    pub block_y: u32,
    /// Shared memory size in bytes
    pub shared_memory: u32,
    /// Stream to launch on
    pub stream: CudaStream,
}

impl Default for KernelLaunchConfig {
    fn default() -> Self {
        KernelLaunchConfig {
            grid_x: 1,
            grid_y: 1,
            block_x: 256,
            block_y: 1,
            shared_memory: 0,
            stream: CudaStream::new(0),
        }
    }
}

impl KernelLaunchConfig {
    /// Validate kernel configuration against device properties.
    pub fn validate(&self, device: &CudaDevice) -> CudaResult<()> {
        if self.grid_x == 0 || self.block_x == 0 {
            return Err(CudaError::InvalidKernelConfig(
                "Grid and block dimensions must be > 0".to_string(),
            ));
        }

        let total_threads = (self.block_x as u64) * (self.block_y as u64);
        if total_threads > device.properties.max_threads_per_block as u64 {
            return Err(CudaError::InvalidKernelConfig(format!(
                "Total threads {} exceeds max {}",
                total_threads, device.properties.max_threads_per_block
            )));
        }

        Ok(())
    }

    /// Create a 1D kernel config.
    pub fn grid_1d(grid_size: u32, block_size: u32) -> Self {
        KernelLaunchConfig { grid_x: grid_size, block_x: block_size, ..Default::default() }
    }

    /// Create a 2D kernel config.
    pub fn grid_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        KernelLaunchConfig { grid_x, grid_y, block_x, block_y, ..Default::default() }
    }
}

/// Kernel launch result with timing information.
#[derive(Debug, Clone)]
pub struct KernelLaunchResult {
    /// Whether launch succeeded
    pub success: bool,
    /// Kernel execution time (if synchronized)
    pub execution_time: Option<Duration>,
    /// Error message if failed
    pub error: Option<String>,
}

/// CUDA kernel launcher (orchestrates kernel execution).
pub struct CudaKernelLauncher {
    device: CudaDevice,
}

impl CudaKernelLauncher {
    /// Create a new kernel launcher for the given device.
    pub fn new(device_id: u32) -> CudaResult<Self> {
        let device = CudaDevice::new(device_id)?;
        Ok(CudaKernelLauncher { device })
    }

    /// Get reference to underlying device.
    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Launch generate_noise kernel.
    pub fn launch_generate_noise(
        &self,
        _num_samples: u32,
        _seed: u64,
        _output_device_mem: &mut CudaMemoryHandle,
        config: &KernelLaunchConfig,
    ) -> CudaResult<KernelLaunchResult> {
        config.validate(&self.device)?;

        // TODO: Implement actual kernel launch
        // This is a stub that succeeds
        Ok(KernelLaunchResult {
            success: true,
            execution_time: None, // Async launch doesn't measure time
            error: None,
        })
    }

    /// Launch add_signal kernel.
    pub fn launch_add_signal(
        &self,
        _num_samples: u32,
        _signal_device_mem: &CudaMemoryHandle,
        _noise_device_mem: &CudaMemoryHandle,
        _output_device_mem: &mut CudaMemoryHandle,
        config: &KernelLaunchConfig,
    ) -> CudaResult<KernelLaunchResult> {
        config.validate(&self.device)?;

        // TODO: Implement actual kernel launch
        Ok(KernelLaunchResult { success: true, execution_time: None, error: None })
    }

    /// Launch extrema finding kernel.
    pub fn launch_find_extrema(
        &self,
        _num_samples: u32,
        _signal_device_mem: &CudaMemoryHandle,
        _max_indices_device_mem: &mut CudaMemoryHandle,
        _min_indices_device_mem: &mut CudaMemoryHandle,
        _max_count_device_mem: &mut CudaMemoryHandle,
        _min_count_device_mem: &mut CudaMemoryHandle,
        config: &KernelLaunchConfig,
    ) -> CudaResult<KernelLaunchResult> {
        config.validate(&self.device)?;

        // TODO: Implement actual kernel launch
        Ok(KernelLaunchResult { success: true, execution_time: None, error: None })
    }

    /// Synchronize device and return execution time.
    pub fn synchronize(&self) -> CudaResult<Duration> {
        let start = Instant::now();
        self.device.synchronize()?;
        Ok(start.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_device_handle() {
        let handle = CudaDeviceHandle { device_id: 0 };
        assert_eq!(handle.device_id, 0);
    }

    #[test]
    fn test_cuda_memory_handle() {
        unsafe {
            let ptr = 0x1000 as *mut std::ffi::c_void;
            let mem = CudaMemoryHandle::new(ptr, 1024);
            assert_eq!(mem.size, 1024);
        }
    }

    #[test]
    fn test_cuda_stream() {
        let stream = CudaStream::new(0);
        assert_eq!(stream.id(), 0);
    }

    #[test]
    fn test_kernel_launch_config_default() {
        let config = KernelLaunchConfig::default();
        assert_eq!(config.grid_x, 1);
        assert_eq!(config.block_x, 256);
        assert_eq!(config.block_y, 1);
    }

    #[test]
    fn test_kernel_launch_config_1d() {
        let config = KernelLaunchConfig::grid_1d(128, 256);
        assert_eq!(config.grid_x, 128);
        assert_eq!(config.block_x, 256);
    }

    #[test]
    fn test_kernel_launch_config_2d() {
        let config = KernelLaunchConfig::grid_2d(64, 64, 16, 16);
        assert_eq!(config.grid_x, 64);
        assert_eq!(config.grid_y, 64);
        assert_eq!(config.block_x, 16);
        assert_eq!(config.block_y, 16);
    }

    #[test]
    fn test_cuda_device_new() {
        let device = CudaDevice::new(0);
        assert!(device.is_ok());
        let dev = device.unwrap();
        assert_eq!(dev.handle().device_id, 0);
    }

    #[test]
    fn test_cuda_device_properties() {
        let device = CudaDevice::new(0).unwrap();
        let props = device.properties();
        assert!(props.total_global_mem > 0);
        assert_eq!(props.warp_size, 32);
    }

    #[test]
    fn test_cuda_device_allocate() {
        let device = CudaDevice::new(0).unwrap();
        let result = device.allocate(1024);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cuda_device_allocate_zero() {
        let device = CudaDevice::new(0).unwrap();
        let result = device.allocate(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cuda_kernel_launcher_new() {
        let launcher = CudaKernelLauncher::new(0);
        assert!(launcher.is_ok());
    }

    #[test]
    fn test_cuda_error_display() {
        let err = CudaError::DeviceNotFound(0);
        assert_eq!(err.to_string(), "CUDA device 0 not found");
    }
}
