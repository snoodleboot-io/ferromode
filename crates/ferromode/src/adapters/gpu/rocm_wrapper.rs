#![warn(missing_docs)]

//! Safe Rust wrappers for HIP (AMD ROCm) kernel execution.
//!
//! Provides FFI bindings to HIP kernels with error handling, memory management,
//! and synchronization primitives. All unsafe code is confined to this module.
//!
//! HIP is AMD's heterogeneous-compute interface that provides a subset of the
//! CUDA API translated to work on AMD GPUs. This provides a familiar interface
//! for CUDA developers while supporting AMD hardware.

/// HIP device handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipDeviceHandle {
    /// Device ordinal (0 = first GPU)
    pub device_id: u32,
}

/// HIP memory allocation handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HipMemoryHandle {
    /// Pointer to device memory (opaque to Rust)
    device_ptr: *mut std::ffi::c_void,
    /// Allocation size in bytes
    pub size: u64,
}

impl HipMemoryHandle {
    /// Create a memory handle from device pointer and size.
    ///
    /// # Safety
    /// The caller must ensure the pointer is valid GPU memory allocated via hipMalloc.
    pub unsafe fn new(device_ptr: *mut std::ffi::c_void, size: u64) -> Self {
        HipMemoryHandle { device_ptr, size }
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

/// HIP stream for async operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HipStream {
    /// Stream handle (opaque to Rust)
    stream_id: u32,
}

impl HipStream {
    /// Create a new HIP stream.
    pub fn new(stream_id: u32) -> Self {
        HipStream { stream_id }
    }

    /// Get the stream ID for FFI use.
    pub fn id(&self) -> u32 {
        self.stream_id
    }
}

/// HIP error type.
#[derive(Debug, Clone)]
pub enum HipError {
    /// Device not found or invalid
    DeviceNotFound(u32),
    /// Device initialization failed
    InitializationFailed(String),
    /// HIP API call returned an error code
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

impl std::fmt::Display for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HipError::DeviceNotFound(id) => write!(f, "HIP device {} not found", id),
            HipError::InitializationFailed(msg) => {
                write!(f, "HIP initialization failed: {}", msg)
            }
            HipError::ApiError(msg) => write!(f, "HIP API error: {}", msg),
            HipError::MemoryAllocationFailed { requested_size, available_memory } => {
                write!(
                    f,
                    "HIP memory allocation failed: requested {} bytes, {} available",
                    requested_size, available_memory
                )
            }
            HipError::MemoryTransferFailed(msg) => {
                write!(f, "HIP memory transfer failed: {}", msg)
            }
            HipError::KernelLaunchFailed { kernel_name, reason } => {
                write!(f, "HIP kernel '{}' launch failed: {}", kernel_name, reason)
            }
            HipError::SynchronizationFailed(msg) => {
                write!(f, "HIP synchronization failed: {}", msg)
            }
            HipError::InvalidKernelConfig(msg) => write!(f, "Invalid HIP kernel config: {}", msg),
        }
    }
}

impl std::error::Error for HipError {}

/// Result type for HIP operations.
pub type HipResult<T> = Result<T, HipError>;

/// HIP device properties (query results).
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name
    pub name: String,
    /// Total global memory in bytes
    pub total_global_mem: u64,
    /// Wavesize (32 or 64 depending on architecture)
    pub wavesize: u32,
    /// Max threads per block
    pub max_threads_per_block: u32,
    /// Number of compute units
    pub compute_units: u32,
    /// GPU architecture family (e.g., "gfx906", "gfx90a", "gfx1030")
    pub gcn_arch: String,
}

/// HIP device management interface.
pub struct HipDevice {
    handle: HipDeviceHandle,
    properties: DeviceProperties,
}

impl HipDevice {
    /// Get device properties from HIP API.
    ///
    /// Queries the device using real HIP runtime API calls.
    /// This is the interface to the actual AMD GPU hardware.
    pub fn new_from_rocm(device_id: u32) -> HipResult<Self> {
        // Validate device exists using hipGetDeviceCount
        let mut device_count: i32 = 0;
        unsafe {
            let result = hipGetDeviceCount(&mut device_count);
            if result != 0 {
                return Err(HipError::InitializationFailed(
                    "Failed to get device count from HIP".to_string(),
                ));
            }
        }

        if device_id >= device_count as u32 {
            return Err(HipError::DeviceNotFound(device_id));
        }

        // Set current device
        unsafe {
            let result = hipSetDevice(device_id as i32);
            if result != 0 {
                return Err(HipError::InitializationFailed(format!(
                    "Failed to set HIP device {}: error code {}",
                    device_id, result
                )));
            }
        }

        // Query device properties
        let properties = unsafe {
            let mut props: hipDeviceProp_t = std::mem::zeroed();
            let result = hipGetDeviceProperties(&mut props, device_id as i32);
            if result != 0 {
                return Err(HipError::ApiError(format!(
                    "hipGetDeviceProperties failed with code {}",
                    result
                )));
            }

            // Extract GCN architecture from device name
            // Typical names: "gfx906", "gfx90a", "gfx1030", etc.
            let name_cstr = std::ffi::CStr::from_ptr(props.gcnArchName.as_ptr());
            let name = name_cstr.to_str().unwrap_or("unknown").to_string();

            let gcn_arch = name.clone();

            DeviceProperties {
                name,
                total_global_mem: props.totalGlobalMem,
                wavesize: props.waveSize,
                max_threads_per_block: props.maxThreadsPerBlock as u32,
                compute_units: props.multiProcessorCount as u32,
                gcn_arch,
            }
        };

        Ok(HipDevice { handle: HipDeviceHandle { device_id }, properties })
    }

    /// Get device handle.
    pub fn handle(&self) -> HipDeviceHandle {
        self.handle
    }

    /// Get device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Allocate device memory.
    ///
    /// Allocates memory on the GPU device using hipMalloc.
    pub fn malloc(&self, size: u64) -> HipResult<HipMemoryHandle> {
        if size == 0 {
            return Err(HipError::InvalidKernelConfig("Cannot allocate 0 bytes".to_string()));
        }

        unsafe {
            let mut device_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let result = hipMalloc(&mut device_ptr, size);
            if result != 0 {
                // Try to get available memory for error message
                let mut available = 0u64;
                let _ = hipMemGetInfo(&mut available, std::ptr::null_mut());
                return Err(HipError::MemoryAllocationFailed {
                    requested_size: size,
                    available_memory: available,
                });
            }

            Ok(HipMemoryHandle::new(device_ptr, size))
        }
    }

    /// Free device memory.
    pub fn free(&self, handle: HipMemoryHandle) -> HipResult<()> {
        unsafe {
            let result = hipFree(handle.as_ptr() as *mut std::ffi::c_void);
            if result != 0 {
                return Err(HipError::MemoryTransferFailed(format!(
                    "hipFree failed with code {}",
                    result
                )));
            }
        }
        Ok(())
    }

    /// Copy memory from host to device.
    pub fn memcpy_htod(&self, host: &[f64], device: &mut HipMemoryHandle) -> HipResult<()> {
        let size_bytes = host.len() * std::mem::size_of::<f64>();
        if size_bytes > device.size as usize {
            return Err(HipError::MemoryTransferFailed(format!(
                "Host buffer ({} bytes) exceeds device buffer ({} bytes)",
                size_bytes, device.size
            )));
        }

        unsafe {
            let result = hipMemcpy(
                device.as_mut_ptr(),
                host.as_ptr() as *const std::ffi::c_void,
                size_bytes,
                hipMemcpyKind::hipMemcpyHostToDevice as u32,
            );
            if result != 0 {
                return Err(HipError::MemoryTransferFailed(format!(
                    "hipMemcpy H2D failed with code {}",
                    result
                )));
            }
        }
        Ok(())
    }

    /// Copy memory from device to host.
    pub fn memcpy_dtoh(&self, device: &HipMemoryHandle, host: &mut [f64]) -> HipResult<()> {
        let size_bytes = host.len() * std::mem::size_of::<f64>();
        if size_bytes > device.size as usize {
            return Err(HipError::MemoryTransferFailed(format!(
                "Host buffer ({} bytes) exceeds device buffer ({} bytes)",
                size_bytes, device.size
            )));
        }

        unsafe {
            let result = hipMemcpy(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                device.as_ptr(),
                size_bytes,
                hipMemcpyKind::hipMemcpyDeviceToHost as u32,
            );
            if result != 0 {
                return Err(HipError::MemoryTransferFailed(format!(
                    "hipMemcpy D2H failed with code {}",
                    result
                )));
            }
        }
        Ok(())
    }

    /// Copy memory from device to host.
    pub fn memcpy_dtoh(&self, device: &HipMemoryHandle, host: &mut [f64]) -> HipResult<()> {
        let size_bytes = host.len() * std::mem::size_of::<f64>();
        if size_bytes > device.size as usize {
            return Err(HipError::MemoryTransferFailed(format!(
                "Host buffer ({} bytes) exceeds device buffer ({} bytes)",
                size_bytes, device.size
            )));
        }

        unsafe {
            let result = hipMemcpy(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                device.as_ptr(),
                size_bytes,
                hipMemcpyKind::hipMemcpyDeviceToHost as u32,
            );
            if result != 0 {
                return Err(HipError::MemoryTransferFailed(
                    format!("hipMemcpy D2H failed with code {}", result),
                ));
            }
        }
        Ok(())
    }
        }
        Ok(())
    }

    /// Synchronize device (wait for all operations to complete).
    pub fn synchronize(&self) -> HipResult<()> {
        unsafe {
            let result = hipDeviceSynchronize();
            if result != 0 {
                return Err(HipError::SynchronizationFailed(format!(
                    "hipDeviceSynchronize failed with code {}",
                    result
                )));
            }
        }
        Ok(())
    }

    /// Get available device memory.
    pub fn get_available_memory(&self) -> HipResult<u64> {
        unsafe {
            let mut available = 0u64;
            let result = hipMemGetInfo(&mut available, std::ptr::null_mut());
            if result != 0 {
                return Err(HipError::ApiError(format!(
                    "hipMemGetInfo failed with code {}",
                    result
                )));
            }
            Ok(available)
        }
    }
}

impl Drop for HipDevice {
    fn drop(&mut self) {
        let _ = unsafe { hipDeviceReset() };
    }
}

/// HIP kernel launcher for EMD operations.
pub struct HipKernelLauncher {
    device: HipDevice,
}

impl HipKernelLauncher {
    /// Create a new kernel launcher for device.
    pub fn new(device: HipDevice) -> Self {
        HipKernelLauncher { device }
    }

    /// Get reference to underlying device.
    pub fn device(&self) -> &HipDevice {
        &self.device
    }

    /// Launch generate_noise kernel.
    ///
    /// Generates Gaussian noise on GPU.
    pub fn launch_generate_noise(
        &self,
        seed: u64,
        scale: f64,
        output: &mut HipMemoryHandle,
        size: u32,
    ) -> HipResult<()> {
        let expected_bytes = size as u64 * std::mem::size_of::<f64>() as u64;
        if output.size < expected_bytes {
            return Err(HipError::InvalidKernelConfig(format!(
                "Output buffer too small: {} < {}",
                output.size, expected_bytes
            )));
        }

        unsafe {
            use crate::adapters::gpu::rocm_kernel_bindings::launch_generate_noise_hip;
            let result =
                launch_generate_noise_hip(seed, scale, output.as_mut_ptr() as *mut f64, size);
            if result != 0 {
                return Err(HipError::KernelLaunchFailed {
                    kernel_name: "generate_noise_hip".to_string(),
                    reason: format!("HIP error code {}", result),
                });
            }
        }

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;
        Ok(())
    }

    /// Launch add_signal kernel.
    ///
    /// Computes: output[i] = signal[i] + noise_scale * noise[i]
    pub fn launch_add_signal(
        &self,
        signal: &HipMemoryHandle,
        noise: &HipMemoryHandle,
        noise_scale: f64,
        output: &mut HipMemoryHandle,
        size: u32,
    ) -> HipResult<()> {
        let expected_bytes = size as u64 * std::mem::size_of::<f64>() as u64;

        if signal.size < expected_bytes {
            return Err(HipError::InvalidKernelConfig(format!(
                "Signal buffer too small: {} < {}",
                signal.size, expected_bytes
            )));
        }
        if noise.size < expected_bytes {
            return Err(HipError::InvalidKernelConfig(format!(
                "Noise buffer too small: {} < {}",
                noise.size, expected_bytes
            )));
        }
        if output.size < expected_bytes {
            return Err(HipError::InvalidKernelConfig(format!(
                "Output buffer too small: {} < {}",
                output.size, expected_bytes
            )));
        }

        unsafe {
            use crate::adapters::gpu::rocm_kernel_bindings::launch_add_signal_hip;
            let result = launch_add_signal_hip(
                signal.as_ptr() as *const f64,
                noise.as_ptr() as *const f64,
                noise_scale,
                output.as_mut_ptr() as *mut f64,
                size,
            );
            if result != 0 {
                return Err(HipError::KernelLaunchFailed {
                    kernel_name: "add_signal_hip".to_string(),
                    reason: format!("HIP error code {}", result),
                });
            }
        }

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;
        Ok(())
    }

    /// Launch find_extrema kernel.
    ///
    /// Finds local maxima and minima in signal.
    pub fn launch_find_extrema(
        &self,
        signal: &HipMemoryHandle,
        max_indices: &mut HipMemoryHandle,
        min_indices: &mut HipMemoryHandle,
        max_count: &mut HipMemoryHandle,
        min_count: &mut HipMemoryHandle,
        size: u32,
    ) -> HipResult<()> {
        let expected_signal_bytes = size as u64 * std::mem::size_of::<f64>() as u64;
        let expected_indices_bytes = size as u64 * std::mem::size_of::<u32>() as u64;
        let counter_bytes = std::mem::size_of::<u32>() as u64;

        if signal.size < expected_signal_bytes {
            return Err(HipError::InvalidKernelConfig(format!(
                "Signal buffer too small: {} < {}",
                signal.size, expected_signal_bytes
            )));
        }
        if max_indices.size < expected_indices_bytes {
            return Err(HipError::InvalidKernelConfig(format!(
                "Max indices buffer too small: {} < {}",
                max_indices.size, expected_indices_bytes
            )));
        }
        if min_indices.size < expected_indices_bytes {
            return Err(HipError::InvalidKernelConfig(format!(
                "Min indices buffer too small: {} < {}",
                min_indices.size, expected_indices_bytes
            )));
        }
        if max_count.size < counter_bytes {
            return Err(HipError::InvalidKernelConfig("Max count buffer too small".to_string()));
        }
        if min_count.size < counter_bytes {
            return Err(HipError::InvalidKernelConfig("Min count buffer too small".to_string()));
        }

        unsafe {
            use crate::adapters::gpu::rocm_kernel_bindings::launch_find_extrema_hip;
            let result = launch_find_extrema_hip(
                signal.as_ptr() as *const f64,
                max_indices.as_mut_ptr() as *mut u32,
                min_indices.as_mut_ptr() as *mut u32,
                max_count.as_mut_ptr() as *mut u32,
                min_count.as_mut_ptr() as *mut u32,
                size,
            );
            if result != 0 {
                return Err(HipError::KernelLaunchFailed {
                    kernel_name: "find_extrema_hip".to_string(),
                    reason: format!("HIP error code {}", result),
                });
            }
        }

        // Synchronize to ensure kernel completes
        self.device.synchronize()?;
        Ok(())
    }
}

/* HIP API declarations (C FFI bindings to HIP runtime) */

#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
struct hipDeviceProp_t {
    name: [i8; 256],
    uuid: [u8; 16],
    luid: [u8; 8],
    luidDeviceNodeMask: u32,
    totalGlobalMem: u64,
    sharedMemPerBlock: u64,
    regsPerBlock: i32,
    warpSize: i32,
    memPitch: u64,
    maxThreadsPerBlock: i32,
    maxThreadsDim: [i32; 3],
    maxGridSize: [i32; 3],
    clockRate: i32,
    totalConstMem: u64,
    major: i32,
    minor: i32,
    textureAlignment: u64,
    texturePitchAlignment: u64,
    deviceOverlap: i32,
    multiProcessorCount: i32,
    kernelExecTimeoutEnabled: i32,
    integrated: i32,
    canMapHostMemory: i32,
    computeMode: i32,
    maxTexture1D: i32,
    maxTexture1DMipmap: i32,
    maxTexture1DLinear: i32,
    maxTexture2D: [i32; 2],
    maxTexture2DMipmap: [i32; 2],
    maxTexture2DLinear: [i32; 3],
    maxTexture2DGather: [i32; 2],
    maxTexture3D: [i32; 3],
    maxTexture3DAlt: [i32; 3],
    maxTextureCubemap: i32,
    maxTextureCubemapLayered: [i32; 2],
    maxSurface1D: i32,
    maxSurface1DLayered: [i32; 2],
    maxSurface2D: [i32; 2],
    maxSurface2DLayered: [i32; 3],
    maxSurface3D: [i32; 3],
    maxSurfaceCubemap: i32,
    maxSurfaceCubemapLayered: [i32; 2],
    surfaceAlignment: u64,
    concurrentKernels: i32,
    ECCEnabled: i32,
    pciBusID: i32,
    pciDeviceID: i32,
    pciDomainID: i32,
    tccDriver: i32,
    asyncEngineCount: i32,
    unifiedAddressing: i32,
    memoryClockRate: i32,
    memoryBusWidth: i32,
    l2CacheSize: i32,
    persistingL2CacheMaxSize: i32,
    maxThreadsPerMultiProcessor: i32,
    streamPrioritiesSupported: i32,
    globalL1CacheSupported: i32,
    localL1CacheSupported: i32,
    sharedMemPerMultiprocessor: u64,
    regsPerMultiprocessor: i32,
    managedMemory: i32,
    isMultiGpuBoard: i32,
    multiGpuBoardGroupID: i32,
    hostNativeAtomic: i32,
    singleToDoublePrecisionPerfRatio: i32,
    pageableMemoryAccess: i32,
    pageableMemoryAccessUsesHostPageTables: i32,
    directManagedMemAccessFromHost: i32,
    concurrentManagedAccess: i32,
    computePreemptionSupported: i32,
    canUseHostPointerForRegisteredMem: i32,
    canUse64BitStreamMEM: i32,
    canUseStreamWaitValue64: i32,
    lMemBaseAddrs: u64,
    lMemSize: u64,
    maxSharedMemoryPerMultiProcessor: u64,
    isSharedMemPerMultiprocessorIncludingGlobals: i32,
    sharedMemPerBlockOptin: u64,
    pageableMemoryAccessUsesHostPageTables: i32,
    hostRegisterSupported: i32,
    sparseHipLaunch: i32,
    hostRegisterReadOnlySupported: i32,
    gpuDirectRDMASupported: i32,
    gpuDirectRDMAFlushWritesOptions: u32,
    gpuDirectRDMAWritesOrdering: i32,
    memoryAccessesIncludeNvDeferred: i32,
    streamWaitValue64Supported: i32,
    deferredManagedMemSupported: i32,
    ipcEventSupported: i32,
    clusterLaunch: i32,
    cooperativeMultiDeviceLaunch: i32,
    asicRevision: u32,
    managedMemorySupported: i32,
    hwAssistedCoherentMemory: i32,
    hwCoherentHostMemory: i32,
    cooperativeLaunchSupported: i32,
    maxBlockDimX: u32,
    maxBlockDimY: u32,
    maxBlockDimZ: u32,
    maxGridDimX: u32,
    maxGridDimY: u32,
    maxGridDimZ: u32,
    maxThreadsPerBlock: u32,
    waveSize: u32,
    workgroupMaxSize: u32,
    gridMaxSize: u32,
    imageSupport: u32,
    maxMemAllocSize: u64,
    clockInstructionRate: u32,
    arch: hipDeviceArch_t,
    hdpMemFlushCntl: *mut u32,
    hdpRegFlushCntl: *mut u32,
    cooperativeGroups: u32,
    cooperativeMultiDeviceGroups: u32,
    largeBarSupported: u32,
    isLargeBar: u32,
    supportCodeObject: u32,
    isPhysicalMultiGPU: u32,
    supportVirtualMemory: u32,
    gcnArchName: [u8; 256],
}

#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
struct hipDeviceArch_t {
    hasGlobalInt32Atomics: u32,
    hasGlobalFloatAtomicExch: u32,
    hasSharedInt32Atomics: u32,
    hasSharedFloatAtomicExch: u32,
    hasFloatAtomicAdd: u32,
    hasGlobalInt64Atomics: u32,
    hasSharedInt64Atomics: u32,
    hasDoubles: u32,
    hasWarpVote: u32,
    hasWarpBallot: u32,
    hasWarpShuffle: u32,
    hasFunnelShift: u32,
    hasThreadFenceSystem: u32,
    hasSyncThreadsExt: u32,
    hasSurfaceFuncs: u32,
    has3dGrid: u32,
    hasDynamicParallelism: u32,
    hasDeprecatedTextureFuncs: u32,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[allow(dead_code)]
enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4,
}

#[link(name = "amdhip64")]
extern "C" {
    // Device management
    fn hipGetDeviceCount(count: *mut i32) -> u32;
    fn hipSetDevice(device: i32) -> u32;
    fn hipGetDeviceProperties(prop: *mut hipDeviceProp_t, device: i32) -> u32;
    fn hipDeviceReset() -> u32;
    fn hipDeviceSynchronize() -> u32;

    // Memory management
    fn hipMalloc(ptr: *mut *mut std::ffi::c_void, size: u64) -> u32;
    fn hipFree(ptr: *mut std::ffi::c_void) -> u32;
    fn hipMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        size: u64,
        kind: u32,
    ) -> u32;
    fn hipMemGetInfo(free: *mut u64, total: *mut u64) -> u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hip_device_handle_creation() {
        let handle = HipDeviceHandle { device_id: 0 };
        assert_eq!(handle.device_id, 0);
    }

    #[test]
    fn test_hip_memory_handle_creation() {
        let ptr = 0x1000usize as *mut std::ffi::c_void;
        unsafe {
            let handle = HipMemoryHandle::new(ptr, 1024);
            assert_eq!(handle.size, 1024);
        }
    }

    #[test]
    fn test_hip_error_display() {
        let err = HipError::DeviceNotFound(42);
        let msg = format!("{}", err);
        assert!(msg.contains("42"));
    }
}
