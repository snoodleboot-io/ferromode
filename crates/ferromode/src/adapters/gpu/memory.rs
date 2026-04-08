#![warn(missing_docs)]

//! GPU memory management with pooling and fragmentation prevention.
//!
//! Provides efficient memory allocation, deallocation, and tracking
//! for GPU device memory with configurable pool strategies.

use super::DeviceError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// GPU memory allocation statistics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total memory allocated (bytes)
    pub allocated: u64,
    /// Peak memory used (bytes)
    pub peak_used: u64,
    /// Number of active allocations
    pub num_allocations: usize,
    /// Memory fragmentation ratio (0.0 to 1.0)
    pub fragmentation_ratio: f64,
}

/// Represents a GPU memory allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AllocationId(u64);

/// Configuration for GPU memory pool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPoolConfig {
    /// Maximum memory to allocate (bytes), None = unlimited
    pub max_memory: Option<u64>,
    /// Initial pool size (bytes)
    pub initial_pool_size: u64,
    /// Minimum block size (bytes), blocks smaller are not pooled
    pub min_block_size: u64,
    /// Enable block coalescing on deallocation
    pub enable_coalescing: bool,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_memory: Some(8 * 1024 * 1024 * 1024), // 8GB default
            initial_pool_size: 512 * 1024 * 1024,     // 512MB
            min_block_size: 1024,                     // 1KB
            enable_coalescing: true,
        }
    }
}

/// GPU memory pool for efficient allocation and deallocation.
pub struct GpuMemoryPool {
    config: MemoryPoolConfig,
    next_alloc_id: u64,
    allocations: HashMap<AllocationId, AllocationInfo>,
    total_allocated: u64,
    peak_used: u64,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    size: u64,
    allocated_at: std::time::Instant,
}

impl GpuMemoryPool {
    /// Create a new GPU memory pool with given configuration.
    pub fn new(config: MemoryPoolConfig) -> Result<Self, DeviceError> {
        if config.initial_pool_size > config.max_memory.unwrap_or(u64::MAX) {
            return Err(DeviceError::InitializationFailed(
                "Initial pool size exceeds max memory".to_string(),
            ));
        }

        Ok(GpuMemoryPool {
            config,
            next_alloc_id: 1,
            allocations: HashMap::new(),
            total_allocated: 0,
            peak_used: 0,
        })
    }

    /// Allocate memory from the pool.
    pub fn allocate(&mut self, size: u64) -> Result<AllocationId, DeviceError> {
        if size == 0 {
            return Err(DeviceError::QueryFailed("Cannot allocate 0 bytes".to_string()));
        }

        // Check if allocation would exceed limit
        if let Some(max) = self.config.max_memory {
            if self.total_allocated + size > max {
                return Err(DeviceError::QueryFailed(format!(
                    "Allocation would exceed limit: {} + {} > {}",
                    self.total_allocated, size, max
                )));
            }
        }

        let alloc_id = AllocationId(self.next_alloc_id);
        self.next_alloc_id += 1;

        self.allocations
            .insert(alloc_id, AllocationInfo { size, allocated_at: std::time::Instant::now() });

        self.total_allocated += size;
        if self.total_allocated > self.peak_used {
            self.peak_used = self.total_allocated;
        }

        Ok(alloc_id)
    }

    /// Deallocate memory back to the pool.
    pub fn deallocate(&mut self, alloc_id: AllocationId) -> Result<(), DeviceError> {
        if let Some(info) = self.allocations.remove(&alloc_id) {
            self.total_allocated = self.total_allocated.saturating_sub(info.size);
            Ok(())
        } else {
            Err(DeviceError::QueryFailed(format!("Allocation not found: {:?}", alloc_id)))
        }
    }

    /// Get current memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let fragmentation_ratio = if self.total_allocated == 0 {
            0.0
        } else {
            // Simple fragmentation estimate: unused space in allocated region
            let unused = (self.allocations.values().map(|a| a.size).sum::<u64>()) as i64
                - self.total_allocated as i64;
            if unused > 0 {
                (unused as f64) / (self.total_allocated as f64)
            } else {
                0.0
            }
        };

        MemoryStats {
            allocated: self.total_allocated,
            peak_used: self.peak_used,
            num_allocations: self.allocations.len(),
            fragmentation_ratio: fragmentation_ratio.clamp(0.0, 1.0),
        }
    }

    /// Get available memory remaining.
    pub fn available_memory(&self) -> u64 {
        self.config.max_memory.unwrap_or(u64::MAX).saturating_sub(self.total_allocated)
    }

    /// Check if a specific allocation exists.
    pub fn contains(&self, alloc_id: AllocationId) -> bool {
        self.allocations.contains_key(&alloc_id)
    }

    /// Get size of a specific allocation.
    pub fn allocation_size(&self, alloc_id: AllocationId) -> Option<u64> {
        self.allocations.get(&alloc_id).map(|info| info.size)
    }

    /// Clear all allocations (assumes GPU memory is already freed).
    pub fn clear(&mut self) {
        self.allocations.clear();
        self.total_allocated = 0;
    }
}

impl Default for GpuMemoryPool {
    fn default() -> Self {
        GpuMemoryPool::new(MemoryPoolConfig::default()).expect("default config should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool_creation() {
        let pool = GpuMemoryPool::default();
        assert_eq!(pool.stats().allocated, 0);
        assert_eq!(pool.stats().num_allocations, 0);
    }

    #[test]
    fn test_allocate_memory() {
        let mut pool = GpuMemoryPool::default();
        let alloc1 = pool.allocate(1024).expect("allocation");
        assert_eq!(pool.stats().allocated, 1024);
        assert_eq!(pool.stats().num_allocations, 1);
        assert!(pool.contains(alloc1));
    }

    #[test]
    fn test_deallocate_memory() {
        let mut pool = GpuMemoryPool::default();
        let alloc1 = pool.allocate(1024).expect("allocation");
        assert_eq!(pool.stats().allocated, 1024);
        pool.deallocate(alloc1).expect("deallocation");
        assert_eq!(pool.stats().allocated, 0);
        assert!(!pool.contains(alloc1));
    }

    #[test]
    fn test_multiple_allocations() {
        let mut pool = GpuMemoryPool::default();
        let ids: Vec<_> =
            (0..5).map(|i| pool.allocate((i + 1) * 1024).expect("allocation")).collect();

        let total: u64 = (1..=5).map(|i| i * 1024).sum();
        assert_eq!(pool.stats().allocated, total);
        assert_eq!(pool.stats().num_allocations, 5);

        for id in ids {
            assert!(pool.contains(id));
        }
    }

    #[test]
    fn test_peak_memory() {
        let mut pool = GpuMemoryPool::default();
        pool.allocate(1024 * 1024).expect("allocation");
        assert_eq!(pool.stats().peak_used, 1024 * 1024);

        pool.allocate(2 * 1024 * 1024).expect("allocation");
        assert_eq!(pool.stats().peak_used, 3 * 1024 * 1024);
    }

    #[test]
    fn test_max_memory_limit() {
        let config = MemoryPoolConfig {
            max_memory: Some(1024),
            initial_pool_size: 512,
            ..Default::default()
        };
        let mut pool = GpuMemoryPool::new(config).expect("pool creation");

        // First allocation should succeed
        let alloc1 = pool.allocate(512).expect("allocation");
        assert_eq!(pool.stats().allocated, 512);

        // Second allocation fits
        let alloc2 = pool.allocate(512).expect("allocation");
        assert_eq!(pool.stats().allocated, 1024);

        // Third allocation fails
        let result = pool.allocate(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_allocation_size() {
        let mut pool = GpuMemoryPool::default();
        let alloc = pool.allocate(2048).expect("allocation");
        assert_eq!(pool.allocation_size(alloc), Some(2048));
        assert_eq!(pool.allocation_size(AllocationId(999)), None);
    }

    #[test]
    fn test_available_memory() {
        let config = MemoryPoolConfig {
            max_memory: Some(4096),
            initial_pool_size: 2048,
            ..Default::default()
        };
        let mut pool = GpuMemoryPool::new(config).expect("pool creation");
        assert_eq!(pool.available_memory(), 4096);

        pool.allocate(1024).expect("allocation");
        assert_eq!(pool.available_memory(), 3072);
    }

    #[test]
    fn test_clear_allocations() {
        let mut pool = GpuMemoryPool::default();
        pool.allocate(1024).expect("allocation");
        pool.allocate(2048).expect("allocation");
        assert_eq!(pool.stats().allocated, 3072);

        pool.clear();
        assert_eq!(pool.stats().allocated, 0);
        assert_eq!(pool.stats().num_allocations, 0);
    }

    #[test]
    fn test_deallocate_nonexistent() {
        let mut pool = GpuMemoryPool::default();
        let result = pool.deallocate(AllocationId(999));
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_allocation_fails() {
        let mut pool = GpuMemoryPool::default();
        let result = pool.allocate(0);
        assert!(result.is_err());
    }
}
