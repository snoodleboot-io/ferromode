#![warn(missing_docs)]

//! Adapter layer for hardware acceleration and optimization.
//!
//! Provides specialized implementations for various hardware backends
//! and optimization techniques, while maintaining API compatibility with
//! core algorithms.

pub mod gpu;

pub use gpu::{GpuAdapter, GpuConfig};
