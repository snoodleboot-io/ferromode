#![warn(missing_docs)]

//! Adapter layer for hardware acceleration and optimization.
//!
//! Provides specialized implementations for various hardware backends
//! and optimization techniques while maintaining API compatibility with
//! core algorithms.

pub mod boundary_prediction;
pub mod gpu;
pub mod streaming;

pub use boundary_prediction::{BoundaryPredictionConfig, BoundarySelector, LstmModel};
pub use gpu::{GpuAdapter, GpuConfig};
pub use streaming::{
    AdaptiveAlgorithm, ArModel, BoundaryPrediction, IntermittencyMetrics, PredictorState,
    RingBuffer, StreamingDecomposer, StreamingState,
};
