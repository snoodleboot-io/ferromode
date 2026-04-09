#![warn(missing_docs)]

//! Adapter layer for hardware acceleration and optimization.
//!
//! Provides specialized implementations for various hardware backends
//! and optimization techniques while maintaining API compatibility with
//! core algorithms.

pub mod boundary_prediction;
pub mod gpu;
pub mod multidim;
pub mod streaming;

pub use boundary_prediction::{BoundaryPredictionConfig, BoundarySelector, LstmModel};
pub use gpu::{GpuAdapter, GpuConfig};
pub use multidim::{
    decompose_image_2d_separable, find_local_extrema_2d, pad_symmetric_1d, unpad_1d,
    DecompositionMetadata, Extrema2D, Image2D, Image2DDecomposition, Volume3D,
    Volume3DDecomposition,
};
pub use streaming::{
    AdaptiveAlgorithm, ArModel, BoundaryPrediction, IntermittencyMetrics, PredictorState,
    RingBuffer, StreamingDecomposer, StreamingState,
};
