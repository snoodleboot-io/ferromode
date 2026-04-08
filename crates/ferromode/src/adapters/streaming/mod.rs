//! Streaming decomposition adapter for real-time EMD processing.
//!
//! This module provides chunk-based decomposition functionality that maintains
//! state across multiple signal chunks, enabling bounded-memory, low-latency
//! processing of continuous or large signals.
//!
//! # Architecture
//!
//! The streaming adapter implements the Adapter pattern from the v2.0 architecture.
//! It sits between the Python bindings and the domain layer (algorithms, boundary, sifting),
//! managing state and orchestrating boundary prediction while delegating core
//! decomposition to existing v1.x modules.
//!
//! ## Core Components
//!
//! - **`StreamingState`**: Maintains history, envelopes, and predictor parameters
//! - **`StreamingDecomposer`**: Processes chunks with state management
//! - **`BoundaryPrediction`**: Trait for predicting signal extensions
//! - **Metrics**: Intermittency detection for adaptive algorithm selection
//!
//! ## Example
//!
//! ```ignore
//! use ferromode::adapters::streaming::{StreamingDecomposer, ArModel};
//! use ferromode::types::Signal;
//!
//! let predictor = Box::new(ArModel::new(3)?);
//! let mut decomposer = StreamingDecomposer::new(
//!     config,
//!     predictor,
//!     1024,  // buffer_size
//! )?;
//!
//! // Process chunks as they arrive
//! for chunk in signal_stream {
//!     let result = decomposer.decompose_chunk(&chunk)?;
//!     println!("Chunk {}: {} IMFs", decomposer.chunk_id(), result.imfs.len());
//! }
//! ```

pub mod decomposer;
pub mod predictor;
pub mod state;

pub use decomposer::{StreamingConfig, StreamingDecomposer};
pub use predictor::{ArModel, BoundaryPrediction};
pub use state::{
    AdaptiveAlgorithm, IntermittencyMetrics, PredictorState, RingBuffer, SiftingIteration,
    StreamingState,
};
