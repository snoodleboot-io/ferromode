//! Streaming decomposition adapter for real-time EMD processing.

pub mod state;
pub mod predictor;
pub mod decomposer;

pub use decomposer::StreamingDecomposer;
pub use predictor::{BoundaryPrediction, ArModel};
pub use state::{
    StreamingState, RingBuffer, SiftingIteration, IntermittencyMetrics, AdaptiveAlgorithm,
    PredictorState,
};
