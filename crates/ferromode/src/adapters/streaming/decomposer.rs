//! Streaming decomposition orchestrator.
//!
//! This module provides the `StreamingDecomposer` struct that manages
//! state-preserving, chunk-based EMD decomposition for signals that cannot
//! fit in memory or arrive continuously.

use crate::adapters::streaming::predictor::BoundaryPrediction;
use crate::adapters::streaming::state::{IntermittencyMetrics, PredictorState, StreamingState};
use crate::algorithms::emd;
use crate::error::EmdError;
use crate::types::{DecompositionResult, EmdConfig, Signal};

/// Result of decomposing a single chunk.
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// Intrinsic Mode Functions extracted from this chunk
    pub imfs: Vec<Vec<f64>>,
    /// Remainder signal
    pub remainder: Vec<f64>,
    /// Metrics computed for this chunk (for algorithm adaptation)
    pub metrics: IntermittencyMetrics,
}

/// Configuration for streaming decomposition.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Base EMD configuration
    pub base_config: EmdConfig,
    /// Size of signal chunks to process
    pub chunk_size: usize,
    /// Capacity of ring buffer for signal history
    pub buffer_size: usize,
    /// Capacity of sifting history
    pub sifting_history_size: usize,
}

impl StreamingConfig {
    /// Create a new streaming configuration.
    pub fn new(
        base_config: EmdConfig,
        chunk_size: usize,
        buffer_size: usize,
        sifting_history_size: usize,
    ) -> Result<Self, EmdError> {
        if chunk_size == 0 {
            return Err(EmdError::InvalidConfig("chunk_size must be > 0".to_string()));
        }
        if buffer_size == 0 {
            return Err(EmdError::InvalidConfig("buffer_size must be > 0".to_string()));
        }
        Ok(Self { base_config, chunk_size, buffer_size, sifting_history_size })
    }
}

/// Streaming EMD decomposer with state management.
///
/// Processes signals in fixed-size chunks while maintaining state across
/// boundaries to reduce end effects and ensure envelope continuity.
///
/// # Example
///
/// ```ignore
/// use ferromode::adapters::streaming::{StreamingDecomposer, ArModel};
/// use ferromode::types::{Signal, EmdConfig, BoundaryConditionType};
///
/// let config = EmdConfig::default();
/// let predictor = Box::new(ArModel::new(3)?);
/// let mut decomposer = StreamingDecomposer::new(config, predictor, 1024)?;
///
/// for chunk_data in data_stream {
///     let signal = Signal::from_slice(&chunk_data)?;
///     let result = decomposer.decompose_chunk(&signal)?;
///     println!("Extracted {} IMFs", result.imfs.len());
/// }
/// ```
pub struct StreamingDecomposer {
    config: StreamingConfig,
    state: StreamingState,
}

impl StreamingDecomposer {
    /// Create a new streaming decomposer.
    ///
    /// # Arguments
    /// * `base_config` — EMD configuration for decomposition
    /// * `predictor` — Boundary prediction model (Box<dyn BoundaryPrediction>)
    /// * `buffer_size` — Ring buffer capacity for history (bytes)
    ///
    /// # Returns
    /// A new decomposer ready for chunk processing
    ///
    /// # Errors
    /// Returns `InvalidConfig` if configuration is invalid
    pub fn new(
        base_config: EmdConfig,
        predictor: Box<dyn PredictorState>,
        buffer_size: usize,
    ) -> Result<Self, EmdError> {
        let chunk_size = 1024;
        let sifting_history_size = 100;

        let config =
            StreamingConfig::new(base_config, chunk_size, buffer_size, sifting_history_size)?;

        let state = StreamingState::new(predictor, buffer_size, sifting_history_size);

        Ok(Self { config, state })
    }

    /// Decompose a single chunk with state management.
    ///
    /// This method:
    /// 1. Predicts boundary extensions to reduce end effects
    /// 2. Decomposes the extended signal using domain algorithms
    /// 3. Trims results back to original chunk size
    /// 4. Updates state for continuity in next chunk
    ///
    /// # Arguments
    /// * `chunk` — Signal chunk to decompose
    ///
    /// # Returns
    /// ChunkResult with IMFs, remainder, and metrics
    ///
    /// # Errors
    /// Returns `EmdError` if decomposition fails
    pub fn decompose_chunk(&mut self, chunk: &Signal) -> Result<ChunkResult, EmdError> {
        // Validate input
        if chunk.len() < 10 {
            return Err(EmdError::InsufficientData);
        }

        let chunk_data = chunk.values();

        // Compute metrics for algorithm adaptation
        let metrics = self.compute_intermittency_metrics(chunk_data);
        self.state.update_metrics(metrics);

        // Predict boundaries for extension
        let n_extend = (chunk_data.len() / 4).min(64).max(10);
        let pred_start = self.state.predictor_ref().predict_next(chunk_data, n_extend);
        let pred_end = self.state.predictor_ref().predict_next(chunk_data, n_extend);

        // Build extended signal with predictions
        let mut extended = Vec::with_capacity(chunk_data.len() + 2 * n_extend);
        extended.extend(pred_start.iter().rev()); // Reverse for causal ordering
        extended.extend_from_slice(chunk_data);
        extended.extend(&pred_end);

        let extended_signal = Signal::from_slice(&extended)?;

        // Decompose using domain algorithm
        let result = emd::decompose(&extended_signal, &self.config.base_config)?;

        // Trim IMFs back to original chunk length
        let imfs = result
            .imfs
            .iter()
            .map(|imf| {
                let start = n_extend;
                let end = (start + chunk_data.len()).min(imf.len());
                imf[start..end].to_vec()
            })
            .collect();

        let remainder = {
            let start = n_extend;
            let end = (start + chunk_data.len()).min(result.remainder.len());
            result.remainder[start..end].to_vec()
        };

        // Update state
        self.state.push_samples(chunk_data);
        self.state.next_chunk();

        // TODO: Update envelope tracking for continuity (Phase 4)

        Ok(ChunkResult { imfs, remainder, metrics })
    }

    /// Compute intermittency metrics for the chunk.
    ///
    /// Metrics guide adaptive algorithm selection (EMD vs EEMD vs CEEMDAN).
    fn compute_intermittency_metrics(&self, signal: &[f64]) -> IntermittencyMetrics {
        // Placeholder implementation for v2.0
        // Full implementation in Phase 4 (T-266)

        let spectral_entropy = 0.3; // TODO: Compute from FFT
        let extrema_spacing_cv = 0.2; // TODO: From extrema positions
        let stationarity_score = 0.8; // TODO: Composite score

        IntermittencyMetrics::new(spectral_entropy, extrema_spacing_cv, stationarity_score)
    }

    /// Get current chunk ID.
    pub fn chunk_id(&self) -> u64 {
        self.state.chunk_id
    }

    /// Get reference to internal state.
    pub fn state(&self) -> &StreamingState {
        &self.state
    }

    /// Get mutable reference to internal state.
    pub fn state_mut(&mut self) -> &mut StreamingState {
        &mut self.state
    }

    /// Reset state for a new stream (cleanup).
    pub fn reset(&mut self) {
        self.state.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::streaming::predictor::ArModel;

    struct MockPredictor;

    impl crate::adapters::streaming::state::PredictorState for MockPredictor {
        fn predict_next(&self, signal: &[f64], n_ahead: usize) -> Vec<f64> {
            if signal.is_empty() {
                vec![0.0; n_ahead]
            } else {
                vec![signal[signal.len() - 1]; n_ahead]
            }
        }

        fn update(&mut self, _signal: &[f64]) -> Result<(), EmdError> {
            Ok(())
        }

        fn clone_box(&self) -> Box<dyn crate::adapters::streaming::state::PredictorState> {
            Box::new(MockPredictor)
        }
    }

    #[test]
    fn test_streaming_config_new() {
        let config = EmdConfig::default();
        let sc = StreamingConfig::new(config, 512, 1024, 100).unwrap();
        assert_eq!(sc.chunk_size, 512);
        assert_eq!(sc.buffer_size, 1024);
    }

    #[test]
    fn test_streaming_config_zero_chunk_size() {
        let config = EmdConfig::default();
        let result = StreamingConfig::new(config, 0, 1024, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_decomposer_new() {
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();
        assert_eq!(decomposer.chunk_id(), 0);
    }

    #[test]
    fn test_streaming_decomposer_chunk_id_increments() {
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();

        // Create a simple sinusoidal signal
        let signal_data: Vec<f64> =
            (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
        let signal = Signal::from_slice(&signal_data).unwrap();

        let result = decomposer.decompose_chunk(&signal);
        assert!(result.is_ok());
        assert_eq!(decomposer.chunk_id(), 1);
    }

    #[test]
    fn test_streaming_decomposer_insufficient_data() {
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();

        let signal = Signal::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let result = decomposer.decompose_chunk(&signal);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_decomposer_reset() {
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();

        let signal_data: Vec<f64> =
            (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 50.0).sin()).collect();
        let signal = Signal::from_slice(&signal_data).unwrap();

        let _ = decomposer.decompose_chunk(&signal);
        assert_eq!(decomposer.chunk_id(), 1);

        decomposer.reset();
        assert_eq!(decomposer.chunk_id(), 0);
    }

    #[test]
    fn test_chunk_result_creation() {
        let imfs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let remainder = vec![5.0, 6.0];
        let metrics = IntermittencyMetrics::new(0.3, 0.2, 0.8);

        let result = ChunkResult { imfs: imfs.clone(), remainder: remainder.clone(), metrics };
        assert_eq!(result.imfs.len(), 2);
        assert_eq!(result.remainder.len(), 2);
    }
}
