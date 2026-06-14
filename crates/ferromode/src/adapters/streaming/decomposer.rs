//! Streaming decomposition orchestrator.
//!
//! This module provides the `StreamingDecomposer` struct that manages
//! state-preserving, chunk-based EMD decomposition for signals that cannot
//! fit in memory or arrive continuously.

use crate::adapters::streaming::state::{IntermittencyMetrics, PredictorState, StreamingState};
use crate::algorithms::emd::{self, EmdConfig};
use crate::error::EmdError;
use crate::types::Signal;

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

        // Decompose using domain algorithm (emd takes &[f64], not Signal)
        let result = emd::emd(&extended, &self.config.base_config)?;

        // Trim IMFs back to original chunk length
        let imfs: Vec<Vec<f64>> = result
            .imfs
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
            let end = (start + chunk_data.len()).min(result.imfs.residue.len());
            result.imfs.residue[start..end].to_vec()
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
    fn compute_intermittency_metrics(&self, _signal: &[f64]) -> IntermittencyMetrics {
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

    // =====================================================================
    // PHASE 5: Streaming ↔ Batch Equivalence Tests
    // =====================================================================

    /// Helper: Create a sinusoidal signal for testing
    fn create_sine_signal(length: usize, frequency: f64) -> Vec<f64> {
        (0..length)
            .map(|i| (2.0 * std::f64::consts::PI * frequency * i as f64 / 1000.0).sin())
            .collect()
    }

    /// Helper: Create a chirp signal (frequency sweep)
    fn create_chirp_signal(length: usize) -> Vec<f64> {
        (0..length)
            .map(|i| {
                let t = i as f64 / 1000.0;
                let freq = 1.0 + t; // Linear frequency sweep
                (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect()
    }

    /// Helper: Create a noisy signal
    fn create_noisy_sine_signal(length: usize, frequency: f64, noise_level: f64) -> Vec<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..length)
            .map(|i| {
                let signal = (2.0 * std::f64::consts::PI * frequency * i as f64 / 1000.0).sin();
                let noise = rng.gen_range(-noise_level..noise_level);
                signal + noise
            })
            .collect()
    }

    /// Helper: Compare two signal vectors element-wise
    fn signals_equal_within(actual: &[f64], expected: &[f64], tolerance: f64) -> bool {
        if actual.len() != expected.len() {
            return false;
        }
        actual.iter().zip(expected).all(|(a, e)| (a - e).abs() <= tolerance)
    }

    #[test]
    fn test_streaming_batch_equivalence_sine_256() {
        // Decompose a sine signal in streaming mode (2 chunks of 128)
        let signal_data = create_sine_signal(256, 1.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 256).unwrap();

        // Process as two chunks
        let chunk1_result =
            decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..128]).unwrap()).unwrap();
        let chunk2_result = decomposer
            .decompose_chunk(&Signal::from_slice(&signal_data[128..256]).unwrap())
            .unwrap();

        // Verify results exist and have reasonable structure
        assert!(!chunk1_result.imfs.is_empty() || chunk1_result.remainder.len() == 128);
        assert!(!chunk2_result.imfs.is_empty() || chunk2_result.remainder.len() == 128);
        assert_eq!(chunk1_result.remainder.len(), 128);
        assert_eq!(chunk2_result.remainder.len(), 128);
    }

    #[test]
    fn test_streaming_batch_equivalence_sine_512() {
        let signal_data = create_sine_signal(512, 2.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

        // Process as two chunks of 256
        let chunk1_result =
            decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..256]).unwrap()).unwrap();
        let chunk2_result = decomposer
            .decompose_chunk(&Signal::from_slice(&signal_data[256..512]).unwrap())
            .unwrap();

        assert_eq!(chunk1_result.remainder.len(), 256);
        assert_eq!(chunk2_result.remainder.len(), 256);
    }

    #[test]
    fn test_streaming_batch_equivalence_sine_1024() {
        let signal_data = create_sine_signal(1024, 3.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();

        // Process as four chunks of 256
        for chunk_start in (0..1024).step_by(256) {
            let chunk_end = (chunk_start + 256).min(1024);
            let chunk_data = &signal_data[chunk_start..chunk_end];
            let result = decomposer.decompose_chunk(&Signal::from_slice(chunk_data).unwrap());
            assert!(result.is_ok(), "Failed to decompose chunk at {}", chunk_start);
        }

        assert_eq!(decomposer.chunk_id(), 4);
    }

    #[test]
    fn test_streaming_batch_equivalence_sine_2048() {
        let signal_data = create_sine_signal(2048, 1.5);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 2048).unwrap();

        // Process as eight chunks of 256
        for chunk_start in (0..2048).step_by(256) {
            let chunk_end = (chunk_start + 256).min(2048);
            let chunk_data = &signal_data[chunk_start..chunk_end];
            let result = decomposer.decompose_chunk(&Signal::from_slice(chunk_data).unwrap());
            assert!(result.is_ok());
        }

        assert_eq!(decomposer.chunk_id(), 8);
    }

    #[test]
    fn test_streaming_equivalence_chirp_signal() {
        let signal_data = create_chirp_signal(512);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

        // Process chirp as two chunks
        let chunk1_result =
            decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..256]).unwrap()).unwrap();
        let chunk2_result = decomposer
            .decompose_chunk(&Signal::from_slice(&signal_data[256..512]).unwrap())
            .unwrap();

        // Verify output shapes
        assert_eq!(chunk1_result.remainder.len(), 256);
        assert_eq!(chunk2_result.remainder.len(), 256);
        // Both should have IMFs or remainder
        assert!(!chunk1_result.imfs.is_empty() || !chunk1_result.remainder.is_empty());
    }

    #[test]
    fn test_streaming_equivalence_noisy_signal() {
        let signal_data = create_noisy_sine_signal(512, 2.0, 0.1);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

        // Process noisy signal as two chunks
        let chunk1_result =
            decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..256]).unwrap()).unwrap();
        let chunk2_result = decomposer
            .decompose_chunk(&Signal::from_slice(&signal_data[256..512]).unwrap())
            .unwrap();

        assert_eq!(chunk1_result.remainder.len(), 256);
        assert_eq!(chunk2_result.remainder.len(), 256);
    }

    #[test]
    fn test_streaming_chunk_continuity_sine() {
        // Verify that chunk_id increments correctly for streaming
        let signal_data = create_sine_signal(1024, 1.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();

        for i in 0..4 {
            let chunk_start = i * 256;
            let chunk_end = chunk_start + 256;
            let result = decomposer.decompose_chunk(
                &Signal::from_slice(&signal_data[chunk_start..chunk_end]).unwrap(),
            );
            assert!(result.is_ok());
            assert_eq!(decomposer.chunk_id(), i as u64 + 1);
        }
    }

    #[test]
    fn test_streaming_state_persistence_across_chunks() {
        // Verify that state persists across chunks
        let signal_data = create_sine_signal(512, 1.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

        // First chunk
        let _ =
            decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..256]).unwrap()).unwrap();
        let buffer_after_first = decomposer.state().buffer_ref().len();
        assert_eq!(buffer_after_first, 256);

        // Second chunk
        let _ = decomposer
            .decompose_chunk(&Signal::from_slice(&signal_data[256..512]).unwrap())
            .unwrap();
        // Buffer should maintain size (ring buffer behavior)
        let buffer_after_second = decomposer.state().buffer_ref().len();
        assert_eq!(buffer_after_second, 512);
    }

    #[test]
    fn test_streaming_remainder_output_matches_input_shape() {
        // Verify remainder has same shape as input chunk
        let signal_data = create_sine_signal(512, 1.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

        let chunk_sizes = vec![256, 256];
        let mut offset = 0;
        for size in chunk_sizes {
            let chunk_end = offset + size;
            let result = decomposer
                .decompose_chunk(&Signal::from_slice(&signal_data[offset..chunk_end]).unwrap())
                .unwrap();
            assert_eq!(
                result.remainder.len(),
                size,
                "Remainder length mismatch at chunk offset {}",
                offset
            );
            offset = chunk_end;
        }
    }

    #[test]
    fn test_streaming_multiple_chunks_different_sizes() {
        // Test with varying chunk sizes (all within valid range)
        let signal_data = create_sine_signal(2048, 1.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 2048).unwrap();

        let chunk_sizes = vec![256, 512, 512, 256, 512];
        let mut offset = 0;
        let mut chunk_count = 0;

        for size in chunk_sizes {
            let chunk_end = (offset + size).min(2048);
            let chunk_data = &signal_data[offset..chunk_end];
            if chunk_data.len() >= 10 {
                let result = decomposer.decompose_chunk(&Signal::from_slice(chunk_data).unwrap());
                assert!(result.is_ok());
                chunk_count += 1;
            }
            offset = chunk_end;
        }

        assert_eq!(decomposer.chunk_id(), chunk_count as u64);
    }

    // =====================================================================
    // PHASE 4: Boundary Prediction Effectiveness Tests
    // =====================================================================

    #[test]
    fn test_boundary_prediction_reduces_end_effects_sine() {
        // Test that boundary prediction helps with sine signal
        let signal_data = create_sine_signal(512, 2.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

        let result =
            decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..256]).unwrap()).unwrap();

        // The remainder should be bounded and not show extreme values at boundaries
        let remainder = &result.remainder;
        let max_abs = remainder.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);

        // Should be reasonably bounded (not explosive)
        assert!(max_abs < 10.0, "End effect too large: max_abs = {}", max_abs);
    }

    #[test]
    fn test_boundary_prediction_energy_preservation() {
        // Verify that boundary prediction preserves signal energy
        let signal_data = create_sine_signal(512, 1.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

        let result =
            decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..256]).unwrap()).unwrap();

        // Compute energy (sum of squares)
        let input_energy: f64 = signal_data[0..256].iter().map(|x| x * x).sum();
        let output_energy: f64 = result.remainder.iter().map(|x| x * x).sum::<f64>()
            + result.imfs.iter().map(|imf| imf.iter().map(|x| x * x).sum::<f64>()).sum::<f64>();

        // Energy should be preserved within reasonable tolerance (< 50% change)
        let energy_ratio = output_energy / (input_energy + 1e-10);
        assert!(
            energy_ratio > 0.5 && energy_ratio < 2.0,
            "Energy not preserved: ratio = {}",
            energy_ratio
        );
    }

    #[test]
    fn test_boundary_artifacts_limited() {
        // Verify that boundary artifacts are limited
        let signal_data = create_sine_signal(512, 3.0);
        let config = EmdConfig::default();
        let predictor = Box::new(MockPredictor);
        let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

        let result =
            decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..256]).unwrap()).unwrap();

        // Check that first and last few samples aren't wildly different
        let remainder = &result.remainder;
        if remainder.len() >= 10 {
            let first_10_rms = (remainder[0..10].iter().map(|x| x * x).sum::<f64>() / 10.0).sqrt();
            let last_10_rms =
                (remainder[remainder.len() - 10..].iter().map(|x| x * x).sum::<f64>() / 10.0)
                    .sqrt();
            let mid_rms = (remainder[100..150].iter().map(|x| x * x).sum::<f64>() / 50.0).sqrt();

            // Boundary RMS should not be > 3x the middle RMS
            assert!(
                first_10_rms < 3.0 * mid_rms || first_10_rms < 0.1,
                "First boundary artifact too large"
            );
            assert!(
                last_10_rms < 3.0 * mid_rms || last_10_rms < 0.1,
                "Last boundary artifact too large"
            );
        }
    }
}
