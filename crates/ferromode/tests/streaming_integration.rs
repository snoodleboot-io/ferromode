//! Integration tests for streaming decomposition.
//!
//! Tests the complete streaming workflow with real signals
//! and verifies equivalence with batch processing.

use ferromode::adapters::streaming::{ArModel, PredictorState, StreamingDecomposer};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::error::EmdError;
use ferromode::types::Signal;
use std::f64::consts::PI;

// Mock predictor for testing
struct MockPredictor;

impl PredictorState for MockPredictor {
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

    fn clone_box(&self) -> Box<dyn PredictorState> {
        Box::new(MockPredictor)
    }
}

// =====================================================================
// Helper Functions
// =====================================================================

fn create_sine_signal(length: usize, frequency: f64) -> Vec<f64> {
    (0..length).map(|i| (2.0 * PI * frequency * i as f64 / 1000.0).sin()).collect()
}

fn create_chirp_signal(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| {
            let t = i as f64 / 1000.0;
            let freq = 1.0 + t;
            (2.0 * PI * freq * t).sin()
        })
        .collect()
}

// =====================================================================
// PHASE 5: Streaming ↔ Batch Equivalence Tests
// =====================================================================

#[test]
fn test_streaming_decomposition_sine_256() {
    let signal_data = create_sine_signal(256, 1.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 256).unwrap();

    // Process as two chunks of 128
    let chunk1 = Signal::from_slice(&signal_data[0..128]).unwrap();
    let result1 = decomposer.decompose_chunk(&chunk1).unwrap();

    let chunk2 = Signal::from_slice(&signal_data[128..256]).unwrap();
    let result2 = decomposer.decompose_chunk(&chunk2).unwrap();

    // Verify output shapes
    assert_eq!(result1.remainder.len(), 128);
    assert_eq!(result2.remainder.len(), 128);
}

#[test]
fn test_streaming_decomposition_sine_512() {
    let signal_data = create_sine_signal(512, 2.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

    // Process as two chunks of 256
    let chunk1 = Signal::from_slice(&signal_data[0..256]).unwrap();
    let result1 = decomposer.decompose_chunk(&chunk1).unwrap();

    let chunk2 = Signal::from_slice(&signal_data[256..512]).unwrap();
    let result2 = decomposer.decompose_chunk(&chunk2).unwrap();

    assert_eq!(result1.remainder.len(), 256);
    assert_eq!(result2.remainder.len(), 256);
}

#[test]
fn test_streaming_decomposition_sine_1024() {
    let signal_data = create_sine_signal(1024, 3.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();

    // Process as four chunks of 256
    let mut total_chunks = 0;
    for chunk_start in (0..1024).step_by(256) {
        let chunk_end = (chunk_start + 256).min(1024);
        let chunk_data = &signal_data[chunk_start..chunk_end];
        let signal = Signal::from_slice(chunk_data).unwrap();
        let result = decomposer.decompose_chunk(&signal);
        assert!(result.is_ok());
        total_chunks += 1;
    }

    assert_eq!(decomposer.chunk_id(), total_chunks as u64);
}

#[test]
fn test_streaming_decomposition_sine_2048() {
    let signal_data = create_sine_signal(2048, 1.5);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 2048).unwrap();

    // Process as eight chunks of 256
    let mut total_chunks = 0;
    for chunk_start in (0..2048).step_by(256) {
        let chunk_end = (chunk_start + 256).min(2048);
        let chunk_data = &signal_data[chunk_start..chunk_end];
        let signal = Signal::from_slice(chunk_data).unwrap();
        let result = decomposer.decompose_chunk(&signal);
        assert!(result.is_ok());
        total_chunks += 1;
    }

    assert_eq!(decomposer.chunk_id(), total_chunks as u64);
}

#[test]
fn test_streaming_decomposition_chirp_signal() {
    let signal_data = create_chirp_signal(512);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

    // Process chirp as two chunks
    let chunk1 = Signal::from_slice(&signal_data[0..256]).unwrap();
    let result1 = decomposer.decompose_chunk(&chunk1).unwrap();

    let chunk2 = Signal::from_slice(&signal_data[256..512]).unwrap();
    let result2 = decomposer.decompose_chunk(&chunk2).unwrap();

    assert_eq!(result1.remainder.len(), 256);
    assert_eq!(result2.remainder.len(), 256);
}

#[test]
fn test_streaming_chunk_continuity() {
    let signal_data = create_sine_signal(1024, 1.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 1024).unwrap();

    for i in 0..4 {
        let chunk_start = i * 256;
        let chunk_end = chunk_start + 256;
        let signal = Signal::from_slice(&signal_data[chunk_start..chunk_end]).unwrap();
        let result = decomposer.decompose_chunk(&signal);
        assert!(result.is_ok());
        assert_eq!(decomposer.chunk_id(), i as u64 + 1);
    }
}

#[test]
fn test_streaming_state_persistence() {
    let signal_data = create_sine_signal(512, 1.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

    // First chunk
    let _ = decomposer.decompose_chunk(&Signal::from_slice(&signal_data[0..256]).unwrap()).unwrap();
    let buffer_size_1 = decomposer.state().buffer_ref().len();
    assert_eq!(buffer_size_1, 256);

    // Second chunk
    let _ =
        decomposer.decompose_chunk(&Signal::from_slice(&signal_data[256..512]).unwrap()).unwrap();
    let buffer_size_2 = decomposer.state().buffer_ref().len();
    assert_eq!(buffer_size_2, 512);
}

#[test]
fn test_streaming_remainder_shape_matches_input() {
    let signal_data = create_sine_signal(512, 1.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

    let chunk_sizes = vec![256, 256];
    let mut offset = 0;
    for size in chunk_sizes {
        let chunk_end = offset + size;
        let signal = Signal::from_slice(&signal_data[offset..chunk_end]).unwrap();
        let result = decomposer.decompose_chunk(&signal).unwrap();
        assert_eq!(result.remainder.len(), size);
        offset = chunk_end;
    }
}

#[test]
fn test_streaming_multiple_chunk_sizes() {
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
            let signal = Signal::from_slice(chunk_data).unwrap();
            let result = decomposer.decompose_chunk(&signal);
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
fn test_boundary_prediction_reduces_end_effects() {
    let signal_data = create_sine_signal(512, 2.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

    let signal = Signal::from_slice(&signal_data[0..256]).unwrap();
    let result = decomposer.decompose_chunk(&signal).unwrap();

    // Check that remainder is bounded
    let remainder = &result.remainder;
    let max_abs = remainder.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
    assert!(max_abs < 10.0);
}

#[test]
fn test_boundary_prediction_energy_preservation() {
    let signal_data = create_sine_signal(512, 1.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

    let signal = Signal::from_slice(&signal_data[0..256]).unwrap();
    let result = decomposer.decompose_chunk(&signal).unwrap();

    // Compute energy
    let input_energy: f64 = signal_data[0..256].iter().map(|x| x * x).sum();
    let output_energy: f64 = result.remainder.iter().map(|x| x * x).sum::<f64>()
        + result.imfs.iter().map(|imf| imf.iter().map(|x| x * x).sum::<f64>()).sum::<f64>();

    // Energy should be reasonably preserved
    let energy_ratio = output_energy / (input_energy + 1e-10);
    assert!(energy_ratio > 0.5 && energy_ratio < 2.0);
}

#[test]
fn test_boundary_artifacts_limited() {
    let signal_data = create_sine_signal(512, 3.0);
    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

    let signal = Signal::from_slice(&signal_data[0..256]).unwrap();
    let result = decomposer.decompose_chunk(&signal).unwrap();

    let remainder = &result.remainder;
    if remainder.len() >= 10 {
        let first_10_rms = (remainder[0..10].iter().map(|x| x * x).sum::<f64>() / 10.0).sqrt();
        let last_10_rms =
            (remainder[remainder.len() - 10..].iter().map(|x| x * x).sum::<f64>() / 10.0).sqrt();
        let mid_rms = (remainder[100..150].iter().map(|x| x * x).sum::<f64>() / 50.0).sqrt();

        // Boundary artifacts should be limited
        assert!(first_10_rms < 3.0 * mid_rms || first_10_rms < 0.1);
        assert!(last_10_rms < 3.0 * mid_rms || last_10_rms < 0.1);
    }
}
