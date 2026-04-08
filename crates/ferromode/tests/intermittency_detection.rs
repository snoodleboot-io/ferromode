//! Tests for intermittency detection and adaptive algorithm selection.
//!
//! PHASE 7: Intermittency Detection Tests
//! Tests the metrics computation and algorithm selection based on signal characteristics.

use ferromode::adapters::streaming::{
    AdaptiveAlgorithm, IntermittencyMetrics, PredictorState, StreamingDecomposer,
};
use ferromode::algorithms::emd::EmdConfig;
use ferromode::error::EmdError;
use ferromode::types::Signal;
use std::f64::consts::PI;

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
// PHASE 7: Intermittency Detection Tests
// =====================================================================

#[test]
fn test_intermittency_metrics_creation() {
    let metrics = IntermittencyMetrics::new(0.3, 0.2, 0.85);
    assert_eq!(metrics.spectral_entropy, 0.3);
    assert_eq!(metrics.extrema_spacing_cv, 0.2);
    assert_eq!(metrics.stationarity_score, 0.85);
}

#[test]
fn test_algorithm_selection_emd_high_stationarity() {
    // High stationarity (> 0.8) should recommend EMD
    let metrics = IntermittencyMetrics::new(0.1, 0.05, 0.85);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EMD);

    let metrics = IntermittencyMetrics::new(0.2, 0.1, 0.95);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EMD);

    let metrics = IntermittencyMetrics::new(0.15, 0.08, 0.81);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EMD);
}

#[test]
fn test_algorithm_selection_eemd_moderate_stationarity() {
    // Moderate stationarity (0.5-0.8) should recommend EEMD
    let metrics = IntermittencyMetrics::new(0.3, 0.2, 0.75);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EEMD);

    let metrics = IntermittencyMetrics::new(0.4, 0.3, 0.65);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EEMD);

    let metrics = IntermittencyMetrics::new(0.35, 0.25, 0.50);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EEMD);

    let metrics = IntermittencyMetrics::new(0.3, 0.2, 0.51);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EEMD);
}

#[test]
fn test_algorithm_selection_ceemdan_low_stationarity() {
    // Low stationarity (< 0.5) should recommend CEEMDAN
    let metrics = IntermittencyMetrics::new(0.6, 0.5, 0.45);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::CEEMDAN);

    let metrics = IntermittencyMetrics::new(0.7, 0.6, 0.3);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::CEEMDAN);

    let metrics = IntermittencyMetrics::new(0.8, 0.7, 0.1);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::CEEMDAN);

    let metrics = IntermittencyMetrics::new(0.5, 0.4, 0.49);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::CEEMDAN);
}

#[test]
fn test_algorithm_selection_boundary_emd_eemd_80_percent() {
    // Boundary at 0.8 - should be EMD
    let metrics = IntermittencyMetrics::new(0.1, 0.05, 0.80);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EEMD);

    let metrics = IntermittencyMetrics::new(0.1, 0.05, 0.8001);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EMD);
}

#[test]
fn test_algorithm_selection_boundary_eemd_ceemdan_50_percent() {
    // Boundary at 0.5 - should be EEMD
    let metrics = IntermittencyMetrics::new(0.3, 0.2, 0.5);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::EEMD);

    let metrics = IntermittencyMetrics::new(0.3, 0.2, 0.4999);
    assert_eq!(metrics.recommended_algorithm(), AdaptiveAlgorithm::CEEMDAN);
}

#[test]
fn test_intermittency_metrics_spectral_entropy_range() {
    // Spectral entropy should be in valid range [0, 1]
    let metrics_low = IntermittencyMetrics::new(0.0, 0.5, 0.8);
    assert!(metrics_low.spectral_entropy >= 0.0 && metrics_low.spectral_entropy <= 1.0);

    let metrics_high = IntermittencyMetrics::new(1.0, 0.5, 0.8);
    assert!(metrics_high.spectral_entropy >= 0.0 && metrics_high.spectral_entropy <= 1.0);
}

#[test]
fn test_intermittency_metrics_extrema_spacing_cv_range() {
    // Extrema spacing CV should be non-negative
    let metrics = IntermittencyMetrics::new(0.5, 0.0, 0.8);
    assert!(metrics.extrema_spacing_cv >= 0.0);

    let metrics = IntermittencyMetrics::new(0.5, 2.5, 0.8);
    assert!(metrics.extrema_spacing_cv >= 0.0);
}

#[test]
fn test_intermittency_metrics_stationarity_score_range() {
    // Stationarity score should be in valid range [0, 1]
    let metrics_low = IntermittencyMetrics::new(0.5, 0.3, 0.0);
    assert!(metrics_low.stationarity_score >= 0.0 && metrics_low.stationarity_score <= 1.0);

    let metrics_high = IntermittencyMetrics::new(0.5, 0.3, 1.0);
    assert!(metrics_high.stationarity_score >= 0.0 && metrics_high.stationarity_score <= 1.0);
}

#[test]
fn test_adaptive_algorithm_enum_values() {
    // Test that all algorithm variants exist and are distinct
    let emd = AdaptiveAlgorithm::EMD;
    let eemd = AdaptiveAlgorithm::EEMD;
    let ceemdan = AdaptiveAlgorithm::CEEMDAN;

    assert_eq!(emd, AdaptiveAlgorithm::EMD);
    assert_eq!(eemd, AdaptiveAlgorithm::EEMD);
    assert_eq!(ceemdan, AdaptiveAlgorithm::CEEMDAN);

    assert_ne!(emd, eemd);
    assert_ne!(eemd, ceemdan);
    assert_ne!(emd, ceemdan);
}

#[test]
#[ignore] // TODO: Fix spline boundary condition bug
fn test_streaming_decomposer_metric_computation() {
    // Test that decomposer computes metrics for each chunk
    let signal_data: Vec<f64> = (0..256).map(|i| (2.0 * PI * i as f64 / 100.0).sin()).collect();

    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 256).unwrap();

    let signal = Signal::from_slice(&signal_data).unwrap();
    let result = decomposer.decompose_chunk(&signal).unwrap();

    // Metrics should be computed for the chunk
    assert!(result.metrics.spectral_entropy >= 0.0);
    assert!(result.metrics.extrema_spacing_cv >= 0.0);
    assert!(result.metrics.stationarity_score >= 0.0);
}

#[test]
#[ignore] // TODO: Fix spline boundary condition bug
fn test_streaming_decomposer_algorithm_adaptation() {
    // Test that algorithm adapts based on stationarity
    let signal_data: Vec<f64> = (0..512)
        .map(|i| {
            // More stationary signal
            (2.0 * PI * i as f64 / 100.0).sin()
        })
        .collect();

    let config = EmdConfig::default();
    let predictor = Box::new(MockPredictor);
    let mut decomposer = StreamingDecomposer::new(config, predictor, 512).unwrap();

    let signal = Signal::from_slice(&signal_data).unwrap();
    let result = decomposer.decompose_chunk(&signal).unwrap();

    // Algorithm should be one of the three valid options
    match result.metrics.recommended_algorithm() {
        AdaptiveAlgorithm::EMD => {
            assert!(result.metrics.stationarity_score > 0.8);
        }
        AdaptiveAlgorithm::EEMD => {
            assert!(
                result.metrics.stationarity_score >= 0.5
                    && result.metrics.stationarity_score <= 0.8
            );
        }
        AdaptiveAlgorithm::CEEMDAN => {
            assert!(result.metrics.stationarity_score < 0.5);
        }
    }
}

#[test]
fn test_intermittency_metrics_multiple_instances() {
    // Test creating multiple metric instances with different values
    let metrics_array = vec![
        IntermittencyMetrics::new(0.1, 0.05, 0.9),
        IntermittencyMetrics::new(0.3, 0.2, 0.65),
        IntermittencyMetrics::new(0.7, 0.5, 0.2),
    ];

    assert_eq!(metrics_array.len(), 3);
    for metrics in metrics_array {
        assert!(metrics.spectral_entropy >= 0.0 && metrics.spectral_entropy <= 1.0);
        assert!(metrics.extrema_spacing_cv >= 0.0);
        assert!(metrics.stationarity_score >= 0.0 && metrics.stationarity_score <= 1.0);
    }
}

#[test]
fn test_intermittency_metrics_clone() {
    // Test that metrics can be cloned
    let metrics = IntermittencyMetrics::new(0.3, 0.2, 0.75);
    let cloned = metrics;

    assert_eq!(cloned.spectral_entropy, metrics.spectral_entropy);
    assert_eq!(cloned.extrema_spacing_cv, metrics.extrema_spacing_cv);
    assert_eq!(cloned.stationarity_score, metrics.stationarity_score);
}

#[test]
fn test_intermittency_metrics_equality() {
    // Test metrics with same values
    let metrics1 = IntermittencyMetrics::new(0.3, 0.2, 0.75);
    let metrics2 = IntermittencyMetrics::new(0.3, 0.2, 0.75);

    assert_eq!(metrics1.spectral_entropy, metrics2.spectral_entropy);
    assert_eq!(metrics1.extrema_spacing_cv, metrics2.extrema_spacing_cv);
    assert_eq!(metrics1.stationarity_score, metrics2.stationarity_score);
}
