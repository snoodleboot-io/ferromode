//! Time-frequency entropy analysis.
//!
//! This module provides sliding window spectral entropy for time-frequency
//! representation and complexity scoring.

use crate::analysis::entropy::{spectral_entropy_normalized, EntropyAnalysis};
use crate::error::EmdError;

/// Compute sliding window spectral entropy (time-frequency representation).
///
/// Applies FFT over sliding windows of the signal to produce a time-frequency
/// entropy representation. Each window produces one entropy value.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of FFT window
/// * `overlap` - Overlap ratio in [0, 1) (0 = no overlap, 0.5 = 50% overlap)
///
/// # Returns
///
/// 2D vector where each row represents a time window's entropy values.
/// Currently each window produces one value (the spectral entropy).
///
/// # Errors
///
/// Returns error if signal is empty or invalid parameters provided.
///
/// # Example
///
/// ```ignore
/// # use ferromode::analysis::sliding_window_spectral_entropy;
/// let signal = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5];
/// let entropy_time_freq = sliding_window_spectral_entropy(&signal, 4, 0.5)?;
/// ```
pub fn sliding_window_spectral_entropy(
    signal: &[f64],
    window_size: usize,
    overlap: f64,
) -> Result<Vec<Vec<f64>>, EmdError> {
    if signal.is_empty() {
        return Err(EmdError::EmptySignal);
    }

    if window_size == 0 || window_size > signal.len() {
        return Err(EmdError::InvalidConfig(
            "window size must be positive and not exceed signal length".to_string(),
        ));
    }

    if overlap < 0.0 || overlap >= 1.0 {
        return Err(EmdError::InvalidConfig("overlap must be in [0, 1)".to_string()));
    }

    let stride = ((window_size as f64) * (1.0 - overlap)).ceil() as usize;
    let stride = stride.max(1); // Ensure stride is at least 1

    let mut result = Vec::new();
    let mut pos = 0;

    while pos + window_size <= signal.len() {
        let window = &signal[pos..pos + window_size];
        let entropy = spectral_entropy_normalized(window)?;
        result.push(vec![entropy]);
        pos += stride;
    }

    Ok(result)
}

/// Compute overall complexity score from entropy analysis.
///
/// Combines all three entropy metrics into a single [0, 1] complexity score.
/// This score indicates overall signal complexity: 0 = very regular, 1 = very complex.
///
/// # Arguments
///
/// * `analysis` - Entropy analysis result
///
/// # Returns
///
/// Complexity score in [0, 1]
///
/// # Formula
///
/// ```text
/// score = (mean_spectral + mean_permutation + normalized_sample) / 3
/// ```
///
/// # Example
///
/// ```ignore
/// # use ferromode::analysis::{EntropyAnalysis, complexity_score};
/// # let analysis = /* ... */;
/// let score = complexity_score(&analysis);
/// println!("Overall complexity: {}", score);
/// ```
pub fn complexity_score(analysis: &EntropyAnalysis) -> f64 {
    // Already computed in EntropyAnalysis.complexity_score
    // This function provides direct access to the value
    analysis.complexity_score
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::EntropyAnalysis;
    use crate::types::{AlgorithmType, DecompositionResult, ImfCollection};
    use std::time::Duration;

    #[test]
    fn test_sliding_window_spectral_entropy_basic() {
        let signal: Vec<f64> =
            (0..64).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin()).collect();

        let entropy_tf = sliding_window_spectral_entropy(&signal, 8, 0.0).unwrap();
        assert!(entropy_tf.len() > 0, "should produce entropy values");
        for values in &entropy_tf {
            assert_eq!(values.len(), 1, "each window should produce one value");
            assert!(values[0] >= 0.0 && values[0] <= 1.0, "entropy should be in [0, 1]");
        }
    }

    #[test]
    fn test_sliding_window_spectral_entropy_with_overlap() {
        let signal: Vec<f64> =
            (0..32).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin()).collect();

        let entropy_no_overlap = sliding_window_spectral_entropy(&signal, 8, 0.0).unwrap();
        let entropy_50_overlap = sliding_window_spectral_entropy(&signal, 8, 0.5).unwrap();

        // With overlap, we should get more windows
        assert!(entropy_50_overlap.len() > entropy_no_overlap.len());
    }

    #[test]
    fn test_sliding_window_spectral_entropy_empty_signal() {
        let signal: Vec<f64> = vec![];
        let result = sliding_window_spectral_entropy(&signal, 4, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sliding_window_spectral_entropy_invalid_window() {
        let signal = vec![1.0, 2.0, 3.0];
        let result = sliding_window_spectral_entropy(&signal, 0, 0.5);
        assert!(result.is_err());

        let result = sliding_window_spectral_entropy(&signal, 10, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_sliding_window_spectral_entropy_invalid_overlap() {
        let signal = vec![1.0, 2.0, 3.0, 4.0];

        let result = sliding_window_spectral_entropy(&signal, 2, -0.1);
        assert!(result.is_err());

        let result = sliding_window_spectral_entropy(&signal, 2, 1.0);
        assert!(result.is_err());

        let result = sliding_window_spectral_entropy(&signal, 2, 1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_complexity_score_returns_valid_range() {
        let imfs = vec![vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0], vec![0.1, 0.2, 0.1, 0.2, 0.1, 0.2]];
        let residue = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        let imf_collection = ImfCollection::new(imfs, residue);
        let result = DecompositionResult::new(
            AlgorithmType::EMD,
            imf_collection,
            Duration::from_secs(1),
            5,
            "test".to_string(),
        );

        let analysis = EntropyAnalysis::from_decomposition(&result).unwrap();
        let score = complexity_score(&analysis);

        assert!(score >= 0.0 && score <= 1.0, "complexity score must be in [0, 1]");
    }

    #[test]
    fn test_complexity_score_constant_signal() {
        // Constant signal should have very low complexity
        let imfs = vec![vec![1.0; 10]];
        let residue = vec![1.0; 10];

        let imf_collection = ImfCollection::new(imfs, residue);
        let result = DecompositionResult::new(
            AlgorithmType::EMD,
            imf_collection,
            Duration::from_secs(1),
            5,
            "test".to_string(),
        );

        let analysis = EntropyAnalysis::from_decomposition(&result).unwrap();
        let score = complexity_score(&analysis);

        assert!(score < 0.3, "constant signal should have low complexity");
    }

    #[test]
    fn test_complexity_score_multiple_imfs() {
        // Multiple IMFs with varying complexity
        let imfs = vec![
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0], // Regular
            vec![1.0, 2.0, 3.0, 1.5, 2.5, 0.8], // More complex
        ];
        let residue = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        let imf_collection = ImfCollection::new(imfs, residue);
        let result = DecompositionResult::new(
            AlgorithmType::EMD,
            imf_collection,
            Duration::from_secs(1),
            5,
            "test".to_string(),
        );

        let analysis = EntropyAnalysis::from_decomposition(&result).unwrap();

        // Check that means are computed
        assert!(analysis.mean_spectral_entropy >= 0.0);
        assert!(analysis.mean_permutation_entropy >= 0.0);
        assert!(analysis.mean_sample_entropy >= 0.0);

        // Complexity score should be in valid range
        let score = complexity_score(&analysis);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
