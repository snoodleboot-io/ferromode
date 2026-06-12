//! Entropy-based complexity metrics for signal analysis.
//!
//! This module implements three entropy measures for analyzing signal complexity:
//!
//! - **Spectral Entropy**: Measures disorder in the frequency domain
//! - **Permutation Entropy**: Captures temporal complexity via ordinal patterns
//! - **Sample Entropy**: Measures self-similarity and regularity

use crate::error::EmdError;
use crate::types::DecompositionResult;
use num_complex::Complex64;
use rustfft::FftPlanner;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper Functions
// ---------------------------------------------------------------------------

/// Pad signal to next power of 2 for FFT
fn pad_to_power_of_two(signal: &[f64]) -> (Vec<Complex64>, usize) {
    let n = signal.len();
    let _padded_len = 1 << (n as u32).next_power_of_two().trailing_zeros();
    let mut padded_len = 1;
    while padded_len < n {
        padded_len <<= 1;
    }

    let buffer: Vec<Complex64> = signal
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .chain(std::iter::repeat(Complex64::new(0.0, 0.0)))
        .take(padded_len)
        .collect();
    (buffer, n)
}

/// Compute FFT of a real signal
fn compute_fft(signal: &[f64]) -> Result<Vec<Complex64>, EmdError> {
    if signal.is_empty() {
        return Err(EmdError::EmptySignal);
    }

    let (mut buffer, _) = pad_to_power_of_two(signal);
    let n = buffer.len();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    Ok(buffer)
}

/// Calculate standard deviation of a signal
fn standard_deviation(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let variance = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / signal.len() as f64;

    variance.sqrt()
}

/// Compute factorial of a number
fn factorial(n: usize) -> u64 {
    (1..=n as u64).product()
}

// ---------------------------------------------------------------------------
// T-330: Spectral Entropy
// ---------------------------------------------------------------------------

/// Compute spectral entropy of a signal.
///
/// Spectral entropy measures the disorder in the frequency domain.
/// Lower values indicate narrowband signals (concentrated energy),
/// higher values indicate broadband/noise (distributed energy).
///
/// # Formula
///
/// Given power spectrum `P(f)` normalized as probability distribution `p(f)`:
/// ```text
/// H = -∑ p(f) · ln(p(f))
/// ```
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// Spectral entropy value (typically 0 to ln(N))
///
/// # Errors
///
/// Returns `EmdError::EmptySignal` if signal is empty.
///
/// # Example
///
/// ```
/// # use ferromode::analysis::spectral_entropy;
/// # let signal = vec![1.0, 2.0, 1.0, 2.0];
/// # let entropy = spectral_entropy(&signal).unwrap();
/// # assert!(entropy >= 0.0);
/// ```
pub fn spectral_entropy(signal: &[f64]) -> Result<f64, EmdError> {
    if signal.is_empty() {
        return Err(EmdError::EmptySignal);
    }

    // Compute FFT
    let fft_result = compute_fft(signal)?;

    // Compute power spectrum: P = |X|²
    let power_spectrum: Vec<f64> = fft_result.iter().map(|c| c.norm().powi(2)).collect();

    // Normalize to probability distribution
    let total_power: f64 = power_spectrum.iter().sum();
    if total_power < 1e-10 {
        return Ok(0.0); // Zero power = zero entropy
    }

    let p_norm: Vec<f64> = power_spectrum.iter().map(|p| p / total_power).collect();

    // Compute Shannon entropy: H = -∑ p * ln(p)
    // Skip zero probabilities to avoid -inf terms
    let entropy: f64 = p_norm.iter().filter(|&&p| p > 1e-10).map(|&p| -p * p.ln()).sum();

    Ok(entropy)
}

/// Spectral entropy normalized to [0, 1].
///
/// Divides spectral entropy by its maximum value `ln(N)` to get
/// a normalized measure where 0 = pure sinusoid, 1 = white noise.
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// Normalized spectral entropy in [0, 1]
///
/// # Example
///
/// ```
/// # use ferromode::analysis::spectral_entropy_normalized;
/// # let signal = vec![1.0, 2.0, 1.0, 2.0];
/// # let entropy = spectral_entropy_normalized(&signal).unwrap();
/// # assert!(entropy >= 0.0 && entropy <= 1.0);
/// ```
pub fn spectral_entropy_normalized(signal: &[f64]) -> Result<f64, EmdError> {
    let entropy = spectral_entropy(signal)?;
    let max_entropy = (signal.len() as f64).ln();
    if max_entropy > 0.0 {
        Ok((entropy / max_entropy).min(1.0))
    } else {
        Ok(0.0)
    }
}

// ---------------------------------------------------------------------------
// T-331: Permutation Entropy
// ---------------------------------------------------------------------------

/// Compute permutation entropy of a signal.
///
/// Permutation entropy measures temporal complexity via ordinal patterns.
/// It captures nonlinear and chaotic behavior by counting unique ordinal
/// permutations in sliding windows.
///
/// Lower values indicate more regular/periodic signals,
/// higher values indicate more random/chaotic signals.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `embedding_dim` - Embedding dimension (typically 3-4)
///
/// # Returns
///
/// Permutation entropy value (typically 0 to ln(m!))
///
/// # Errors
///
/// Returns `EmdError::InsufficientData` if signal is shorter than `embedding_dim`.
///
/// # Example
///
/// ```
/// # use ferromode::analysis::permutation_entropy;
/// # let signal = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0];
/// # let entropy = permutation_entropy(&signal, 3).unwrap();
/// # assert!(entropy >= 0.0);
/// ```
pub fn permutation_entropy(signal: &[f64], embedding_dim: usize) -> Result<f64, EmdError> {
    if signal.len() < embedding_dim {
        return Err(EmdError::InsufficientData);
    }

    // Count ordinal patterns
    let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();

    for i in 0..(signal.len() - embedding_dim + 1) {
        let window = &signal[i..i + embedding_dim];

        // Get ordinal pattern: ranking of values
        let mut indices: Vec<usize> = (0..embedding_dim).collect();
        indices.sort_by(|&a, &b| {
            window[a].partial_cmp(&window[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        *pattern_counts.entry(indices).or_insert(0) += 1;
    }

    // Compute Shannon entropy of patterns
    let total_patterns = (signal.len() - embedding_dim + 1) as f64;
    let mut entropy = 0.0;

    for count in pattern_counts.values() {
        let p = *count as f64 / total_patterns;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }

    Ok(entropy)
}

/// Permutation entropy normalized to [0, 1].
///
/// Divides permutation entropy by its maximum value `ln(m!)` where `m` is
/// the embedding dimension.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `embedding_dim` - Embedding dimension
///
/// # Returns
///
/// Normalized permutation entropy in [0, 1]
///
/// # Example
///
/// ```
/// # use ferromode::analysis::permutation_entropy_normalized;
/// # let signal = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0];
/// # let entropy = permutation_entropy_normalized(&signal, 3).unwrap();
/// # assert!(entropy >= 0.0 && entropy <= 1.0);
/// ```
pub fn permutation_entropy_normalized(
    signal: &[f64],
    embedding_dim: usize,
) -> Result<f64, EmdError> {
    let entropy = permutation_entropy(signal, embedding_dim)?;
    let max_entropy = (factorial(embedding_dim) as f64).ln();
    if max_entropy > 0.0 {
        Ok((entropy / max_entropy).min(1.0))
    } else {
        Ok(0.0)
    }
}

// ---------------------------------------------------------------------------
// T-332: Sample Entropy
// ---------------------------------------------------------------------------

/// Count matching templates within tolerance threshold.
///
/// Uses Chebyshev distance (maximum absolute difference).
fn count_matching_templates(
    signal: &[f64],
    embedding_dim: usize,
    tolerance: f64,
) -> Result<u64, EmdError> {
    let mut count = 0u64;

    for i in 0..(signal.len() - embedding_dim) {
        let template_i = &signal[i..i + embedding_dim];

        for j in (i + 1)..(signal.len() - embedding_dim) {
            let template_j = &signal[j..j + embedding_dim];

            // Chebyshev distance: max absolute difference
            let max_diff = template_i
                .iter()
                .zip(template_j.iter())
                .map(|(&a, &b)| (a - b).abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            if max_diff <= tolerance {
                count += 1;
            }
        }
    }

    Ok(count)
}

/// Compute sample entropy of a signal.
///
/// Sample entropy measures self-similarity and regularity.
/// Lower values indicate more regular/predictable signals,
/// higher values indicate more complex/random signals.
///
/// More robust than other entropy measures but slower O(N²).
///
/// # Formula
///
/// ```text
/// SampEn(m, r, N) = -ln(C(m+1, r) / C(m, r))
/// ```
///
/// where `C(m, r)` is the count of template matches within tolerance `r`.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `embedding_dim` - Embedding dimension (typically 1-3)
/// * `tolerance` - Similarity threshold (None = 0.2 * std_dev)
///
/// # Returns
///
/// Sample entropy value (typically 0 to ~2)
///
/// # Errors
///
/// Returns `EmdError::InsufficientData` if signal is too short.
///
/// # Example
///
/// ```
/// # use ferromode::analysis::sample_entropy;
/// # let signal = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
/// # let entropy = sample_entropy(&signal, 2, None).unwrap();
/// # assert!(entropy >= 0.0);
/// ```
pub fn sample_entropy(
    signal: &[f64],
    embedding_dim: usize,
    tolerance: Option<f64>,
) -> Result<f64, EmdError> {
    if signal.len() < embedding_dim + 1 {
        return Err(EmdError::InsufficientData);
    }

    // Default tolerance: 0.2 * standard deviation
    let tolerance = tolerance.unwrap_or_else(|| 0.2 * standard_deviation(signal));

    // Count template matches for dimension m and m+1
    let count_m = count_matching_templates(signal, embedding_dim, tolerance)?;
    let count_m1 = count_matching_templates(signal, embedding_dim + 1, tolerance)?;

    // SampEn = -ln(count_m+1 / count_m)
    // If either count is 0, return 0 (complete regularity)
    if count_m == 0 || count_m1 == 0 {
        Ok(0.0)
    } else {
        let ratio = count_m1 as f64 / count_m as f64;
        Ok(-ratio.ln())
    }
}

// ---------------------------------------------------------------------------
// Trait for Extensibility
// ---------------------------------------------------------------------------

/// Trait for entropy metric implementations.
///
/// Enables extensible entropy analysis with pluggable implementations.
pub trait EntropyMetric: Send + Sync {
    /// Compute entropy metric for a signal
    fn compute(&self, signal: &[f64]) -> Result<f64, EmdError>;

    /// Name of this entropy metric
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Analysis Result Type
// ---------------------------------------------------------------------------

/// Complete entropy analysis combining all three metrics.
///
/// Computed from a decomposition result, this structure contains
/// individual entropy values for each IMF plus aggregate statistics.
#[derive(Clone, Debug)]
pub struct EntropyAnalysis {
    /// Normalized spectral entropy for each IMF
    pub spectral_entropy: Vec<f64>,

    /// Normalized permutation entropy for each IMF
    pub permutation_entropy: Vec<f64>,

    /// Sample entropy for each IMF
    pub sample_entropy: Vec<f64>,

    /// Mean spectral entropy across IMFs
    pub mean_spectral_entropy: f64,

    /// Mean permutation entropy across IMFs
    pub mean_permutation_entropy: f64,

    /// Mean sample entropy across IMFs
    pub mean_sample_entropy: f64,

    /// Overall complexity score [0, 1]
    pub complexity_score: f64,

    /// Indices of high-complexity IMFs (entropy > 0.7)
    pub high_complexity_imfs: Vec<usize>,
}

impl EntropyAnalysis {
    /// Analyze entropy of all IMFs from a decomposition result.
    ///
    /// Computes all three entropy metrics for each IMF and aggregates
    /// into summary statistics.
    ///
    /// # Arguments
    ///
    /// * `decomposition` - Decomposition result with IMFs
    ///
    /// # Returns
    ///
    /// Complete entropy analysis
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use ferromode::analysis::EntropyAnalysis;
    /// # let decomposition = /* ... */;
    /// let analysis = EntropyAnalysis::from_decomposition(&decomposition)?;
    /// println!("Complexity: {}", analysis.complexity_score);
    /// ```
    pub fn from_decomposition(decomposition: &DecompositionResult) -> Result<Self, EmdError> {
        let imfs = &decomposition.imfs.imfs;

        if imfs.is_empty() {
            return Err(EmdError::InsufficientData);
        }

        let mut spectral = Vec::new();
        let mut permutation = Vec::new();
        let mut sample = Vec::new();

        // Compute entropy for each IMF
        for imf in imfs {
            spectral.push(spectral_entropy_normalized(imf)?);
            permutation.push(permutation_entropy_normalized(imf, 3)?);
            sample.push(sample_entropy(imf, 2, None)?);
        }

        // Compute means
        let mean_spectral = spectral.iter().sum::<f64>() / spectral.len() as f64;
        let mean_permutation = permutation.iter().sum::<f64>() / permutation.len() as f64;
        let mean_sample = sample.iter().sum::<f64>() / sample.len() as f64;

        // Complexity score: average of three metrics
        // Normalize sample entropy to [0, 1] first (typical max ~2)
        let normalized_sample = (mean_sample / 2.0).min(1.0);
        let complexity_score = (mean_spectral + mean_permutation + normalized_sample) / 3.0;

        // Identify high-complexity IMFs (spectral entropy > 0.7)
        let high_complexity_imfs =
            spectral.iter().enumerate().filter(|(_, &e)| e > 0.7).map(|(i, _)| i).collect();

        Ok(EntropyAnalysis {
            spectral_entropy: spectral,
            permutation_entropy: permutation,
            sample_entropy: sample,
            mean_spectral_entropy: mean_spectral,
            mean_permutation_entropy: mean_permutation,
            mean_sample_entropy: mean_sample,
            complexity_score,
            high_complexity_imfs,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // T-330: Spectral Entropy Tests

    #[test]
    fn test_spectral_entropy_constant_signal() {
        // Constant signal has all energy at DC, low entropy
        let signal = vec![1.0; 16];
        let entropy = spectral_entropy(&signal).unwrap();
        assert!(entropy < 0.5, "constant signal should have low entropy");
    }

    #[test]
    fn test_spectral_entropy_sinusoid() {
        // Pure sinusoid has energy at one frequency, low entropy
        let signal: Vec<f64> =
            (0..64).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 64.0).sin()).collect();
        let entropy = spectral_entropy(&signal).unwrap();
        assert!(entropy < 2.0, "sinusoid should have relatively low entropy");
    }

    #[test]
    fn test_spectral_entropy_normalized_range() {
        let signal: Vec<f64> =
            (0..64).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 64.0).sin()).collect();
        let entropy = spectral_entropy_normalized(&signal).unwrap();
        assert!(entropy >= 0.0 && entropy <= 1.0, "normalized entropy must be in [0, 1]");
    }

    #[test]
    fn test_spectral_entropy_empty_signal() {
        let signal: Vec<f64> = vec![];
        let result = spectral_entropy(&signal);
        assert!(result.is_err());
    }

    // T-331: Permutation Entropy Tests

    #[test]
    fn test_permutation_entropy_constant_signal() {
        // Constant signal has one pattern, entropy = 0
        let signal = vec![1.0; 10];
        let entropy = permutation_entropy(&signal, 3).unwrap();
        assert!(entropy < 0.1, "constant signal should have zero entropy");
    }

    #[test]
    fn test_permutation_entropy_sine() {
        // Periodic signal has limited patterns
        let signal: Vec<f64> =
            (0..64).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin()).collect();
        let entropy = permutation_entropy(&signal, 3).unwrap();
        assert!(entropy > 0.1, "periodic signal should have nonzero entropy");
        assert!(entropy < 2.0, "periodic signal should have limited entropy");
    }

    #[test]
    fn test_permutation_entropy_normalized() {
        let signal: Vec<f64> =
            (0..64).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin()).collect();
        let entropy = permutation_entropy_normalized(&signal, 3).unwrap();
        assert!(entropy >= 0.0 && entropy <= 1.0, "normalized entropy must be in [0, 1]");
    }

    #[test]
    fn test_permutation_entropy_insufficient_data() {
        let signal = vec![1.0, 2.0];
        let result = permutation_entropy(&signal, 5);
        assert!(result.is_err());
    }

    // T-332: Sample Entropy Tests

    #[test]
    fn test_sample_entropy_constant_signal() {
        // Constant signal is perfectly regular, entropy should be very low
        // For a constant signal all values are identical, so tolerance is 0
        // All matches still occur, but entropy approaches 0 as predictability increases
        let signal = vec![1.0; 20];
        let entropy = sample_entropy(&signal, 2, None).unwrap();
        assert!(entropy < 0.2, "constant signal should have very low entropy");
    }

    #[test]
    fn test_sample_entropy_sine() {
        // Periodic signal has low entropy
        let signal: Vec<f64> =
            (0..100).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 32.0).sin()).collect();
        let entropy = sample_entropy(&signal, 2, None).unwrap();
        assert!(entropy > 0.0, "sine signal should have nonzero entropy");
        assert!(entropy < 1.0, "sine signal should have moderate entropy");
    }

    #[test]
    fn test_sample_entropy_custom_tolerance() {
        let signal = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5];
        let entropy_auto = sample_entropy(&signal, 2, None).unwrap();
        let entropy_tight = sample_entropy(&signal, 2, Some(0.1)).unwrap();

        // Tighter tolerance should give higher entropy (fewer matches)
        assert!(entropy_tight >= entropy_auto || (entropy_tight - entropy_auto).abs() < 0.01);
    }

    #[test]
    fn test_sample_entropy_insufficient_data() {
        let signal = vec![1.0, 2.0];
        let result = sample_entropy(&signal, 5, None);
        assert!(result.is_err());
    }

    // Integration Tests

    #[test]
    fn test_spectral_entropy_white_noise() {
        // White noise has higher entropy than sinusoid
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut signal: Vec<f64> = (-50..50).map(|i| i as f64).collect();
        signal.shuffle(&mut rng);

        let entropy_sine: Vec<f64> =
            (0..64).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 64.0).sin()).collect();
        let entropy_sine = spectral_entropy(&entropy_sine).unwrap();

        let entropy_noise = spectral_entropy(&signal).unwrap();
        assert!(entropy_noise > entropy_sine, "white noise should have higher entropy than sine");
    }

    #[test]
    fn test_entropy_metrics_consistency() {
        // All metrics should be deterministic
        let signal = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5, 3.5, 2.0];

        let se1 = spectral_entropy(&signal).unwrap();
        let se2 = spectral_entropy(&signal).unwrap();
        assert_eq!(se1, se2);

        let pe1 = permutation_entropy(&signal, 3).unwrap();
        let pe2 = permutation_entropy(&signal, 3).unwrap();
        assert_eq!(pe1, pe2);

        let sae1 = sample_entropy(&signal, 2, Some(0.5)).unwrap();
        let sae2 = sample_entropy(&signal, 2, Some(0.5)).unwrap();
        assert_eq!(sae1, sae2);
    }

    #[test]
    fn test_entropy_analysis_structure() {
        // Test that EntropyAnalysis can be created from valid data
        use crate::types::{AlgorithmType, ImfCollection};
        use std::time::Duration;

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
        assert_eq!(analysis.spectral_entropy.len(), 2);
        assert_eq!(analysis.permutation_entropy.len(), 2);
        assert_eq!(analysis.sample_entropy.len(), 2);
        assert!(analysis.complexity_score >= 0.0 && analysis.complexity_score <= 1.0);
    }
}
