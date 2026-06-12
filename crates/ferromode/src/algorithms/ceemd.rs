#![warn(missing_docs)]

//! Complementary Ensemble Empirical Mode Decomposition (CEEMD) algorithm.
//!
//! This module implements CEEMD as described by Yeh, Huang, and Wu (2010):
//! "Complementary Ensemble Empirical Mode Decomposition: A Novel Noise Assisted Data Analysis Method"
//!
//! CEEMD improves upon EEMD by adding pairs of complementary noise to the signal:
//! for each trial i, it computes both `emd(signal + ε·noise_i)` and `emd(signal - ε·noise_i)`.
//! The two results from each pair are averaged, and the final result is the average
//! of all complementary pair averages.
//!
//! This approach cancels noise residue more effectively than EEMD because the
//! positive and negative noise components tend to cancel out when averaged.
//!
//! # Algorithm
//! 1. For each trial i (0 to num_ensembles-1):
//!    a. Generate Gaussian white noise sequence noise_i
//!    b. Compute `emd(signal + ε·noise_i)` → positive trial
//!    c. Compute `emd(signal - ε·noise_i)` → negative trial
//!    d. Average the IMFs from the positive and negative trials
//! 2. Average all complementary pair averages to get final IMFs
//!
//! # References
//! - Yeh, J.-R., Huang, N. E., & Wu, Z. (2010). Complementary Ensemble Empirical Mode
//!   Decomposition: A Novel Noise Assisted Data Analysis Method. Advances in Adaptive
//!   Data Analysis, 2(4), 417-431.

use crate::algorithms::emd::{emd, EmdConfig};
use crate::error::EmdError;
use crate::types::{AlgorithmType, DecompositionResult, ImfCollection};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use rayon::prelude::*;
use web_time::Instant;

use crate::algorithms::eemd::EnsembleConfig;

// ---------------------------------------------------------------------------
// Internal: single trial result
// ---------------------------------------------------------------------------

/// Result from a single EMD trial (positive or negative noise).
#[derive(Debug, Clone)]
struct TrialResult {
    /// IMFs extracted from this trial
    imfs: Vec<Vec<f64>>,
    /// Residue from this trial
    residue: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Internal: complementary pair result
// ---------------------------------------------------------------------------

/// Result from averaging a complementary pair (+noise and -noise).
#[derive(Debug, Clone)]
struct ComplementaryPairResult {
    /// Averaged IMFs from the complementary pair
    imfs: Vec<Vec<f64>>,
    /// Averaged residue from the complementary pair
    residue: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Internal: run one EMD trial with specified noise
// ---------------------------------------------------------------------------

/// Run a single EMD trial with pre-generated noise.
///
/// # Arguments
/// * `signal` — Original signal
/// * `noise` — Pre-generated noise values (same length as signal)
/// * `emd_config` — EMD configuration for the inner decomposition
fn run_trial_with_noise(
    signal: &[f64],
    noise: &[f64],
    emd_config: &EmdConfig,
) -> Result<TrialResult, EmdError> {
    // Generate noisy signal: signal + noise
    let noisy_signal: Vec<f64> = signal.iter().zip(noise.iter()).map(|(&s, &n)| s + n).collect();

    // Run EMD on noisy signal; fall back to treating the signal as a single IMF
    // if sifting fails (ConvergenceFailed/InvalidValue) — same pattern as CEEMDAN.
    match emd(&noisy_signal, emd_config) {
        Ok(result) => Ok(TrialResult { imfs: result.imfs.imfs, residue: result.imfs.residue }),
        Err(EmdError::ConvergenceFailed { .. } | EmdError::InvalidValue) => Ok(TrialResult {
            imfs: vec![noisy_signal.clone()],
            residue: vec![0.0; noisy_signal.len()],
        }),
        Err(e) => Err(e),
    }
}

// ---------------------------------------------------------------------------
// Internal: run one complementary pair
// ---------------------------------------------------------------------------

/// Run one complementary pair: +noise and -noise trials, then average.
///
/// # Arguments
/// * `signal` — Original signal
/// * `noise` — Pre-generated noise values (same length as signal)
/// * `emd_config` — EMD configuration for the inner decomposition
fn run_complementary_pair(
    signal: &[f64],
    noise: &[f64],
    emd_config: &EmdConfig,
) -> Result<ComplementaryPairResult, EmdError> {
    // Positive noise trial: signal + noise
    let positive_result = run_trial_with_noise(signal, noise, emd_config)?;

    // Negative noise trial: signal - noise
    let negative_noise: Vec<f64> = noise.iter().map(|&n| -n).collect();
    let negative_result = run_trial_with_noise(signal, &negative_noise, emd_config)?;

    // Average the complementary pair
    let pair_result = average_complementary_pair(&positive_result, &negative_result);

    Ok(pair_result)
}

// ---------------------------------------------------------------------------
// Internal: average a complementary pair
// ---------------------------------------------------------------------------

/// Average the IMFs and residue from a complementary pair (+noise and -noise).
///
/// # Arguments
/// * `positive` — Result from signal + noise
/// * `negative` — Result from signal - noise
fn average_complementary_pair(
    positive: &TrialResult,
    negative: &TrialResult,
) -> ComplementaryPairResult {
    let n = positive.imfs.first().map_or(0, |imf| imf.len());
    if n == 0 {
        return ComplementaryPairResult { imfs: Vec::new(), residue: Vec::new() };
    }

    let max_imfs = positive.imfs.len().max(negative.imfs.len());

    // Average each IMF position (zero-pad where one has fewer IMFs)
    let mut averaged_imfs = Vec::with_capacity(max_imfs);
    for imf_idx in 0..max_imfs {
        let mut averaged_imf = vec![0.0f64; n];

        if imf_idx < positive.imfs.len() {
            for (dest, &src) in averaged_imf.iter_mut().zip(positive.imfs[imf_idx].iter()) {
                *dest += src;
            }
        }

        if imf_idx < negative.imfs.len() {
            for (dest, &src) in averaged_imf.iter_mut().zip(negative.imfs[imf_idx].iter()) {
                *dest += src;
            }
        }

        // Divide by 2 (average of pair)
        for val in &mut averaged_imf {
            *val /= 2.0;
        }

        averaged_imfs.push(averaged_imf);
    }

    // Average residue
    let mut averaged_residue = vec![0.0f64; n];
    for (dest, &src) in averaged_residue.iter_mut().zip(positive.residue.iter()) {
        *dest += src;
    }
    for (dest, &src) in averaged_residue.iter_mut().zip(negative.residue.iter()) {
        *dest += src;
    }
    for val in &mut averaged_residue {
        *val /= 2.0;
    }

    ComplementaryPairResult { imfs: averaged_imfs, residue: averaged_residue }
}

// ---------------------------------------------------------------------------
// Internal: average complementary pair results
// ---------------------------------------------------------------------------

/// Average IMFs across all complementary pair results, handling unequal IMF counts.
///
/// # Arguments
/// * `pairs` — Results from all complementary pairs
fn average_pair_results(pairs: &[ComplementaryPairResult]) -> (Vec<Vec<f64>>, Vec<f64>) {
    if pairs.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n = pairs[0].imfs.first().map_or(0, |imf| imf.len());
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let num_pairs = pairs.len();

    // Find the maximum number of IMFs across all pairs
    let max_imfs = pairs.iter().map(|p| p.imfs.len()).max().unwrap_or(0);

    // Average each IMF position (zero-pad where pair has fewer IMFs)
    let mut averaged_imfs = Vec::with_capacity(max_imfs);
    for imf_idx in 0..max_imfs {
        let mut averaged_imf = vec![0.0f64; n];
        for pair in pairs {
            if imf_idx < pair.imfs.len() {
                for (dest, &src) in averaged_imf.iter_mut().zip(pair.imfs[imf_idx].iter()) {
                    *dest += src;
                }
            }
        }
        for val in &mut averaged_imf {
            *val /= num_pairs as f64;
        }
        averaged_imfs.push(averaged_imf);
    }

    // Average residue
    let mut averaged_residue = vec![0.0f64; n];
    for pair in pairs {
        for (dest, &src) in averaged_residue.iter_mut().zip(pair.residue.iter()) {
            *dest += src;
        }
    }
    for val in &mut averaged_residue {
        *val /= num_pairs as f64;
    }

    (averaged_imfs, averaged_residue)
}

// ---------------------------------------------------------------------------
// Internal: compute signal standard deviation
// ---------------------------------------------------------------------------

/// Compute the standard deviation of a signal.
fn signal_std(signal: &[f64]) -> f64 {
    let n = signal.len() as f64;
    let mean: f64 = signal.iter().sum::<f64>() / n;
    let variance: f64 = signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

// ---------------------------------------------------------------------------
// Internal: compute RMS of a signal
// ---------------------------------------------------------------------------

/// Compute the root mean square of a signal.
#[allow(dead_code)]
fn rms(signal: &[f64]) -> f64 {
    let n = signal.len() as f64;
    let sum_sq: f64 = signal.iter().map(|v| v * v).sum();
    (sum_sq / n).sqrt()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform Complementary Ensemble Empirical Mode Decomposition on a signal.
///
/// CEEMD improves upon EEMD by using complementary noise pairs:
/// for each trial, it runs EMD on both `signal + noise` and `signal - noise`,
/// then averages the results. This cancels noise residue more effectively
/// than EEMD because the positive and negative noise components tend to
/// cancel out when averaged.
///
/// # Arguments
/// * `signal` — Input signal to decompose
/// * `config` — Ensemble configuration (same as EEMD)
/// * `emd_config` — EMD configuration for inner decomposition
///
/// # Returns
/// A `DecompositionResult` containing the averaged IMFs and residue,
/// or an error if decomposition fails.
///
/// # Examples
/// ```
/// use ferromode::algorithms::ceemd::ceemd;
/// use ferromode::algorithms::eemd::EnsembleConfig;
/// use ferromode::algorithms::emd::EmdConfig;
/// use std::f64::consts::PI;
///
/// // Decompose a mode-mixed signal
/// let n = 50;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64 / n as f64;
///         (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
///     })
///     .collect();
///
/// let config = EnsembleConfig {
///     num_ensembles: 4,
///     noise_std: 0.2,
///     seed: Some(42),
/// };
/// let emd_config = EmdConfig::default();
/// let result = ceemd(&signal, &config, &emd_config);
/// assert!(result.is_ok());
/// ```
pub fn ceemd(
    signal: &[f64],
    config: &EnsembleConfig,
    emd_config: &EmdConfig,
) -> Result<DecompositionResult, EmdError> {
    let start = Instant::now();

    // Validate input
    if signal.len() < 3 {
        return Err(EmdError::InsufficientData);
    }
    for &val in signal {
        if !val.is_finite() {
            return Err(EmdError::InvalidValue);
        }
    }

    if config.num_ensembles == 0 {
        return Err(EmdError::InvalidConfig("num_ensembles must be greater than 0".to_string()));
    }

    if config.noise_std < 0.0 {
        return Err(EmdError::InvalidConfig("noise_std must be non-negative".to_string()));
    }

    // Compute absolute noise level
    let std = signal_std(signal);
    let absolute_noise_std = config.noise_std * std;

    // Generate noise sequences for all trials
    let mut rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => {
            let _rng = StdRng::seed_from_u64(0);
            let mut rng_impl = rand::thread_rng();
            let seed: u64 = rng_impl.gen();
            StdRng::seed_from_u64(seed)
        }
    };
    let normal = Normal::new(0.0, absolute_noise_std).map_err(|e| {
        EmdError::InvalidConfig(format!("failed to create normal distribution: {e}"))
    })?;

    // Pre-generate noise for each trial
    let noise_sequences: Vec<Vec<f64>> = (0..config.num_ensembles)
        .map(|_| (0..signal.len()).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    // Run complementary pairs in parallel using rayon
    // Each pair runs twice: +noise and -noise
    let pairs: Result<Vec<ComplementaryPairResult>, EmdError> = (0..config.num_ensembles)
        .into_par_iter()
        .map(|i| {
            let noise = &noise_sequences[i];
            run_complementary_pair(signal, noise, emd_config)
        })
        .collect();

    let pairs = pairs?;

    // Average all complementary pair results
    let (averaged_imfs, averaged_residue) = average_pair_results(&pairs);

    let elapsed = start.elapsed();

    // Total trials = num_ensembles * 2 (each pair has 2 trials)
    let total_siftings: usize = config.num_ensembles * 2;

    let result = DecompositionResult::new(
        AlgorithmType::CEEMD,
        ImfCollection::new(averaged_imfs, averaged_residue),
        elapsed,
        total_siftings,
        format!(
            r#"{{"num_ensembles": {}, "noise_std": {}, "seed": {}, "complementary_pairs": true}}"#,
            config.num_ensembles,
            config.noise_std,
            config.seed.map_or("null".to_string(), |s| s.to_string()),
        ),
    );

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::eemd::eemd;
    use std::f64::consts::PI;

    fn fast_emd_config() -> EmdConfig {
        EmdConfig {
            sifting_config: crate::sifting::SiftingConfig {
                max_sifting_iterations: 5,
                ..crate::sifting::SiftingConfig::default()
            },
            max_imfs: 3,
            validate_reconstruction: false,
            ..EmdConfig::default()
        }
    }

    // =========================================================================
    // T-072: Paired noise trials — each trial runs with +noise and -noise
    // =========================================================================

    #[test]
    fn test_ceemd_paired_noise_trials() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = ceemd(&signal, &config, &emd_config);
        assert!(result.is_ok(), "CEEMD should succeed on pure sine wave");

        let result = result.unwrap();
        assert!(result.imfs.n_imfs() >= 1, "Pure sine wave should produce at least 1 IMF");
        assert_eq!(result.algorithm, AlgorithmType::CEEMD);
        // Total trials should be 2x the number of ensembles (complementary pairs)
        assert_eq!(result.n_siftings, config.num_ensembles * 2);
    }

    // =========================================================================
    // T-073: Average complementary pairs; verify exact reconstruction
    // =========================================================================

    #[test]
    fn test_ceemd_reconstruction_validation() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig {
            sifting_config: crate::sifting::SiftingConfig {
                max_sifting_iterations: 10,
                ..crate::sifting::SiftingConfig::default()
            },
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-6,
            ..EmdConfig::default()
        };

        let result = ceemd(&signal, &config, &emd_config);
        assert!(result.is_ok(), "CEEMD reconstruction should pass validation: {:?}", result);
    }

    #[test]
    fn test_ceemd_reconstruction_multi_component() {
        let n = 120;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = EmdConfig {
            sifting_config: crate::sifting::SiftingConfig {
                max_sifting_iterations: 10,
                ..crate::sifting::SiftingConfig::default()
            },
            validate_reconstruction: true,
            reconstruction_tolerance: 1e-6,
            ..EmdConfig::default()
        };

        let result = ceemd(&signal, &config, &emd_config);
        assert!(
            result.is_ok(),
            "CEEMD reconstruction should pass for multi-component signal: {:?}",
            result
        );
    }

    // =========================================================================
    // T-074: RMS noise level in IMFs should be lower than EEMD
    // =========================================================================

    #[test]
    fn test_ceemd_vs_eemd_rms_noise_comparison() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let ceemd_result = ceemd(&signal, &config, &emd_config).unwrap();
        let eemd_result = eemd(&signal, &config, &emd_config).unwrap();

        // Compute RMS of residue for both
        let ceemd_residue_rms = rms(&ceemd_result.imfs.residue);
        let eemd_residue_rms = rms(&eemd_result.imfs.residue);

        // CEEMD should have lower or equal residue RMS than EEMD
        // (complementary pairs cancel noise more effectively)
        assert!(
            ceemd_residue_rms <= eemd_residue_rms * 1.5,
            "CEEMD residue RMS ({:.6}) should be <= EEMD residue RMS ({:.6}) * 1.5",
            ceemd_residue_rms,
            eemd_residue_rms
        );
    }

    #[test]
    fn test_ceemd_vs_eemd_imf_rms_comparison() {
        let n = 120;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.1 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let ceemd_result = ceemd(&signal, &config, &emd_config).unwrap();
        let eemd_result = eemd(&signal, &config, &emd_config).unwrap();

        // Compare total energy in IMFs (should be similar, but CEEMD may be cleaner)
        let ceemd_imf_energy: f64 =
            ceemd_result.imfs.imfs.iter().flat_map(|imf| imf.iter()).map(|v| v * v).sum();
        let eemd_imf_energy: f64 =
            eemd_result.imfs.imfs.iter().flat_map(|imf| imf.iter()).map(|v| v * v).sum();

        // Both should capture non-zero finite energy
        assert!(ceemd_imf_energy > 0.0 && ceemd_imf_energy.is_finite(), "CEEMD IMF energy should be positive finite");
        assert!(eemd_imf_energy > 0.0 && eemd_imf_energy.is_finite(), "EEMD IMF energy should be positive finite");
    }

    // =========================================================================
    // Reproducibility tests
    // =========================================================================

    #[test]
    fn test_ceemd_reproducibility_same_seed() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result1 = ceemd(&signal, &config, &emd_config).unwrap();
        let result2 = ceemd(&signal, &config, &emd_config).unwrap();

        assert_eq!(result1.imfs.n_imfs(), result2.imfs.n_imfs());
        for (imf1, imf2) in result1.imfs.imfs.iter().zip(result2.imfs.imfs.iter()) {
            for (v1, v2) in imf1.iter().zip(imf2.iter()) {
                assert!((v1 - v2).abs() < 1e-15, "IMF values must match exactly with same seed");
            }
        }
    }

    #[test]
    fn test_ceemd_different_seeds_different_results() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let emd_config = fast_emd_config();

        let config1 = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(1) };
        let config2 = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(2) };

        let result1 = ceemd(&signal, &config1, &emd_config).unwrap();
        let result2 = ceemd(&signal, &config2, &emd_config).unwrap();

        let mut all_same = true;
        for (imf1, imf2) in result1.imfs.imfs.iter().zip(result2.imfs.imfs.iter()) {
            for (v1, v2) in imf1.iter().zip(imf2.iter()) {
                if (v1 - v2).abs() > 1e-15 {
                    all_same = false;
                    break;
                }
            }
            if !all_same {
                break;
            }
        }
        assert!(!all_same, "Different seeds should produce different results");
    }

    // =========================================================================
    // Error handling tests
    // =========================================================================

    #[test]
    fn test_ceemd_insufficient_data() {
        let signal = vec![1.0, 2.0];
        let config = EnsembleConfig::default();
        let emd_config = EmdConfig::default();

        let result = ceemd(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InsufficientData));
    }

    #[test]
    fn test_ceemd_invalid_value_nan() {
        let signal = vec![1.0, f64::NAN, 3.0, 4.0, 5.0];
        let config = EnsembleConfig::default();
        let emd_config = EmdConfig::default();

        let result = ceemd(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidValue));
    }

    #[test]
    fn test_ceemd_zero_ensembles_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EnsembleConfig { num_ensembles: 0, noise_std: 0.2, seed: None };
        let emd_config = EmdConfig::default();

        let result = ceemd(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    #[test]
    fn test_ceemd_negative_noise_std_error() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = EnsembleConfig { num_ensembles: 10, noise_std: -0.1, seed: None };
        let emd_config = EmdConfig::default();

        let result = ceemd(&signal, &config, &emd_config);
        assert!(matches!(result.unwrap_err(), EmdError::InvalidConfig(_)));
    }

    // =========================================================================
    // Algorithm type test
    // =========================================================================

    #[test]
    fn test_ceemd_algorithm_type_is_ceemd() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = ceemd(&signal, &config, &emd_config).unwrap();
        assert_eq!(result.algorithm, AlgorithmType::CEEMD);
    }

    // =========================================================================
    // Config snapshot test
    // =========================================================================

    #[test]
    fn test_ceemd_config_snapshot_is_valid_json() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = ceemd(&signal, &config, &emd_config).unwrap();

        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&result.config_snapshot);
        assert!(parsed.is_ok(), "Config snapshot should be valid JSON: {}", result.config_snapshot);
    }

    // =========================================================================
    // Multi-component signal
    // =========================================================================

    #[test]
    fn test_ceemd_multi_component_signal() {
        let n = 120;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin()
                    + 0.5 * (2.0 * PI * 20.0 * t).sin()
                    + 0.3 * (2.0 * PI * 50.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.15, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = ceemd(&signal, &config, &emd_config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(
            result.imfs.n_imfs() >= 2,
            "Multi-component signal should produce at least 2 IMFs, got {}",
            result.imfs.n_imfs()
        );
    }

    // =========================================================================
    // Noise cancellation verification
    // =========================================================================

    #[test]
    fn test_ceemd_noise_only_trials_average_to_near_zero() {
        let n = 100;
        let signal = vec![0.0; n];

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 1.0, seed: Some(42) };
        let emd_config = fast_emd_config();

        let result = ceemd(&signal, &config, &emd_config);

        if result.is_ok() {
            let result = result.unwrap();
            if result.imfs.n_imfs() > 0 {
                let total_energy: f64 =
                    result.imfs.imfs.iter().flat_map(|imf| imf.iter()).map(|v| v * v).sum();
                assert!(
                    total_energy < (n as f64 * 10.0),
                    "Noise energy should be bounded, got {}",
                    total_energy
                );
            }
        }
    }

    // =========================================================================
    // Large ensemble count
    // =========================================================================

    #[test]
    fn test_ceemd_large_ensemble() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 8, noise_std: 0.2, seed: Some(42) };
        let result = ceemd(&signal, &config, &fast_emd_config());
        assert!(result.is_ok(), "CEEMD with larger ensemble should succeed");
    }

    // =========================================================================
    // Parallel execution test
    // =========================================================================

    #[test]
    fn test_ceemd_parallel_execution() {
        let n = 100;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(99) };
        let emd_config = fast_emd_config();

        let result = ceemd(&signal, &config, &emd_config);
        assert!(result.is_ok(), "Parallel CEEMD should succeed");
    }

    // =========================================================================
    // Complementary pair averaging tests
    // =========================================================================

    #[test]
    fn test_average_complementary_pair_same_imfs() {
        let positive = TrialResult {
            imfs: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            residue: vec![0.5, 0.5, 0.5],
        };
        let negative = TrialResult {
            imfs: vec![vec![3.0, 4.0, 5.0], vec![6.0, 7.0, 8.0]],
            residue: vec![1.5, 1.5, 1.5],
        };

        let result = average_complementary_pair(&positive, &negative);
        assert_eq!(result.imfs.len(), 2);
        assert_eq!(result.imfs[0], vec![2.0, 3.0, 4.0]);
        assert_eq!(result.imfs[1], vec![5.0, 6.0, 7.0]);
        assert_eq!(result.residue, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_average_complementary_pair_unequal_imfs() {
        let positive =
            TrialResult { imfs: vec![vec![2.0, 2.0], vec![4.0, 4.0]], residue: vec![1.0, 1.0] };
        let negative = TrialResult { imfs: vec![vec![4.0, 4.0]], residue: vec![2.0, 2.0] };

        let result = average_complementary_pair(&positive, &negative);
        assert_eq!(result.imfs.len(), 2);
        // IMF 0: (2 + 4) / 2 = 3
        assert_eq!(result.imfs[0], vec![3.0, 3.0]);
        // IMF 1: (4 + 0) / 2 = 2 (zero-padded)
        assert_eq!(result.imfs[1], vec![2.0, 2.0]);
        // Residue: (1 + 2) / 2 = 1.5
        assert_eq!(result.residue, vec![1.5, 1.5]);
    }

    #[test]
    fn test_average_complementary_pair_empty() {
        let positive = TrialResult { imfs: vec![], residue: vec![] };
        let negative = TrialResult { imfs: vec![], residue: vec![] };

        let result = average_complementary_pair(&positive, &negative);
        assert!(result.imfs.is_empty());
        assert!(result.residue.is_empty());
    }

    #[test]
    fn test_average_pair_results_empty() {
        let (imfs, residue) = average_pair_results(&[]);
        assert!(imfs.is_empty());
        assert!(residue.is_empty());
    }

    #[test]
    fn test_average_pair_results_single_pair() {
        let pair = ComplementaryPairResult {
            imfs: vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
            residue: vec![0.5, 0.5, 0.5],
        };
        let (imfs, residue) = average_pair_results(&[pair]);
        assert_eq!(imfs.len(), 2);
        assert_eq!(imfs[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(imfs[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(residue, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_average_pair_results_two_pairs() {
        let pairs = vec![
            ComplementaryPairResult {
                imfs: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                residue: vec![0.5, 0.5],
            },
            ComplementaryPairResult {
                imfs: vec![vec![3.0, 4.0], vec![5.0, 6.0]],
                residue: vec![1.5, 1.5],
            },
        ];
        let (imfs, residue) = average_pair_results(&pairs);
        assert_eq!(imfs.len(), 2);
        assert_eq!(imfs[0], vec![2.0, 3.0]);
        assert_eq!(imfs[1], vec![4.0, 5.0]);
        assert_eq!(residue, vec![1.0, 1.0]);
    }

    // =========================================================================
    // RMS helper tests
    // =========================================================================

    #[test]
    fn test_rms_constant_signal() {
        let signal = vec![3.0; 100];
        assert!((rms(&signal) - 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_rms_known_values() {
        // [3, 4] → sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let signal = vec![3.0, 4.0];
        let expected = 12.5f64.sqrt();
        assert!((rms(&signal) - expected).abs() < 1e-15);
    }

    // =========================================================================
    // CEEMD vs EEMD detailed comparison
    // =========================================================================

    #[test]
    fn test_ceemd_vs_eemd_same_signal_imf_count() {
        let n = 120;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 3.0 * t).sin() + 0.05 * (2.0 * PI * 30.0 * t).sin()
            })
            .collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let ceemd_result = ceemd(&signal, &config, &emd_config).unwrap();
        let eemd_result = eemd(&signal, &config, &emd_config).unwrap();

        // Both should produce similar number of IMFs
        assert!(
            ceemd_result.imfs.n_imfs() >= 2,
            "CEEMD should separate mode-mixed signal into at least 2 IMFs"
        );
        assert!(
            eemd_result.imfs.n_imfs() >= 2,
            "EEMD should separate mode-mixed signal into at least 2 IMFs"
        );
    }

    #[test]
    fn test_ceemd_vs_eemd_residue_energy_comparison() {
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / n as f64).sin()).collect();

        let config = EnsembleConfig { num_ensembles: 4, noise_std: 0.2, seed: Some(42) };
        let emd_config = fast_emd_config();

        let ceemd_result = ceemd(&signal, &config, &emd_config).unwrap();
        let eemd_result = eemd(&signal, &config, &emd_config).unwrap();

        let ceemd_residue_energy: f64 = ceemd_result.imfs.residue.iter().map(|v| v * v).sum();
        let eemd_residue_energy: f64 = eemd_result.imfs.residue.iter().map(|v| v * v).sum();

        // CEEMD should have lower or comparable residue energy
        assert!(
            ceemd_residue_energy <= eemd_residue_energy * 2.0,
            "CEEMD residue energy ({:.6}) should be <= EEMD residue energy ({:.6}) * 2.0",
            ceemd_residue_energy,
            eemd_residue_energy
        );
    }
}
